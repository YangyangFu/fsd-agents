import copy
import logging
import random
import pickle
import numpy as np
import torch
from os import path as osp

from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
import mmcv

from mmengine.logging import print_log
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.structures import Det3DDataSample
from fsd.registry import DATASETS
from .nuscenes_eval_bev import NuScenesEvalBEVFormer

@DATASETS.register_module()
class NuScenesDatasetBEVFormer(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, 
                queue_length=4, 
                bev_size=(200, 200), 
                overlap_test=False, 
                use_can_bus=True,
                *args, 
                **kwargs):
        
        """
        Initializes the NuScenesDatasetBEV class.
        Args:
            queue_length (int, optional): The length of the queue. Defaults to 4. Only useful for training.
            bev_size (tuple, optional): The size of the bird's eye view. Defaults to (200, 200).
            overlap_test (bool, optional): Flag to indicate if overlap test is enabled. Defaults to False.
            use_can_bus (bool, optional): Flag to indicate if CAN bus data should be used. Defaults to True.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.use_can_bus = use_can_bus
        
        # Nuscenes for can bus info as a wrapper
        self.nusc = NuScenes(version=self.metainfo['version'], dataroot=self.data_root, verbose=True)
        if self.use_can_bus:
            self.can_bus = NuScenesCanBus(dataroot=self.data_root)

    def _get_can_bus_info(self, input_dict):
        """Get can_bus information given the sample token in the input_dict.
        """
        sample_token = input_dict['sample_token']
        sample = self.nusc.get('sample', sample_token)
        scene_token = sample['scene_token']
        scene_name = self.nusc.get('scene', scene_token)['name']
        sample_timestamp = sample['timestamp']
        
        # get can bus information
        try:
            pose_list = self.can_bus.get_messages(scene_name, 'pose')
        except:
            return np.zeros(18)  # server scenes do not have can bus information.
        can_bus = []
        # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
        last_pose = pose_list[0]
        for i, pose in enumerate(pose_list):
            if pose['utime'] > sample_timestamp:
                break
            last_pose = pose
        # first 16 elements
        pos = last_pose['pos']
        orientation = last_pose['orientation']
        can_bus.extend(pos)
        can_bus.extend(orientation)
        for key in ['accel', 'rotation_rate', 'vel']:
            can_bus.extend(pose[key])  
        # the last two numbers are reserved for later calculation of rotation angle.
        can_bus.extend([0., 0.])
        
        
        # update pose from calibrated sensor data
        rotation = Quaternion(matrix=input_dict['ego2global_rotation'], atol=1e-6)
        translation = input_dict['ego2global_translation']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        
        # save to input_dict
        input_dict.update(
            can_bus=np.array(can_bus),
            scene_token=scene_token,
        )

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d
            
    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        # random sample historical data for current frame
        queue = []
        index_list = list(range(index-self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if self.use_can_bus:
                self._get_can_bus_info(input_dict)
                
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            
            # add scene token to metainfo if not exists
            if 'scene_token' not in example['data_samples'].metainfo:
                example['data_samples'].set_field(name='scene_token', value=input_dict['scene_token'], field_type='metainfo', dtype=None)
            # add can bus information to metainfo
            if self.use_can_bus:
                example['data_samples'].set_field(name='can_bus', value=input_dict['can_bus'], field_type='metainfo', dtype=None)
            if self.filter_empty_gt:
                # after pipeline drop the example with empty annotations
                # return None to random another in `__getitem__`
                if example is None or len(
                        example['data_samples'].gt_instances_3d.labels_3d) == 0:
                    return None
            queue.append(example)
            
        return self.union2one(queue)


    def union2one(self, queue):
        imgs_list = [each['inputs']['img'] for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        
        # compute the delta orientation and position of adjacent frames
        for i, each in enumerate(queue):
            metas_map[i] = each['data_samples'].metainfo
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                if self.use_can_bus:
                    prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                    prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                    metas_map[i]['can_bus'][:3] = 0
                    metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                if self.use_can_bus:
                    tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3]) 
                    tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                    metas_map[i]['can_bus'][:3] -= prev_pos
                    metas_map[i]['can_bus'][-1] -= prev_angle
                    prev_pos = copy.deepcopy(tmp_pos)
                    prev_angle = copy.deepcopy(tmp_angle)
        queue[-1]["inputs"]['img'] = torch.stack(imgs_list) # (L, N, C, H, W)
        queue[-1]["img_metas"] = metas_map

        queue = queue[-1]
        
        return queue

    
    def prepare_test_data(self, index):
        """Prepare testing data."""
        input_dict = self.get_data_info(index)
        if self.use_can_bus:
            self._get_can_bus_info(input_dict)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)

        # add scene token to metainfo if not exists
        example['data_samples'].set_field(
            name='scene_token', 
            value=input_dict['scene_token'], 
            field_type='metainfo', 
            dtype=None)
        # add can bus information to metainfo
        if self.use_can_bus:
            example['data_samples'].set_field(
                name='can_bus', 
                value=input_dict['can_bus'], 
                field_type='metainfo', 
                dtype=None)
        
        # be consistent with the training data format, adding img_metas
        example["img_metas"] = copy.deepcopy(example["data_samples"].metainfo)
        
        return example
    
    # overwrite the get_data_info method
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_token (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        if self.serialize_data:
            start_addr = 0 if index == 0 else self.data_address[index - 1].item()
            end_addr = self.data_address[index].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            info = pickle.loads(bytes)  # type: ignore
        else:
            info = copy.deepcopy(self.data_list[index])

        # standard protocal
        input_dict = dict(
            sample_token=info['token'],
            pts_filename=info['lidar_points']['lidar_path'],
            sweeps=info.get('lidar_sweeps', []),
            ego2global_translation=np.array(info['ego2global'])[:3, -1], # 3
            ego2global_rotation=np.array(info['ego2global'])[:3, :3], # 3x3
            #prev_idx=info['prev'],
            #next_idx=info['next'],
            #scene_token=info['scene_token'],
            #can_bus=info['can_bus'],
            #frame_idx=info['frame_idx'],
            sample_idx=info['sample_idx'],
            timestamp=info['timestamp'],
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['images'].items():
                image_paths.append(cam_info['img_path'])
                # obtain lidar to image transformation matrix
                #lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                #lidar2cam_t = cam_info[
                #    'sensor2lidar_translation'] @ lidar2cam_r.T
                #lidar2cam_rt = np.eye(4)
                #lidar2cam_rt[:3, :3] = lidar2cam_r.T
                #lidar2cam_rt[3, :3] = -lidar2cam_t
                
                lidar2cam = np.array(cam_info['lidar2cam'])
                intrinsic = np.array(cam_info['cam2img'])
    
                viewpad = np.eye(4)
                viewpad[:3, :3] = intrinsic
                
                # lidar to image
                lidar2img = viewpad @ lidar2cam 
                #lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            #annos = self.get_ann_info(index)
            input_dict['ann_info'] = info['ann_info']

        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            print_log(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()
            
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root,
                             verbose=True)

        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        self.nusc_eval = NuScenesEvalBEVFormer(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            overlap_test=self.overlap_test,
            data_infos=self.data_infos
        )
        self.nusc_eval.main(plot_examples=0, render_curves=False)
        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail
