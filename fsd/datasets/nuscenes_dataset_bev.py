"""Nuscenes Dataset for 3D object detection used by BEVFormer.
"""

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
from mmdet3d.structures import Det3DDataSample
from fsd.registry import DATASETS
from .nuscenes_dataset import NuScenesDatasetPlan3D
from .eval_utils.nuscenes_eval_bev import NuScenesEvalBEVFormer

@DATASETS.register_module()
class NuScenesDatasetBEVFormer(NuScenesDatasetPlan3D):
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
        
    def _update_can_bus_info(self, input_dict: dict) -> dict:
        """Update can bus info for the current sample to follow VAD paper.
        
        - remove (steer, throttle, brake) from original can bus
        - add yaw in radians and yaw in degrees
        
        Args:
            input_dict (dict): Raw info dict.
        """
        # get can bus info
        can_bus = copy.deepcopy(input_dict['ego_can_bus'])[:-3]
        
        # add yaw in radians and yaw in degrees
        yaw = quaternion_yaw(Quaternion(can_bus[3:7]))
        # the original paper did this, so ...
        if yaw < 0:
            yaw += 2 * np.pi
        yaw_degree = yaw / np.pi * 180
        can_bus = np.concatenate([can_bus, [yaw, yaw_degree]])
        
        input_dict['ego_can_bus'] = can_bus
        return input_dict 
    
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
            
    def parse_ann_info(self, info: dict) -> dict:
        """Process the raw annotation info.

        convert the box convention from MMDET3D to kitti/SECOND box convention
        """
        ann_info = super().parse_ann_info(info)
        
        ## BEVFormer original code uses SECOND box convention
        gt_bboxes_3d_data = ann_info['gt_bboxes_3d'].tensor.clone()
        # lwh to wlh 
        gt_bboxes_3d_data[:, 3:6] = gt_bboxes_3d_data[:, [4, 3, 5]]
        # MMDET3D yaw definition to KITTI/SECOND box yaw definition
        gt_bboxes_3d_data[:, 6] = -gt_bboxes_3d_data[:, 6] - np.pi/2
        gt_bboxes_3d_KITTI = ann_info['gt_bboxes_3d'].new_box(
            data = gt_bboxes_3d_data,
        )
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d_KITTI
        return ann_info
    
    def _prepare_data(self, index) -> dict:
        """Prepare data given sample index.
        """
        # get data info
        input_dict = self.get_data_info(index)
        if not input_dict:
            return None
                            
        # update can bus for bev use
        if self.with_can_bus:
            input_dict = self._update_can_bus_info(input_dict)
        
        return input_dict
    
    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = [i for i in range(index - self.queue_length, index)]
        index_list = np.random.choice(index_list, size=self.queue_length, replace=False)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        
        for idx in index_list:
            # in case out of range 
            idx = max(0, idx)
            
            # prepare data
            input_dict = self._prepare_data(idx)
            if input_dict is None:
                return None
            
            # assemble for data pipeline
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            if self.filter_empty_gt and \
                    (example is None or
                        ~(example['data_samples'].gt_instances_3d.label != -1).any()):
                return None

            queue.append(example)

        return self._combine_history_for_bev(queue)
    
    def _combine_history_for_bev(self, queue):
        """Combine historical frames for BEV
        """
        imgs_list = [data['inputs']['img'] for data in queue]
        
        prev_scene_token = None
        prev_pos = None
        prev_yaw = None
        bev_metas = [{} for _ in range(len(queue))]
        
        # calculate the delta orientation and position for adjacent frames for BEV
        for idx, data in enumerate(queue):
            bev_metas[idx] = data['data_samples'].metainfo
            if bev_metas[idx]['scene_token'] != prev_scene_token:
                # new scene
                prev_scene_token = bev_metas[idx]['scene_token']
                bev_metas[idx]['prev_bev_exists'] = False
                bev_attr = None
                if self.with_can_bus:
                    bev_attr = copy.deepcopy(bev_metas[idx]['ego_can_bus'])
                    prev_pos = copy.deepcopy(bev_attr[0:3]) # in world frame
                    prev_yaw = float(bev_attr[-2]/np.pi * 180) # radians to degree
                    bev_attr[0:3] = 0
                    bev_attr[-1] = 0 # BEVFormer uses yaw in radians and yaw degree in can_bus
            else:
                bev_metas[idx]['prev_bev_exists'] = True
                if self.with_can_bus:
                    # get the previous can_bus
                    bev_attr = copy.deepcopy(bev_metas[idx]['ego_can_bus'])
                    temp_pos = copy.deepcopy(bev_attr[0:3])
                    temp_yaw = float(bev_attr[-2]/np.pi * 180)
                    bev_attr[0:3] -= prev_pos
                    bev_attr[-1] = temp_yaw - prev_yaw
                    prev_pos = temp_pos
                    prev_yaw = temp_yaw
                    
            bev_metas[idx]['bev_attr'] = bev_attr
            
        # assemble
        new_data = {}
        new_data['inputs'] = {}
        new_data['inputs']['img'] = torch.stack(imgs_list, dim=0) # [seq_len,N, 3, H, W] 
        
        data_samples = queue[-1]['data_samples'].clone()
        data_samples.set_metainfo({'bev_metas': bev_metas})
        new_data['data_samples'] = data_samples
        
        return new_data
    
    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        # prepare data
        input_dict = self._prepare_data(index)
        if input_dict is None:
            return None
        
        # pipeline
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        
        # add bev_attr for test
        bev_metas = [{}]
        if self.with_can_bus:
            bev_metas[0] = example['data_samples'].metainfo
            bev_attr = copy.deepcopy(example['data_samples'].metainfo['ego_can_bus'])
            bev_metas[0]['bev_attr'] = bev_attr
            example['data_samples'].set_metainfo({'bev_metas': bev_metas})
            
        return example
 
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
