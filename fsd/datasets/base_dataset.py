# Copyright (c) OpenMMLab. All rights reserved. 
from abc import abstractmethod
import copy
import logging
import numpy as np
import tempfile
import warnings
import os
from os import path as osp
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from collections.abc import Mapping
from terminaltables import AsciiTable

#from mmcv.datasets.builder import DATASETS
from mmengine.config import Config
from mmengine.fileio import load, dump, list_from_file
from mmengine.logging import print_log
from mmengine.dataset import Compose, BaseDataset
from mmdet3d.structures import (get_box_type, LiDARInstance3DBoxes, 
                                DepthInstance3DBoxes, CameraInstance3DBoxes, 
                                BaseInstance3DBoxes)
from fsd.structures import TrajectoryData
from fsd.registry import DATASETS


@DATASETS.register_module()
class BasePlanDataset(BaseDataset):
    """Base Class for 3D planning dataset.
    """
    # dataset metainfo
    METAINFO = {}
    
    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(pts='velodyne', img=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=False, use_camera=True),
                 box_type_3d_original: str = 'Depth', # box cooridnate in the original annotation file
                 box_type_3d: str = 'LiDAR', # targeted box coordinate for the dataset
                 filter_empty_gt: bool = True,
                 past_steps: int = 4, # past trajectory length
                 prediction_steps: int = 6, # motion prediction length if any
                 planning_steps: int = 6, # planning length
                 sample_interval: int = 1, # sample interval # frames skiped per step
                 FPS: int = 2, # frame per second
                 test_mode: bool = False,
                 load_eval_anns: bool = True,
                 show_ins_var: bool = False,
                 with_goal_points: bool = True,
                 **kwargs) -> None:
        
        """Initialize the dataset.
        
        Args:
            data_root (str): Root directory path of the dataset.
            ann_file (str): Path to the annotation file.
            metainfo (dict): Metainfo of the dataset.
            data_prefix (dict): Prefix of the dataset.
                - pts (str): Prefix of point cloud data.
                - img (str): Prefix of image data.
            pipeline (list[dict | callable]): Processing pipeline.
            modality (dict): Modality of the dataset.
                - use_lidar (bool): Whether to use lidar data.
                - use_camera (bool): Whether to use camera data.
            camera_sensors (list[str]): List of camera sensors.
            lidar_sensors (list[str]): List of lidar sensors.
            box_type_3d_original (str): Box type of MMDet3D boxes in the original annotation file.
                - Depth: depth coordinate
                - Lidar: lidar coordinate
                - Camera: camera coordinate
            box_type_3d (str): targeted box type of MMDet3D for the dataset.
                - Depth: depth coordinate
                - Lidar: lidar coordinate
                - Camera: camera coordinate
            filter_empty_gt (bool): Whether to filter empty ground truth. Usually used after pipeline.
            past_steps (int): Number of past steps for trajectory generation.
            prediction_steps (int): Number of prediction steps for agent trajectory generation.
            planning_steps (int): Number of planning steps for ego trajectory generation.
            sample_interval (int): Sample interval for trajectory generation.
            FPS (int): Frame per second in original data
            test_mode (bool): Whether the dataset is in test mode.
            load_eval_anns (bool): Whether to load evaluation annotations.
            show_ins_var (bool): Whether to show instance variation.
            with_goal_points (bool): Whether to add goal points to trajectory.
        
        """ 
        self.filter_empty_gt = filter_empty_gt
        self.load_eval_anns = load_eval_anns
        
        # past and future frames
        self.past_steps = past_steps
        self.prediction_steps = prediction_steps
        self.planning_steps = planning_steps
        self.sample_interval = sample_interval
        self.FPS = FPS
        
        # add goal points to trajectory
        self.with_goal_points = with_goal_points
        
        # modality
        _default_modality_keys = ('use_lidar', 'use_camera')
        if modality is None:
            modality = dict()
        for key in _default_modality_keys:
            if key not in modality:
                modality[key] = False
        self.modality = modality
        assert self.modality['use_lidar'] or self.modality['use_camera'], (
            'Please specify the `modality` (`use_lidar` '
            f', `use_camera`) for {self.__class__.__name__}')        
        
        # boxes
        self.box_type_3d_source = box_type_3d_original
        self.box_type_3d_target = box_type_3d
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        

        # initialize the dataset
        # class names override by providing a new class list in metainfo
        # label_mapping {original_label: new_label}
        if metainfo is not None and 'classes' in metainfo:
            # map unselected classes to -1
            self.label_mapping = {
                i: -1
                for i in range(len(self.METAINFO['classes']))
                
            }
            self.label_mapping[-1] = -1
            for label_idx, name in enumerate(metainfo['classes']):
                ori_label = self.METAINFO['classes'].index(name)
                self.label_mapping[ori_label] = label_idx
                
            self.num_ins_per_cat = [0] * len(metainfo['classes'])
        else:
            self.label_mapping = {
                i: i 
                for i in range(len(self.METAINFO['classes']))
            }
            self.label_mapping[-1] = -1
            self.num_ins_per_cat = [0] * len(self.METAINFO['classes'])
            
        super(BasePlanDataset, self).__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)
        
        # can be accessed by other components in the runner
        self.metainfo['label_mapping'] = self.label_mapping
        self.metainfo['box_type_3d'] = self.box_type_3d
        
        # full initialization        
        if not kwargs.get('lazy_init', False):
            # used for showing variation of the number of instances before and
            # after through the pipeline
            self.show_ins_var = show_ins_var
                        
            # show statistics of this dataset
            print_log('-' * 30, 'current')
            print_log(
                f'The length of {"test" if self.test_mode else "training"} dataset: {len(self)}',  # noqa: E501
                'current')
            content_show = [['category', 'number']]
            for label, num in enumerate(self.num_ins_per_cat):
                cat_name = self.metainfo['classes'][label]
                content_show.append([cat_name, num])
            table = AsciiTable(content_show)
            print_log(
                f'The number of instances per category in the dataset:\n{table.table}',  # noqa: E501
                'current')
    
    #def get_data_info(self, index):
    #    return super().get_data_info(index)
    #
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Use index to get the corresponding annotations, thus the
        evalhook could use this api.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information.
        """
        data_info = self.get_data_info(index)
        # test model
        if 'ann_info' not in data_info:
            ann_info = self.parse_ann_info(data_info)
        else:
            ann_info = data_info['ann_info']

        return ann_info

    def _filter_with_mask(self, ann_info):
        """Filter the annotation info with the mask.
        
        Args:
            ann_info (dict): Annotation information.
                - gt_bboxes_3d (np.ndarray): 3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_instances_names (list[str]): Class names of ground truths.
                - gt_instances_ids (np.ndarray): IDs of ground truths.
                - gt_bboxes_mask (np.ndarray): Mask of ground truths.
                - gt_bboxes_anno_token (np.ndarray): Annotation token of ground truths.
                - gt_bboxes_velocity (np.ndarray): Velocity of ground truths.
                - gt_bboxes_id (np.ndarray): ID of ground truths.
        Returns:
            dict: Filtered annotation information.
        """
        filtered_ann_info = {}
        if 'gt_bboxes_mask' in ann_info:
            filter_mask = ann_info['gt_bboxes_mask']
        else:
            filter_mask = np.ones_like(ann_info['gt_bboxes_3d'], dtype=bool)
        
        # filter all anno info
        for key in ann_info.keys():
            if isinstance(ann_info[key], np.ndarray) or isinstance(ann_info[key], BaseInstance3DBoxes):
                filtered_ann_info[key] = ann_info[key][filter_mask]
            elif isinstance(ann_info[key], list):
                filtered_ann_info[key] = [item[filter_mask] for item in ann_info[key]]
            else:
                filtered_ann_info[key] = ann_info[key]
        
        return filtered_ann_info
        
    def parse_ann_info(self, info):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_instances_names (list[str]): Class names of ground truths.
        """
        num_bboxes = len(info['instances'])
        # empty gt
        if num_bboxes == 0:
            return None

        gt_bboxes_3d = np.array([instance['bbox_3d'] for instance in info['instances']]).astype(np.float32)
        gt_bboxes_mask = np.array([instance['bbox_3d_isvalid'] for instance in info['instances']]).astype(np.bool_)
        gt_bboxes_velocity = np.array([instance['velocity'] for instance in info['instances']]).astype(np.float32)
        gt_bboxes_id = np.array([instance['id'] for instance in info['instances']]).astype(np.str_)
        #gt_bboxes_pose = np.array([instance['pose'] for instance in info['instances']]).astype(np.float32)
        
        # labels as int might change due to user-defined mapping
        gt_labels_3d = [instance['bbox_label_3d'] for instance in info['instances']]
        gt_labels_3d = np.array([self.label_mapping[label] for label in gt_labels_3d]).astype(np.int64)

        # sample_annotation token in nuscenes
        if num_bboxes > 0 and 'annotation_token' in info['instances'][0]:
            gt_bboxes_anno_token = np.array([instance['annotation_token'] for instance in info['instances']])
        
        # bboxes traj
        gt_bboxes_future_trajectory = np.array([instance['future_trajectory'] for instance in info['instances']]).astype(np.float32)
        gt_bboxes_future_yaws = np.array([instance['future_yaw'] for instance in info['instances']]).astype(np.float32)
        gt_bboxes_future_masks = np.array([instance['future_mask'] for instance in info['instances']]).astype(np.bool_)
        gt_bboxes_goal = np.array([instance['goal'] for instance in info['instances']]).astype(np.float32)
        
        # ego related 
        gt_ego_future_trajs = np.array(info['ego']['future_trajectory']).astype(np.float32)
        gt_ego_future_yaws = np.array(info['ego']['future_yaw']).astype(np.float32)
        gt_ego_future_masks = np.array(info['ego']['future_mask']).astype(np.bool_)
    
        # box type conversion
        if self.box_type_3d_source.lower() == 'depth':
            BoxInstance = DepthInstance3DBoxes
        elif self.box_type_3d_source.lower() == 'lidar':
            BoxInstance = LiDARInstance3DBoxes
        elif self.box_type_3d_source.lower() == 'camera':
            BoxInstance = CameraInstance3DBoxes
        else:
            raise ValueError(f"Unknown box type {self.box_type_3d_source}")
        
        self.to_mmdet3d_lidar = self._get_to_mmdet3d_lidar(self.box_type_3d_source, self.box_type_3d_target)
        
        ## ==================================================
        #TODO: this is hard-coded for nuscenes lidar to mmdet3d lidar (i.e., mmdet3d depth to mmdet3d lidar)
        # need implement the transformation based on transformation matrix
        if self.to_mmdet3d_lidar:
            # convert boxes
            # x_mmdet = y_nuscenes, y_mmdet = -x_nuscenes, z_mmdet = z_nuscenes
            #gt_bboxes_3d[:, [0, 1]] = gt_bboxes_3d[:, [1, 0]]
            #gt_bboxes_3d[:, 1] = -gt_bboxes_3d[:, 1]
            # yaw: yaw_mmdet = yaw_nuscenes - pi/2
            #gt_bboxes_3d[:, 6] -= np.pi / 2
            #gt_bboxes_3d[:, 6] = self._limit_yaw(gt_bboxes_3d[:, 6])
            # velocity
            #gt_bboxes_velocity[..., [0, 1]] = gt_bboxes_velocity[..., [1, 0]]
            #gt_bboxes_velocity[..., 1] = -gt_bboxes_velocity[..., 1]
            
            # convert trajectory yaw
            gt_bboxes_future_trajectory[..., [0, 1]] = gt_bboxes_future_trajectory[..., [1, 0]]
            gt_bboxes_future_trajectory[..., 1] = -gt_bboxes_future_trajectory[..., 1]
            gt_bboxes_future_yaws -= np.pi / 2
            gt_bboxes_future_yaws = self._limit_yaw(gt_bboxes_future_yaws)
            # convert goal
            gt_bboxes_goal[..., [0, 1]] = gt_bboxes_goal[..., [1, 0]]
            gt_bboxes_goal[..., 1] = -gt_bboxes_goal[..., 1]
            
            # ego 
            gt_ego_future_trajs[..., [0, 1]] = gt_ego_future_trajs[..., [1, 0]]
            gt_ego_future_trajs[..., 1] = -gt_ego_future_trajs[..., 1]
            gt_ego_future_yaws -= np.pi / 2
            gt_ego_future_yaws = self._limit_yaw(gt_ego_future_yaws)
            
            # convert to mmdet3d lidar coordinate
            # lidar2ego
            if 'lidar2ego' in info['lidar_points']:
                info['lidar_points']['lidar2ego'] = (np.array(info['lidar_points']['lidar2ego']) @ np.linalg.inv(self.to_mmdet3d_lidar)).tolist()
            
            # lidarcam
            for cam in info['images'].keys():
                if 'lidar2cam' in info['images'][cam]:
                    info['images'][cam]['lidar2cam'] = (np.array(info['images'][cam]['lidar2cam']) @ np.linalg.inv(self.to_mmdet3d_lidar)).tolist()

        # add velocity to gt_bboxes_3d
        gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_bboxes_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = BoxInstance(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        
        # planning annotations for future steps
        # (num_boxes, fut, 4) : (x, y, z, yaw)
        gt_bboxes_traj = np.concatenate(
            [gt_bboxes_future_trajectory, gt_bboxes_future_yaws[:, :, None]], 
            axis=-1)
        
        # construct anno info
        ann_info = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_bboxes_mask=gt_bboxes_mask,
            gt_bboxes_id=gt_bboxes_id,
            gt_bboxes_traj=gt_bboxes_traj,
            gt_bboxes_traj_mask=gt_bboxes_future_masks,
            gt_bboxes_goal=gt_bboxes_goal,
            )
        if gt_bboxes_anno_token is not None:
            ann_info['gt_bboxes_anno_token'] = gt_bboxes_anno_token
        
        # filter with mask
        ann_info = self._filter_with_mask(ann_info)
        
        # category statistics
        for label in ann_info['gt_labels_3d']:
            if label != -1:
                self.num_ins_per_cat[label] += 1
        
        # add ego annotation
        gt_ego_traj = np.concatenate(
            [gt_ego_future_trajs, gt_ego_future_yaws[:, None]], 
            axis=-1)
        
        ann_info.update(
            gt_ego_traj=gt_ego_traj,
            gt_ego_traj_mask=gt_ego_future_masks,
        )
        
        return ann_info
    
    def get_map_info(self, info):
        """Get map data info from the given info. 
        """
        pass

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path. And process the `instances` field to
        `ann_info` in training stage.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info = copy.deepcopy(info)
        if self.modality['use_lidar']:
            info['lidar_points']['lidar_path'] = \
                osp.join(
                    self.data_prefix.get('pts', ''),
                    info['lidar_points']['lidar_path'])

            info['num_pts_feats'] = info['lidar_points']['num_pts_feats']
            info['lidar_path'] = info['lidar_points']['lidar_path']
            if 'lidar_sweeps' in info:
                for sweep in info['lidar_sweeps']:
                    file_suffix = sweep['lidar_points']['lidar_path'].split(
                        os.sep)[-1]
                    if 'samples' in sweep['lidar_points']['lidar_path']:
                        sweep['lidar_points']['lidar_path'] = osp.join(
                            self.data_prefix['pts'], file_suffix)
                    else:
                        sweep['lidar_points']['lidar_path'] = osp.join(
                            self.data_prefix['sweeps'], file_suffix)

        if self.modality['use_camera']:
            for cam_id, img_info in info['images'].items():
                if 'img_path' in img_info:
                    if cam_id in self.data_prefix:
                        cam_prefix = self.data_prefix[cam_id]
                    else:
                        cam_prefix = self.data_prefix.get('img', '')
                    img_info['img_path'] = osp.join(cam_prefix,
                                                    img_info['img_path'])

        # parse ego information
        # ego annotation will be parsed in parse_ann_info
        if 'ego' in info:
            ego_keys = ['ego_size', 'ego_velocity', 
                        'ego_yaw_velocity', 'ego_goal', 
                        'ego_can_bus', 'ego_command',
                        'ego_history_trajectory', 'ego_history_yaw',
                        'ego_history_mask'
                        ]
            for key in ego_keys:
                # remove prefix ego
                if key[4:] in info['ego']:
                    if 'mask' in key:
                        info[key] = np.array(info['ego'][key[4:]]).astype(np.bool_)
                    else:
                        info[key] = np.array(info['ego'][key[4:]]).astype(np.float32)
        
        # parse map information
        if 'map' in info:
            map_keys = ['map_location']
            for key in map_keys:
                info[key] = info['map'][key[4:]]
            
        # parse annoation information
        if not self.test_mode:
            # used in training
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['ann_info'] = self.parse_ann_info(info)
            info['eval_ann_info'] = info['ann_info']
                                
        return info

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.
                - img_fields (list[str]): Image fields, inlcuding 'img', 'img_filename', etc
                - pts_fields (list[str]): Point fields, including 'pts', 'pts_filename', etc
                - ego_fields (list[str]): Ego fields, including 
                        'gt_ego_traj', 'ego_world2ego', 'ego_velocity', 'ego_affected_by_lights', 'ego_affected_by_stop_sign', 'ego_is_at_junction', etc
                - bbox3d_fields (list[str]): 3D bbox fields, including 'gt_bboxes_3d', 'gt_labels_3d', 'gt_classes', etc
                - pts_seg_fields (list[str]): Point cloud segmentation fields.
                - grid_fields (list[str]): Grid fields, including "gt_grid_density", "gt_grid_occupancy", etc
        """
        
        results['img_fields'] = []
        results['pts_fields'] = []
        results['ego_fields'] = [] # ['gt_ego_traj']
        results['map_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_seg_fields'] = []
        results['grid_fields'] = [] 
        results['bbox_fields'] = []
        results['img_seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        # get data info
        input_dict = self.get_data_info(index)
        if not input_dict:
            return None
        
        # assemble for data pipeline
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['data_samples'].gt_instances_3d.label != -1).any()):
            return None
        
        # lidar poitns: convert to mmdet3d lidar 
        if 'points' in example['inputs']:
            points = example['inputs']['points']
            if self.TO_MMDET3D_LIDAR is not None:
                points[:, 0] = points[:, 1]
                points[:, 1] = -points[:, 0]
            example['inputs']['points'] = points
        
        return example

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        # assemble for data pipeline
        input_dict = self.get_data_info(index)
        if not input_dict:
            return None

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['data_samples'].gt_instances_3d.label != -1).any()):
            return None
        
        # lidar poitns: convert to mmdet3d lidar 
        if 'points' in example['inputs']:
            points = example['inputs']['points']
            if self.to_mmdet3d_lidar is not None:
                points[:, [0, 1]] = points[:, [1, 0]]
                points[:, 1] = -points[:, 1]
            example['inputs']['points'] = points
            
        return example

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results, \
                tmp_dir is the temporal directory created for saving json \
                files when ``jsonfile_prefix`` is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
            out = f'{pklfile_prefix}.pkl'
        dump(outputs, out)
        return outputs, tmp_dir

    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str]): Metrics to be evaluated.
            iou_thr (list[float]): AP IoU thresholds.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        from mmcv.core.evaluation import indoor_eval
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'
        gt_annos = [info['annos'] for info in self.data_infos]
        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}
        ret_dict = indoor_eval(
            gt_annos,
            results,
            iou_thr,
            label2cat,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d)
        if show:
            self.show(results, out_dir, pipeline=pipeline)

        return ret_dict

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
                idx = self._rand_another()
                continue
            return data

    def _limit_yaw(self, yaw):
        """Limit the yaw to be in the range of [-pi, pi].
        
        Args:
            yaw (float): Yaw angle.
        
        Returns:
            float: Limited yaw angle.
        """
        return (yaw + np.pi) % (2 * np.pi) - np.pi

    @classmethod
    def _get_to_mmdet3d_lidar(cls, box_source: str, box_target: str):
        """Get the transformation matrix from box_source to box_target.
        
        Args:
            box_source (str): Source box type.
            box_target (str): Target box type.
        
        Returns:
            np.ndarray: Transformation matrix from source to target.
        """
        assert box_target.upper() ==  'LIDAR', \
            f'box_target should be LIDAR, but got {box_target}'
        if box_source.upper() == 'LIDAR' and box_target.upper() == 'LIDAR':
            return None
        elif box_source.upper() == 'DEPTH' and box_target.upper() == 'LIDAR':
            return [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        elif box_source.upper() == 'CAMERA' and box_target.upper() == 'LIDAR':
            return [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
        else:
            raise ValueError(f"Unknown box type {box_source} to {box_target}")