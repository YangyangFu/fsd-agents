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
from fsd.datasets.utils import extract_result_dict, get_loading_pipeline
from fsd.registry import DATASETS


@DATASETS.register_module()
class BasePlanDataset(BaseDataset):
    """Base Class for 3D planning dataset.
    """

    # transformation matrix from dataset lidar coordinate to mmdet3d lidar
    # default is identity matrix
    TO_MMDET3D_LIDAR = np.eye(4)
    METAINFO = {}
    
    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(pts='velodyne', img=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=False, use_camera=True),
                 camera_sensors: List[str] = ['CAM_FRONT'],
                 lidar_sensors: List[str] = ['LIDAR_TOP'],
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
        
        self.camera_sensors = [sensor.upper() for sensor in camera_sensors] if camera_sensors is not None else None
        self.lidar_sensors = [sensor.upper() for sensor in lidar_sensors] if lidar_sensors is not None else None
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
        self.box_type_3d_original = box_type_3d_original
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
        gt_bboxes_pose = np.array([instance['pose'] for instance in info['instances']]).astype(np.float32)
        
        # labels as int might change due to user-defined mapping
        gt_labels_3d = [instance['bbox_label_3d'] for instance in info['instances']]
        gt_labels_3d = np.array([self.label_mapping[label] for label in gt_labels_3d]).astype(np.int64)

        # sample_annotation token in nuscenes
        if num_bboxes > 0 and 'annotation_token' in info['instances'][0]:
            gt_bboxes_anno_token = np.array([instance['annotation_token'] for instance in info['instances']])
                
        # add velocity to gt_bboxes_3d
        gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_bboxes_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        if self.box_type_3d_original.lower() == 'depth':
            BoxInstance = DepthInstance3DBoxes
        elif self.box_type_3d_original.lower() == 'lidar':
            BoxInstance = LiDARInstance3DBoxes
        elif self.box_type_3d_original.lower() == 'camera':
            BoxInstance = CameraInstance3DBoxes
        else:
            raise ValueError(f"Unknown box type {self.box_type_3d_original}")
        
        gt_bboxes_3d = BoxInstance(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        
        # planning annotations
        #gt_instances_traj = info.pop('gt_instances_traj')
        #gt_ego_traj = info.pop('gt_ego_traj')
        
        ann_info = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_bboxes_mask=gt_bboxes_mask,
            gt_bboxes_id=gt_bboxes_id,
            gt_bboxes_pose=gt_bboxes_pose,
            )
        if gt_bboxes_anno_token is not None:
            ann_info['gt_bboxes_anno_token'] = gt_bboxes_anno_token
        
        # filter with mask
        ann_info = self._filter_with_mask(ann_info)
        
        # category statistics
        for label in ann_info['gt_labels_3d']:
            if label != -1:
                self.num_ins_per_cat[label] += 1
        
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

        if not self.test_mode:
            # used in training
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = self.parse_ann_info(info)

        return info
        
    def generate_past_future_info(self, index, curr_info):
        """Generate past/future annotation info, such as future trajectory.

            The coordinate system is in the local lidar coord at the current frame.
        """
        # make sure the required keys are in the info
        #required_keys = []
        #for key in required_keys:
        #    if key not in info:
        #        raise ValueError(f"Key {key} is required in the info.")
    
        # make sure the required attribues are set
        
        # generate ego past/future trajectory
        past_future_ego_traj = self._generate_past_future_ego_trajectory(index, curr_info)

        # generate instances past/future trajectory
        past_future_instances_traj = self._generate_past_future_instances_trajectory(index, curr_info)

        # add to the current info
        if 'ann_info' in curr_info:
            curr_info['ann_info']['gt_ego_traj'] = past_future_ego_traj
            curr_info['ann_info']['gt_bboxes_traj'] = past_future_instances_traj
        elif 'eval_ann_info' in curr_info:
            curr_info['eval_ann_info']['gt_ego_traj'] = past_future_ego_traj
            curr_info['eval_ann_info']['gt_bboxes_traj'] = past_future_instances_traj
        else:
            raise ValueError("No ann_info or eval_ann_info in the current info.")

        return curr_info
        
    def _generate_past_future_ego_trajectory(self, index, curr_info):
        """Generate past and future trajectories for ego vehicle, offset from the current frame.

        Args:
            index (_type_): _description_
            info (_type_): _description_
        
        Returns:
            TrajectoryData: Trajectory data for ego vehicle, with a length of (past_steps + 1 + planning_steps)
        """

        index_list = list(range(index - self.past_steps * self.sample_interval, index + self.planning_steps * self.sample_interval + 1, self.sample_interval))
        lidar2ego = curr_info['lidar_points']['lidar2ego']
        ego2world = curr_info['ego2global']
        world2lidar_curr = np.linalg.inv(np.array(ego2world) @ np.array(lidar2ego))
        xyr = np.zeros((self.past_steps + 1 + self.planning_steps, 3)) # past + current + future
        mask = np.zeros((self.past_steps + 1 + self.planning_steps,)) 

        # current frame: 0
        # TODO: why not use ego2lidar instead of 0?
        xyr[self.past_steps, :2] = 0
        xyr[self.past_steps, 2] = 0 # yaw angle
        mask[self.past_steps] = 1
        
        # past/future frames
        for i, idx in enumerate(index_list):
            # skip the current frame
            if idx == index:
                continue
            # check if index is within range
            if idx < 0 or idx >= len(self):
                continue
            # check if the the frames are from the same scene
            adj_info = self.get_data_info(idx)
            if curr_info['scene_token'] != adj_info['scene_token']:
                continue
            
            lidar_adj2ego_adj = adj_info['lidar_points']['lidar2ego']
            ego_adj2world = adj_info['ego2global'] 
            lidar_adj2world = np.array(ego_adj2world) @ np.array(lidar_adj2ego_adj)
            # T12 = T2^-1 * T1
            adj2curr = world2lidar_curr @ lidar_adj2world
            xyr[i, :2] = adj2curr[:2, 3]
            xyr[i, 2] = np.arctan2(adj2curr[1, 0], adj2curr[0, 0]) # [-pi, pi]
            mask[i] = 1
            
        traj = TrajectoryData(
                metainfo=dict(mode='accumulated',
                    num_past_steps=self.past_steps, 
                    num_future_steps=self.planning_steps,
                    time_step=self.sample_interval/self.FPS), 
                data=xyr.astype(np.float32), 
                mask=mask.astype(np.bool_)
                )
        # get goal point
        if self.with_goal_points:
            #TODO: bugs when indexing
            traj.set_field(traj.data[-1, :], 'goal', field_type='metainfo')
        
        # difference mode for traj
        traj.convert_to_mode('difference')
        
        return traj
    
    def _generate_past_future_instances_trajectory(self, index, curr_info):
        """Generate past and future trajectories for instances, 
            centered at the lidar coords in the current frame.

        Args:
            index (_type_): _description_
            info (_type_): _description_
        
        Returns:
            TrajectoryData: Trajectory data for N instances, with a length of (past_steps + 1 + planning_steps)
        """
        index_list = range(index - self.past_steps * self.sample_interval, 
                           index + self.planning_steps * self.sample_interval + 1, 
                           self.sample_interval)
        instances_ids = curr_info['ann_info']['gt_bboxes_id']
        lidar2ego = curr_info['lidar_points']['lidar2ego']
        ego2world = curr_info['ego2global']
        world2lidar_curr = np.linalg.inv(np.array(ego2world) @ np.array(lidar2ego))
        
        # initialize the trajectory data
        trajs = []
                
        # for each instance in the current frame, find its past and future trajectory
        for i, instance_id in enumerate(instances_ids):
            xyr = np.zeros((self.past_steps + 1 + self.planning_steps, 3)) # (T, 3)
            mask = np.zeros((self.past_steps + 1 + self.planning_steps,)) # (T,)    
            
            # box to lidar_curr
            instance2lidar_curr = world2lidar_curr @ curr_info['ann_info']['gt_bboxes_pose'][i] # (4, 4)
            xyr[self.past_steps, :2] = instance2lidar_curr[:2, 3]
            xyr[self.past_steps, 2] = np.arctan2(instance2lidar_curr[1, 0], instance2lidar_curr[0, 0]) # [-pi, pi]
            mask[self.past_steps] = 1
            
            for j, idx in enumerate(index_list):
                # skip the current frame
                if idx == index:
                    continue
                
                # check if index is within range
                if idx < 0 or idx >= len(self):
                    continue
                # check if the the frames are from the same scene
                adj_info = self.get_data_info(idx)
                if curr_info['scene_token'] != adj_info['scene_token']:                    
                    continue
                # instance not found in the adjacent frame
                if instance_id not in adj_info['ann_info']['gt_bboxes_id']:
                    continue
                # box index of the instance in the adjacent frame
                adj_idx = np.where(adj_info['ann_info']['gt_bboxes_id'] == instance_id)[0][0]
                
                # these two should be the same
                #instance2lidar_adj = adj_info['sensors']['LIDAR_TOP']['world2sensor'] @ adj_info['gt_instance2world'][adj_idx]
                #adj2curr = instance2lidar_curr @ np.linalg.inv(instance2lidar_adj)
                # viewing instance in adj frame lidar coords from the current frame's lidar coord
                adj2curr = world2lidar_curr @ adj_info['ann_info']['gt_bboxes_pose'][adj_idx]

                ## 
                xyr[j, :2] = adj2curr[:2, 3]
                xyr[j, 2] = np.arctan2(adj2curr[1, 0], adj2curr[0, 0]) # [-pi, pi]
                mask[j] = 1
                 
            # save as TrajectoryData
            traj = TrajectoryData(
                metainfo=dict(mode='accumulated',
                    num_past_steps=self.past_steps, 
                    num_future_steps=self.planning_steps,
                    time_step=self.sample_interval/self.FPS), 
                data=xyr.astype(np.float32), 
                mask=mask.astype(np.bool_)
            )
            
            if self.with_goal_points:
                #TODO: bugs when indexing
                traj.set_field(traj.data[-1, :], 'goal', field_type='metainfo')
            
            traj.convert_to_mode('difference')
                
            trajs.append(traj)
        
        return trajs

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
        
        # add past/future annotation info, such as future trajectory
        input_dict = self.generate_past_future_info(index, input_dict)
         
        # assemble for data pipeline
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['data_samples'].gt_instances_3d.labels_3d != -1).any()):
            return None
        return example

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        info = self.prepare_planning_info(index)
        # add past/future annotation info, such as future trajectory
        info = self.generate_past_future_info(index, info) 
        # assemble for data pipeline
        input_dict = self.get_data_info(info)
        if not input_dict:
            return None

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['data_samples'].gt_instances.labels != -1).any()):
            return None
        
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
