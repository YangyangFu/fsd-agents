# Nuscenes dataset for planning tasks
from os import path as osp
from typing import Callable, List, Union, Optional
import numpy as np 
import copy
import torch
from mmengine.fileio import load
from mmdet3d.structures import limit_period, CameraInstance3DBoxes, LiDARInstance3DBoxes, DepthInstance3DBoxes

from fsd.datasets import NuScenesDatasetPlan3D
from fsd.datasets.map_utils.vector_map import VectorizedLocalMap
from fsd.datasets.convert_utils import nus_categories, NuScenesNameMapping
from fsd.structures import TrajectoryData
from fsd.utils import one_hot_encoding
from fsd.registry import DATASETS

# FOR NUSCENES
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion

@DATASETS.register_module()
class NuScenesDatasetVAD(NuScenesDatasetPlan3D):
    """NuScenes dataset for 3D planning tasks.
    
    Compared with NuScenesDatasetPlan3D, this dataset used for VAD tasks has the following differences:
    - add local context feature for ego and agents information
    - some coordinate changes to reuse original algorithm checkpoints
    
    """
    
    METAINFO = {
        'name': 'nuscenes-vad',
        'classes': nus_categories,
        'map_classes': ('divider', 'ped_crossing','boundary'),
        'version': 'v1.0-trainval',
        'palette': [
            (255, 158, 0),  # Orange
            (255, 99, 71),  # Tomato
            (255, 140, 0),  # Darkorange
            (255, 127, 80),  # Coral
            (233, 150, 70),  # Darksalmon
            (220, 20, 60),  # Crimson
            (255, 61, 99),  # Red
            (0, 0, 230),  # Blue
            (47, 79, 79),  # Darkslategrey
            (112, 128, 144),  # Slategrey
        ]
    }
    
    def __init__(self,
                 point_cloud_range: Union[List[float], np.ndarray] = [-15, -30, -2.0, 15, 30, 2.0],
                 map_fixed_ptsnum_per_line: int = 20,
                 map_sample_dist: float = 1,
                 map_sample_nums: int = 250,
                 map_padding_value: float = -10000,
                 bev_queue_length: int = 2,
                 bev_size: tuple = (200, 200),
                 *args,
                 **kwargs) -> None:
        
        # local map
        # get data_root from super class arguments
        data_root = kwargs.get('data_root', None)
        if data_root is None:
            raise ValueError('data_root should be provided in the arguments')
        metainfo = kwargs.get('metainfo', None)
        if metainfo is None and self.METAINFO['map_classes'] is None:
            raise ValueError('metainfo should be provided in the arguments')
        map_classes = metainfo['map_classes'] if 'map_classes' in metainfo else self.METAINFO['map_classes']
        
        
        self.point_cloud_range = point_cloud_range
        patch_h = self.point_cloud_range[4] - self.point_cloud_range[1]
        patch_w = self.point_cloud_range[3] - self.point_cloud_range[0]
        self.map_patch_size = (patch_h, patch_w)
        self.map_padding_value = map_padding_value
        self.map_fixed_ptsnum_per_line = map_fixed_ptsnum_per_line
        self.map_sample_dist = map_sample_dist
        self.map_sample_nums = map_sample_nums
        self.local_map = VectorizedLocalMap(
            data_root=data_root,
            patch_size=self.map_patch_size,
            map_classes=map_classes,
            fixed_ptsnum_per_line=self.map_fixed_ptsnum_per_line,
            sample_dist=self.map_sample_dist,
            num_samples=self.map_sample_nums,
            padding_value=self.map_padding_value
        )

        # bev 
        self.bev_queue_length = bev_queue_length
        assert self.bev_queue_length > 0, 'bev_queue_length should be greater than 0'
        self.bev_size = bev_size
      
        # initialize the super class
        super().__init__(*args, **kwargs)
      
    def _get_ego_local_context(self, input_dict: dict) -> np.ndarray:
        """Get ego local context feature as a vector of size 9, 
            (vx, vy, ax, ay, yaw, length, width, vel, kappa)
        
            - vx, vy: velocity in x and y direction in world frame
            - ax, ay: acceleration in x and y direction in ego frame
            - yaw: yaw angle in radians in world coordinate
            - length: length of the ego vehicle
            - width: width of the ego vehicle
            - vel: velocity in longitudinal direction, forward direction in ego frame
            - kappa: curvature
            
        Args:
            input_dict (dict): Input data dictionary.
        
        Returns:
            np.ndarray: Ego local context feature.
        """
        can_bus = input_dict['ego_can_bus']
        ego_velocity = input_dict['ego_velocity']
        ego_yaw_velocity = input_dict['ego_yaw_velocity']
        ego_size = input_dict['ego_size']
        
        local = np.zeros((9,), dtype=np.float32)
        local[0] = ego_velocity[0] # vx
        local[1] = ego_velocity[1] # vy
        
        # acc in ego frame
        local[2:4] = can_bus[7:9] # ax, ay
        
        # rotation speed? rad/s
        local[4] = ego_yaw_velocity
        
        # ego size
        local[5] = ego_size[0] # length
        local[6] = ego_size[1] # width
        
        # logitudianl velocity
        local[7] = can_bus[13]
        
        # curvature
        steer = can_bus[16]
        # if left-driving cities: steer *= -1
        local[8] = steer * 2 / 2.588

        return local

    def _get_agents_local_context(self, input_dict: dict) -> np.ndarray:
        """Get local context feature for agents. Shape is (N, 9), where N is the number of agents.
            - x, y: position in lidar frame
            - yaw: yaw angle in radians in nuscenes lidar frame with respect to x axis
            - vx, vy: velocity in x and y direction in XXX frame??
            - w, l, h: width, length, height
            - type: labels
            
        """
        ann_info = input_dict['ann_info']
        num_agents = ann_info['gt_bboxes_3d'].shape[0]
        local = np.zeros((num_agents, 9), dtype=np.float32)
        
        # x, y
        local[:, 0:2] = ann_info['gt_bboxes_3d'].center[:, :2]
        
        # yaw
        local[:, 2] = ann_info['gt_bboxes_3d'].yaw
        
        # vx, vy
        local[:, 3:5] = ann_info['gt_bboxes_3d'].tensor[:, -2:]

        # w, l, h
        local[:, 5:8] = ann_info['gt_bboxes_3d'].dims[:, [1, 0, 2]]
        # type
        local[:, 8] = ann_info['gt_labels_3d']
        
        return local 
    
    def _get_agents_goal_direction(self, goals: np.ndarray) -> np.ndarray:
                
        num_agents, _ = goals.shape
        directions = np.zeros((num_agents, 1), dtype=np.float32)
        
        directions = []
        for i in range(num_agents):
            goal = goals[i, :2]
            if goal.max() < 1.0: # static
                direction = 9
            else:
                box_yaw = np.arctan2(goal[1], goal[0]) + np.pi # [0, 2pi]
                direction = box_yaw // (np.pi / 4) # 0-8
            directions.append(direction)

        return np.array(directions, dtype=np.int8)
   
    def _add_agents_attributes(self, input_dict: dict) -> None:
        """Add agent attributes to the input dict.
           [future_traj, future_traj_mask, goal, local_context, yaw]

        Args:
            input_dict (dict): Input data dictionary.
            
        """
        fut_traj = []
        fut_traj_mask = []
        goal = []
        # (num_boxes, num_steps, 4) - (x, y, z, yaw)
        fut_traj = input_dict['ann_info']['gt_bboxes_traj']
        fut_traj_xy = fut_traj[:, :, :2]
        fut_traj_yaw = fut_traj[:, :, 3]
        fut_traj_mask = input_dict['ann_info']['gt_bboxes_traj_mask']
        goal = input_dict['ann_info']['gt_bboxes_goal']
        
        # extrack goal direction
        goal_direction = self._get_agents_goal_direction(goal)
        
        # local features from agents
        local = self._get_agents_local_context(input_dict)
        
        attr = np.concatenate([fut_traj_xy[:, :, :2].reshape(-1, self.prediction_steps * 2), 
                fut_traj_mask, 
                goal_direction.reshape(-1, 1),
                local,
                fut_traj_yaw
                ], 
            axis=-1
        ).astype(np.float32)
        
        input_dict['bboxes_context'] = attr 
    
    def _get_ego_history_trajectory(self, input_dict: dict) -> np.ndarray:
        """Get ego history trajectory as a vector of 3
            (x, y, yaw) in current lidar frame.
        """

        traj_xy = input_dict.pop('ego_history_trajectory')[:, :2]
        traj_yaw = input_dict.pop('ego_history_yaw')
        traj_mask = np.array(input_dict['ego_history_mask']).astype(np.bool_)
        
        # 
        traj = np.concatenate([traj_xy.reshape(-1, 2), 
                               traj_yaw.reshape(-1, 1)], axis=-1)
        
        return traj, traj_mask
    
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
        yaw_degree = yaw / np.pi * 180
        can_bus = np.concatenate([can_bus, [yaw, yaw_degree]])
        
        input_dict['ego_can_bus'] = can_bus
        return input_dict 

    # parse local map based on ego position
    def parse_map_ann_info(self, input_dict):
        """Get local map in lidar coord.

        Args:
            input_dict (dict): Input data dictionary.

        Returns:
            np.ndarray: Local map.
        """
        # lidar2ego 
        lidar2ego = input_dict['lidar_points']['lidar2ego']
        # ego2global
        ego2global = input_dict['ego2global']
        
        # lidar2global
        lidar2global = np.array(ego2global) @ np.array(lidar2ego)
        translation = lidar2global[:3, 3]
        rotation = list(Quaternion(matrix=lidar2global[:3, :3], atol=1e-6).q)
        # get local map
        ann_map = self.local_map.gen_vectorized_samples(
            location=input_dict['map_location'],
            lidar2global_translation=translation.tolist(),
            lidar2global_rotation=rotation
        )

        return ann_map
    
    def parse_ann_info(self, info: dict) -> dict:
        """Process the raw annotation info.

        Convert all relative path of needed modality data file to
        the absolute path. Process the `instances`, 'ego' and `map` field
        to the corresponding data format.
        """
        ann_info = super().parse_ann_info(info)
        
        # add local map info
        ann_map = self.parse_map_ann_info(info)
        
        # add to ann_info
        ann_info['gt_map_vectors_pt'] = ann_map['gt_vecs_pts_loc']
        ann_info['gt_map_vectors_label'] = ann_map['gt_vecs_label']
        
        return ann_info
    
    def _prepare_data(self, index) -> dict:
        """Prepare data given sample index.
        """
        # get data info
        input_dict = self.get_data_info(index)
        if not input_dict:
            return None
                            
        # add agent attributes as in original VAD paper
        self._add_agents_attributes(input_dict)
        
        # add ego features
        # history trajectory
        input_dict['ego_history_traj'], input_dict['ego_history_mask'] = \
            self._get_ego_history_trajectory(input_dict)
        # local context
        input_dict['ego_context'] = self._get_ego_local_context(input_dict)

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
        index_list = [i for i in range(index - self.bev_queue_length, index)]
        index_list = np.random.choice(index_list, size=self.bev_queue_length, replace=False)
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