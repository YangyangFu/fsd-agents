# Nuscenes dataset for planning tasks
from os import path as osp
from typing import Callable, List, Union, Optional
import numpy as np 
from mmengine.fileio import load
from mmdet3d.structures import limit_period, CameraInstance3DBoxes, LiDARInstance3DBoxes

from fsd.datasets import NuScenesDatasetPlan3D
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
                 *args,
                 **kwargs) -> None:
        
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
        can_bus = input_dict['can_bus']
        
        local = np.zeros((9,), dtype=np.float32)
        #vx, vy in world frame
        ego_curr = np.array(can_bus[0:2])
        ego_yaw_curr = can_bus[16]
        
        sample_idx_prev = input_dict['sample_idx'] - 1
        sample_idx_next = input_dict['sample_idx'] + 1
        
        # if previous sample exists from the same scene, infer the velocity from prev/curr frame
        # else infer from curr/next frame
        if sample_idx_prev >= 0 and \
            self.get_data_info(sample_idx_prev)['scene_token'] == input_dict['scene_token']:
            data_info_prev = self.get_data_info(sample_idx_prev)
            # get prev pos
            ego_prev = np.array(data_info_prev['ego2global'])[:2, 3]

            # q =-q due to sign ambiguity in quaternion representation
            rot_prev = Quaternion(
                matrix=np.array(data_info_prev['ego2global'])[:3, :3], atol=1e-6
            )
            if rot_prev.w < 0:
                rot_prev = -rot_prev
            ego_yaw_prev = rot_prev.yaw_pitch_roll[0]
        
            ego_v = np.linalg.norm(ego_curr - ego_prev) * self.FPS       
            ego_vx = ego_v * np.cos(ego_yaw_curr + np.pi / 2)
            ego_vy = ego_v * np.sin(ego_yaw_curr + np.pi / 2)
            yaw_speed = (ego_yaw_curr - ego_yaw_prev) * self.FPS
        else:
            data_info_next = self.get_data_info(sample_idx_next)
            ego_next = np.array(data_info_next['ego2global'])[:2, 3]
            # q =-q due to sign ambiguity in quaternion representation
            rot_next = Quaternion(
                matrix=np.array(data_info_next['ego2global'])[:3, :3], atol=1e-6
            )
            if rot_next.w < 0:
                rot_next = -rot_next
            ego_yaw_next = rot_next.yaw_pitch_roll[0]
            
            ego_v = np.linalg.norm(ego_next - ego_curr) * self.FPS
            ego_vx = ego_v * np.cos(ego_yaw_curr + np.pi / 2)
            ego_vy = ego_v * np.sin(ego_yaw_curr + np.pi / 2)
            yaw_speed = (ego_yaw_next - ego_yaw_curr) * self.FPS
        
        local[0] = ego_vx
        local[1] = ego_vy
        
        # acc in ego frame
        local[2:4] = can_bus[7:9] # ax, ay
        
        # rotation speed? rad/s
        local[4] = yaw_speed
        
        # ego size
        local[5] = self.EGO_LENGTH
        local[6] = self.EGO_WIDTH
        
        # logitudianl velocity
        local[7] = can_bus[13]
        
        # curvature
        steer = can_bus[10]
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
        num_agents = input_dict['ann_info']['gt_bboxes_3d'].shape[0]
        local = np.zeros((num_agents, 9), dtype=np.float32)
        
        # x, y
        local[:, 0:2] = input_dict['ann_info']['gt_bboxes_3d'].center[:, :2]
        
        # yaw
        local[:, 2] = input_dict['ann_info']['gt_bboxes_3d'].yaw
        
        # vx, vy
        local[:, 3:5] = input_dict['ann_info']['gt_bboxes_3d'].tensor[:, -2:]

        # w, l, h
        local[:, 5:8] = input_dict['ann_info']['gt_bboxes_3d'].dims[:, [1, 0, 2]]
        # type
        local[:, 8] = input_dict['ann_info']['gt_labels_3d']
        
        return local 
    
    
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
        
        # get ego local context feature
        ego_local = self._get_ego_local_context(input_dict)
        input_dict['ego_local_context'] = ego_local
        agent_local = self._get_agents_local_context(input_dict)
        input_dict['agent_local_context'] = agent_local
        
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