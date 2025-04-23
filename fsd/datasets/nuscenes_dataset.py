# Nuscenes dataset for planning tasks
import os
from os import path as osp
from typing import Callable, List, Union, Optional
import numpy as np 
import copy

from mmengine.fileio import load
from mmdet3d.structures import limit_period, CameraInstance3DBoxes, LiDARInstance3DBoxes
from mmdet3d.datasets import Det3DDataset
from fsd.datasets import BasePlanDataset
from fsd.datasets.convert_utils import nus_categories, NuScenesNameMapping
from fsd.structures import TrajectoryData
from fsd.registry import DATASETS

# FOR NUSCENES
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion

@DATASETS.register_module()
class NuScenesDatasetPlan3D(BasePlanDataset):
    """NuScenes dataset for 3D planning tasks.
    
    """
    # transformation matrix from dataset lidar coordinate to mmdet3d lidar
    # now assumes no transformation
    TO_MMDET3D_LIDAR = np.eye(4)
    
    METAINFO = {
        'name': 'nuscenes',
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
    
    # TODO: find a better way for EGO
    EGO_LENGTH = 4.084
    EGO_WIDTH = 1.85
    
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
                 with_can_bus: bool = True,
                 **kwargs) -> None:
        
        # attribute
        self.with_can_bus = with_can_bus
        if self.with_can_bus:
            self.can_bus = NuScenesCanBus(dataroot=data_root)
            
        # add nuscenes before super().__init__() 
        version = metainfo.get('version', self.METAINFO['version'])
        self.nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
                    
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            camera_sensors=camera_sensors,
            lidar_sensors=lidar_sensors,
            box_type_3d_original=box_type_3d_original,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            past_steps=past_steps,
            prediction_steps=prediction_steps,
            planning_steps=planning_steps,
            sample_interval=sample_interval,
            FPS=FPS,
            test_mode=test_mode,
            load_eval_anns=load_eval_anns,
            show_ins_var=show_ins_var,
            **kwargs)
        
    def _get_can_bus_info(self, input_dict):
        """Get can_bus information given the sample token in the input_dict.
        
        can_bus is a list of 18 elements:
            [x, y, z, qw, qx, qy, qz, ax, ay, az, rx, ry, rz, vx, vy, vz, yaw, steer]
            where:
            - x, y, z are the translation of the ego vehicle in the world coordinate system in meters,
            - qw, qx, qy, qz are the rotation vector in the ego vehicle frame
            - ax, ay, az are the acceleration vector in the ego vehicle frame in m/s^2
            - rx, ry, rz are the angular velocity vector in the ego vehicle frame in rad/s
            - vx, vy, vz are the velocity in the ego vehicle frame in m/s
            - yaw is the yaw angle in the ego vehicle frame in radians
            - steer is the steering angle in radians in range [-7.7, 6.3]. 
                0 indicates no steering, positive values indicate left turns, 
                negative values right turns.

            
        Returns:
            input_dict (dict): Updated input_dict with 'can_bus' key.
        """
        sample_token = input_dict['token']
        sample = self.nusc.get('sample', sample_token)
        scene_token = sample['scene_token']
        scene_name = self.nusc.get('scene', scene_token)['name']
        sample_timestamp = sample['timestamp']
        
        # get can bus information
        try:
            pose_list = self.can_bus.get_messages(scene_name, 'pose')
            steer_list = self.can_bus.get_messages(scene_name, 'steeranglefeedback')
        except:
            # if no can bus information, return a default value
            can_bus = [0]*18
            input_dict.update(
                can_bus=np.array(can_bus).astype(np.float32),
            )
            return input_dict
            
        can_bus = []
        # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
        last_pose = pose_list[0]
        for i, pose in enumerate(pose_list):
            if pose['utime'] > sample_timestamp:
                break
            last_pose = pose
        # first 16 elements (x, y, z, qx, qy, qz, qw, ax, ay, az, rx, ry, rz, vx, vy, vz)
        pos = last_pose['pos'] # 3
        orientation = last_pose['orientation'] # 4
        can_bus.extend(pos) 
        can_bus.extend(orientation)
        for key in ['accel', 'rotation_rate', 'vel']:
            can_bus.extend(pose[key])  
        # the last two numbers are reserved for later calculation of rotation angle.
        can_bus.extend([0., 0.])

        # update pose from calibrated sensor data
        ego2global_cs = np.array(input_dict['ego2global'])
        # q =-q due to sign ambiguity in quaternion representation
        rotation = Quaternion(matrix=ego2global_cs[:3, :3], atol=1e-6)
        if rotation.w < 0:
            rotation = -rotation
        translation = ego2global_cs[:3, 3]
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        
        # calculate yaw angle
        yaw_angle = quaternion_yaw(rotation)
        can_bus[-2] = yaw_angle
        
        # get steering: positive means turn left
        # note in left-handed system, this may need to be flipped to keep consistent with
        # the right-handed system when data are collected from different driving systems.
        # TODO: add singpore for left-handed system 
        last_steer = steer_list[0]
        for i, steer in enumerate(steer_list):
            if steer['utime'] > sample_timestamp:
                break
            last_steer = steer
        steer = last_steer['value']
        can_bus[-1] = steer
        

        # save to input_dict
        input_dict.update(
            can_bus=np.array(can_bus).astype(np.float32),
        )
        
        return input_dict
    
    def _get_nuscenes_box_pose(self, sample_annotation):
        """Get the pose of the box in the world coordinate system.
        """
        box_pose = np.eye(4)
        rotation = Quaternion(sample_annotation['rotation']).rotation_matrix
        translation = np.array(sample_annotation['translation'])
        box_pose[:3, :3] = rotation
        box_pose[:3, 3] = translation
        
        return box_pose
    
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
        ann_tokens_curr = curr_info['ann_info']['gt_bboxes_anno_token']
        
        # initialize the trajectory data
        trajs = []
                
        # for each instance in the current frame, find its past and future trajectory
        for i, _ in enumerate(instances_ids):
            xyr = np.zeros((self.past_steps + 1 + self.planning_steps, 3)) # (T, 3)
            mask = np.zeros((self.past_steps + 1 + self.planning_steps,)) # (T,)    
            
            # box to lidar_curr
            instance2lidar_curr = world2lidar_curr @ curr_info['ann_info']['gt_bboxes_pose'][i] # (4, 4)
            xy_curr = instance2lidar_curr[:2, 3]
            r_curr = np.arctan2(instance2lidar_curr[1, 0], instance2lidar_curr[0, 0]) # [-pi, pi]
            # current xyr 
            xyr[self.past_steps, :2] = xy_curr
            xyr[self.past_steps, 2] = r_curr
            mask[self.past_steps] = 1
            
            # sample annotation 
            ann_token_curr = ann_tokens_curr[i]
            ann_curr = self.nusc.get('sample_annotation', ann_token_curr)
            
            # history traj
            _ann = ann_curr
            for j in range(self.past_steps-1, -1, -1):
                # sample interval
                for k in range(self.sample_interval):
                    if _ann['prev'] == '':
                        break
                    # get the prev sample annotation
                    _ann = self.nusc.get('sample_annotation', _ann['prev'])
                # extract trajectory
                # global coord
                box_pose = self._get_nuscenes_box_pose(_ann)
                # box to lidar_curr
                adj2curr = world2lidar_curr @ box_pose
                # save to the trajectory
                xyr[j, :2] = adj2curr[:2, 3]
                xyr[j, 2] = np.arctan2(adj2curr[1, 0], adj2curr[0, 0]) # [-pi, pi]
                mask[j] = 1
                
            # future traj
            _ann = ann_curr
            for j in range(self.past_steps+1, self.past_steps + self.planning_steps+1):
                # sample interval
                for k in range(self.sample_interval):
                    if _ann['next'] == '':
                        break
                    # get the next sample annotation
                    _ann = self.nusc.get('sample_annotation', _ann['next'])
                    
                # extract trajectory
                # global coord
                box_pose = self._get_nuscenes_box_pose(_ann)
                # box to lidar_curr
                adj2curr = world2lidar_curr @ box_pose
                
                # save to the trajectory
                xyr[j, :2] = adj2curr[:2, 3]
                xyr[j, 2] = np.arctan2(adj2curr[1, 0], adj2curr[0, 0]) # [-pi, pi]
                mask[j] = 1
            
            traj = TrajectoryData(
                    metainfo=dict(
                        mode = 'accumulated',
                        num_past_steps=self.past_steps, 
                        num_future_steps=self.planning_steps,
                        time_step=self.sample_interval/self.FPS)
            )
            traj.data=xyr.astype(np.float32)
            traj.mask=mask.astype(np.bool_)
            
            # to diff mode
            traj.convert_to_mode('difference')
            # save
            trajs.append(traj)
            
        return trajs

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

        # add can bus info: info['can_bus']
        if self.with_can_bus:
            info = self._get_can_bus_info(info)
        
        # add anno info
        if not self.test_mode:
            # used in training
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = self.parse_ann_info(info)

        return info
              
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
        
        # can bus information
        if self.with_can_bus:
            input_dict = self._get_can_bus_info(input_dict)
        
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