from os import path as osp
from typing import Callable, List, Union, Optional
import numpy as np 
from pyquaternion import Quaternion

from mmengine.fileio import load
from mmdet3d.structures import CameraInstance3DBoxes, LiDARInstance3DBoxes
from mmdet3d.datasets import Det3DDataset
from fsd.datasets import Planning3DDataset
from fsd.datasets import map_carla_class_name
from fsd.registry import DATASETS


@DATASETS.register_module()
class CarlaDataset(Planning3DDataset):
    def __init__(self, 
                 with_velocity: bool = True,
                 *args, 
                 **kwargs):
        """_summary_

        Args:
            with_velocity (bool, optional): add velocity to bounding box. Defaults to True.
        """
        super().__init__(*args, **kwargs)
        
        self.with_velocity = with_velocity
          
    def prepare_planning_info(self, index):
        
        # TODO: do we need convert the velocities to camera frame/ego frame???
        raw_info = self.data_infos[index]
        
        # process to standard format in planning library
        info = dict()
        info['folder'] = raw_info['folder']
        info['scene_token'] = raw_info['folder']
        info['frame_index'] = raw_info['frame_idx']
        
        # ego info
        ego = dict()
        ego['size'] = raw_info['ego_size'] # [x, y, z]
        ego['world2ego'] = raw_info['world2ego'] # (4, 4)
        ego['translation'] = raw_info['ego_translation'] # ? 
        ego_yaw = raw_info['ego_yaw'] # in radian
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_yaw))
        ego['rotation'] = rotation
        ego['yaw'] = ego_yaw
        ego['velocity'] = raw_info['ego_vel'] # [x, y, z]
        ego['acceleration'] = raw_info['ego_accel'] # [x, y, z]
        info['ego'] = ego
        
        # sensor info
        sensors = dict()
        for sensor in raw_info['sensors'].keys():
            data = {}
            if 'cam' in sensor.lower():
                data['sensor2ego'] = raw_info['sensors'][sensor]['cam2ego']
                data['world2sensor'] = raw_info['sensors'][sensor]['world2cam']
                data['intrinsic'] = raw_info['sensors'][sensor]['intrinsic']
                data['data_path'] = raw_info['sensors'][sensor]['data_path']
            elif 'lidar' in sensor.lower():
                data['sensor2ego'] = raw_info['sensors'][sensor]['lidar2ego']
                data['world2sensor'] = raw_info['sensors'][sensor]['world2lidar']
                # the raw data anno does not provide lidar data path
                dpath = raw_info['sensors'][sensor].get('data_path', None)
                if dpath is not None:
                    data['data_path'] = dpath
                else:
                    data['data_path'] = osp.join(info['folder'], 'lidar', f'{info["frame_index"]:05d}.laz')
            sensors[sensor] = data        
        info['sensors'] = sensors
        
        ## ground truth
        # --------------------------------------------------------------------------------
        info['gt_instances_ids'] = raw_info['gt_ids']
        info['gt_bboxes_3d'] = raw_info['gt_boxes'][:, :7] # (N, 9) -> (N, 7)
        
        # raw namse are object name in carla, need map to standard class name
        info['gt_instances_names'] = raw_info['gt_names']
        mapped_classes = []
        for name in info['gt_instances_names']:
            mapped_classes.append(map_carla_class_name(name))
        info['gt_instances_names'] = np.array(mapped_classes)
        
        # box masks 
        # filtering out boxes that are hit by lidar points: too close or too far
        info['gt_bboxes_mask'] = raw_info['num_points'] >= -1
        
        # box velocities: coordinate ??
        info['gt_instances_velocities'] = raw_info['gt_boxes'][:, 7:9] # (N, 2)
        info['gt_instances2world'] = raw_info['npc2world'] # (N, 4, 4)
        
        # box center 
        info['box_center'] = (0.5, 0.5, 0.5) # raw data assumes box center is at the center of the box
         
        return info
        
        