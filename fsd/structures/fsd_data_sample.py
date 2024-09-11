from typing import Dict, Union

import torch
from mmengine.structures import BaseDataElement
from fsd.structures.fsd_data import Ego, Instances, Grids

class PlanningDataSample(BaseDataElement):
    """A data structure interface between different components in fsd planning.
    
    For end-to-end planning and control tasks, it is often a good assumption 
    that 3D bounding boxes are part of the annotations. This library assumes 
    3D bounding boxes are in lidar coordinates as default. The following attributes 
    are included:

    Meta Info:
    - metainfo (dict): A dictionary containing metadata information.
        - timestamp: The timestamp of the data sample.
        - lidar2ego: The transformation matrix from lidar to world coordinates.
        - camera2ego: The transformation matrix from camera to world coordinates.
        - camera_instrinsics: The intrinsic matrix of the camera.
        
    Data Fields
    - gt_ego (dict): A dictionary containing the ground truth ego vehicle information.
      - goal_point: The goal point of the ego vehicle.
      - ego2world: The transformation matrix from ego to world coordinates.
      - 
    - gt_instances_3d (InstanceData): Ground truth of 3D instance annotations.
      - metainfo (dict or None): A dictionary containing metadata information.
      - bboxes (LiDAR3DBoxes): 3D bounding boxes in LiDAR coordinates. 
          If the dimension is 7, the boxes are in the format of [x, y, z, dx, dy, dz, heading].
          If the dimension is 9, the boxes are in the format of [x, y, z, dx, dy, dz, heading, vx, vy].
      - labels (torch.Tensor): The class labels of the boxes.
      - ids (torch.Tensor): The instance ids of the boxes.
      - bboxes_mask (torch.Tensor): mask of the boxes to be ignored. 
      - bboxes2world (torch.Tensor): The transformation matrix from the boxes to world coordinates. ???
      - past_traj (TrajectoryData): The past trajectory of the boxes.
      - future_traj (TrajectoryData): The future trajectory of the boxes.
    
    - gt_pts_seg (PixelData): Ground truth of point cloud segmentation.
    
    # To-be-added
    - gt_map (PixelData): Ground truth of the map.  
    """

    # properties
    @property
    def ego(self) -> Ego:
        return self._ego
    @ego.setter 
    def ego(self, value: Ego) -> None:
        self.set_field(value, "_ego", dtype=Ego)
    @ego.deleter
    def ego(self) -> None:
        del self._ego
    
    @property
    def instances(self) -> Instances:
        return self._instances
    @instances.setter
    def instances(self, value: Instances):
        self.set_field(value, "_instances", dtype=Instances)
    @instances.deleter
    def instances(self) -> None:
        del self._instances
        
    @property
    def grids(self) -> Grids:
        return self._grids
    @grids.setter
    def grids(self, value: Grids):
        self.set_field(value, "_grids", dtype=Grids)
    @grids.deleter
    def grids(self) -> None:
        del self._grids