from typing import Dict, Union

import torch
from mmengine.structures import BaseDataElement, InstanceData, PixelData

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
    def gt_ego(self) -> BaseDataElement:
        return self._gt_ego
    @gt_ego.setter 
    def gt_ego(self, value: BaseDataElement) -> None:
        self.set_field(value, "_gt_ego", dtype=BaseDataElement)
    @gt_ego.deleter
    def gt_ego(self) -> None:
        del self._gt_ego
    
    @property
    def gt_instances_3d(self) -> InstanceData:
        return self._gt_instances_3d
    @gt_instances_3d.setter
    def gt_instances_3d(self, value: InstanceData):
        self.set_field(value, "_gt_instances_3d", dtype=InstanceData)
    @gt_instances_3d.deleter
    def gt_instances_3d(self) -> None:
        del self._gt_instances_3d
        
    @property
    def gt_pts_seg(self) -> PixelData:
        return self._gt_pts_seg
    @gt_pts_seg.setter
    def gt_pts_seg(self, value: PixelData):
        self.set_field(value, "_gt_pts_seg", dtype=PixelData)
    @gt_pts_seg.deleter
    def gt_pts_seg(self) -> None:
        del self._gt_pts_seg