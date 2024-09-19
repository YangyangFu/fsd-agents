from typing import Dict, Union

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

    # properties: ego
    @property
    def gt_ego(self) -> Ego:
        return self._gt_ego
    @gt_ego.setter 
    def gt_ego(self, value: Ego) -> None:
        self.set_field(value, "_gt_ego", dtype=Ego)
    @gt_ego.deleter
    def gt_ego(self) -> None:
        del self._gt_ego
    
    @property
    def pred_ego(self) -> Ego:
        return self._pred_ego
    @pred_ego.setter
    def pred_ego(self, value: Ego) -> None:
        self.set_field(value, "_pred_ego", dtype=Ego)
    @pred_ego.deleter
    def pred_ego(self) -> None:
        del self._pred_ego
    
    # properties: instances
    @property
    def gt_instances(self) -> Instances:
        return self._gt_instances
    @gt_instances.setter
    def gt_instances(self, value: Instances):
        self.set_field(value, "_gt_instances", dtype=Instances)
    @gt_instances.deleter
    def gt_instances(self) -> None:
        del self._gt_instances

    @property
    def pred_instances(self) -> Instances:
        return self._pred_instances
    @pred_instances.setter
    def pred_instances(self, value: Instances):
        self.set_field(value, "_pred_instances", dtype=Instances)
    @pred_instances.deleter
    def pred_instances(self) -> None:
        del self._pred_instances
        
    # properties: grids 
    @property
    def gt_grids(self) -> Grids:
        return self._gt_grids
    @gt_grids.setter
    def gt_grids(self, value: Grids):
        self.set_field(value, "_gt_grids", dtype=Grids)
    @gt_grids.deleter
    def gt_grids(self) -> None:
        del self._gt_grids
    
    @property
    def pred_grids(self) -> Grids:
        return self._pred_grids
    @pred_grids.setter
    def pred_grids(self, value: Grids):
        self.set_field(value, "_pred_grids", dtype=Grids)
    @pred_grids.deleter
    def pred_grids(self) -> None:
        del self._pred_grids
    
    #add point cloud annotation field
    @property
    def gt_pts(self) -> BaseDataElement:
        if hasattr(self, "_gt_pts"):
            return self._gt_pts
        return None
    
    @gt_pts.setter
    def gt_pts(self, value: BaseDataElement):
        self.set_field(value, "_gt_pts", dtype=BaseDataElement)
    @gt_pts.deleter
    def gt_pts(self) -> None:
        del self._gt_pts
    
    @property
    def pred_pts(self) -> BaseDataElement:
        if hasattr(self, "_pred_pts"):
            return self._pred_pts
        else:
            return None
    @pred_pts.setter
    def pred_pts(self, value: BaseDataElement):
        self.set_field(value, "_pred_pts", dtype=BaseDataElement)
    @pred_pts.deleter
    def pred_pts(self) -> None:
        del self._pred_pts
    