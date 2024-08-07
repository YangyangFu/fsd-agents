from typing import Dict, List, Optional, Tuple, Union
from deprecated import deprecated

import torch
from mmdet.structures import DetDataSample
from mmdet3d.structures import Det3DDataSample
from mmengine.structures import InstanceData, PixelData

class FSDDataSample(Det3DDataSample):
    """A data structure interface between different components in fsd planning.
    
    
    
    The following attributes are inherited from Det3DDataSample:
        - ``proposals`` (InstanceData): Region proposals used in two-stage
          detectors.
        - ``ignored_instances`` (InstanceData): Instances to be ignored during
          training/testing.
        - ``gt_instances_3d`` (InstanceData): Ground truth of 3D instance
          annotations.
        - ``gt_instances`` (InstanceData): Ground truth of 2D instance
          annotations.
        - ``pred_instances_3d`` (InstanceData): 3D instances of model
          predictions.
          - For point-cloud 3D object detection task whose input modality is
            `use_lidar=True, use_camera=False`, the 3D predictions results are
            saved in `pred_instances_3d`.
          - For vision-only (monocular/multi-view) 3D object detection task
            whose input modality is `use_lidar=False, use_camera=True`, the 3D
            predictions are saved in `pred_instances_3d`.
        - ``pred_instances`` (InstanceData): 2D instances of model predictions.
          - For multi-modality 3D detection task whose input modality is
            `use_lidar=True, use_camera=True`, the 2D predictions are saved in
            `pred_instances`.
        - ``pts_pred_instances_3d`` (InstanceData): 3D instances of model
          predictions based on point cloud.
          - For multi-modality 3D detection task whose input modality is
            `use_lidar=True, use_camera=True`, the 3D predictions based on
            point cloud are saved in `pts_pred_instances_3d` to distinguish
            with `img_pred_instances_3d` which based on image.
        - ``img_pred_instances_3d`` (InstanceData): 3D instances of model
          predictions based on image.
          - For multi-modality 3D detection task whose input modality is
            `use_lidar=True, use_camera=True`, the 3D predictions based on
            image are saved in `img_pred_instances_3d` to distinguish with
            `pts_pred_instances_3d` which based on point cloud.
        - ``gt_pts_seg`` (PointData): Ground truth of point cloud segmentation.
        - ``pred_pts_seg`` (PointData): Prediction of point cloud segmentation.
        - ``eval_ann_info`` (dict or None): Raw annotation, which will be
          passed to evaluator and do the online evaluation.
    
    The following attributes are specific to PlanDataSample:

    """
    
    # deprecated from inherited class
    @deprecated(version='0.1.0', reason="Use `pred_instances_3d` instead.")
    @property
    def pts_pred_instances_3d(self) -> InstanceData:
        return self._pts_pred_instances_3d

    @pts_pred_instances_3d.setter
    def pts_pred_instances_3d(self, value: InstanceData) -> None:
        self.set_field(value, '_pts_pred_instances_3d', dtype=InstanceData)

    @pts_pred_instances_3d.deleter
    def pts_pred_instances_3d(self) -> None:
        del self._pts_pred_instances_3d
    
    @deprecated(version='0.1.0', reason="Use `pred_instances_3d` instead.")
    @property
    def img_pred_instances_3d(self) -> InstanceData:
        return self._img_pred_instances_3d
    
    @img_pred_instances_3d.setter
    def img_pred_instances_3d(self, value: InstanceData) -> None:
        self.set_field(value, '_img_pred_instances_3d', dtype=InstanceData)
      
    @img_pred_instances_3d.deleter
    def img_pred_instances_3d(self) -> None:
        del self._img_pred_instances_3d
      
    # new properties
    
        