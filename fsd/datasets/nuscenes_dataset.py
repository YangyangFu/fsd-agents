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
    
    def _get_can_bus_info(self, info: dict) -> dict:
        """Get can bus info for the current sample.
        
        Overwrite this to customize the can bus info.
        
        Args:
            info (dict): Raw info dict.
        """
        return info

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
        info = super().parse_data_info(info)
        
        # if need additional can bus info, use nusc can bus api 
        if self.with_can_bus:
            info = self._get_can_bus_info(info)
            
        return info
              