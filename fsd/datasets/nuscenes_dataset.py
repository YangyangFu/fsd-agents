# Nuscenes dataset for planning tasks
from os import path as osp
from typing import Callable, List, Union, Optional
import numpy as np 
from pyquaternion import Quaternion

from mmengine.fileio import load
from mmdet3d.structures import limit_period, CameraInstance3DBoxes, LiDARInstance3DBoxes
from mmdet3d.datasets import Det3DDataset
from fsd.datasets import Planning3DDataset
from fsd.datasets.convert_utils import nus_categories, NuScenesNameMapping
from fsd.utils import one_hot_encoding
from fsd.registry import DATASETS

@DATASETS.register_module()
class NuscenesDatasetPlanning(Planning3DDataset):
    """NuScenes dataset for planning tasks.
    
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
                 with_velocity: bool = True,
                 with_can_bus: bool = True,
                 *args,
                 **kwargs):
        """_summary_
        Args:
            with_velocity (bool, optional): add velocity to bounding box. Defaults to True.
            with_can_bus (bool, optional): add can bus information. Defaults to True.
        """
        super().__init__(*args, **kwargs)
        self.with_velocity = with_velocity
        self.with_can_bus = with_can_bus
        
        
    def prepare_planning_info(self, index):
        """ Prepare data for planning library
        """
        
        
    
    
    