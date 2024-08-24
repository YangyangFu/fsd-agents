import pytest 

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.visualization import Visualizer
from fsd.runner import Runner

def test_base_carla_dataset():

    config = Config.fromfile('fsd/configs/_base_/dataset/carla_dataset.py')
    init_default_scope('fsd')
    dl = Runner.build_dataloader(config.dataloader)

    # check samples 
    for sample in dl:
        for key in ['img_fields', 'pts_fields', 'ego_fields', \
                        'bbox3d_fields', 'grid_fields', 'pts_seg_fields', 'bbox_fields', \
                            'box_type_3d', 'box_mode_3d', 'inputs', 'data_samples']:
            assert key in sample, f'{key} not in sample'
        
        break

# add to test 
pytest.main(['-q', 'tests/configs/_base_/test_carla_dataset.py'])