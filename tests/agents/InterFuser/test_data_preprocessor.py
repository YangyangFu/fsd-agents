import pytest
from mmengine.config import Config
from mmengine.registry import init_default_scope
from fsd.runner import Runner
from fsd.agents import InterFuserDataPreprocessor
from fsd.registry import MODELS

ds_config = Config.fromfile('fsd/configs/_base_/dataset/carla_dataset.py')
model_config = Config.fromfile('fsd/configs/_base_/model/interfuser_r50.py')

init_default_scope('fsd')
dl = Runner.build_dataloader(ds_config.dataloader)

def test_planning_data_preprocessor():
    data_preprocessor = InterFuserDataPreprocessor()
    
    for sample in dl:
        print(sample.keys())
        sample = data_preprocessor(sample)
        print(sample.keys())
        
        assert len(sample['inputs']['img']) == 6
        assert sample['inputs']['img'][0].shape == (2, 3, 928, 1600)
        assert 'goal_points' in sample['inputs']
        break

def test_build_data_preprocessor():
    cfg = model_config.model.data_preprocessor
    data_preprocessor = MODELS.build(cfg)
    
    for sample in dl:
        print(sample.keys())
        sample = data_preprocessor(sample)
        print(sample.keys())
        
        assert len(sample['inputs']['img']) == 6
        assert sample['inputs']['img'][0].shape == (2, 3, 928, 1600)
        assert 'goal_points' in sample['inputs']
        break        


pytest.main(["tests/agents/InterFuser/test_data_preprocessor.py"])