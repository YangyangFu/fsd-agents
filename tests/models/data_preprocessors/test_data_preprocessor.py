import pytest
from mmengine.config import Config
from mmengine.registry import init_default_scope
from fsd.runner import Runner
from fsd.models import PlanningDataPreprocessor

config = Config.fromfile('fsd/configs/_base_/dataset/carla_dataset.py')
init_default_scope('fsd')
dl = Runner.build_dataloader(config.train_dataloader)


def test_planning_data_preprocessor():
    data_preprocessor = PlanningDataPreprocessor()
    
    for sample in dl:
        print(sample.keys())
        sample = data_preprocessor(sample)
        print(sample.keys())
        print(sample['inputs']['img'].shape)
        
        assert sample['inputs']['img'].shape == (2, 6, 3, 900, 1600)
        break

pytest.main(["tests/models/data_preprocessors/test_data_preprocessor.py"])