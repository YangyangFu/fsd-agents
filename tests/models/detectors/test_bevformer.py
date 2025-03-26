import pytest 
from mmengine.config import Config
from mmengine.registry import init_default_scope

from fsd.utils import seed_everything
from fsd.registry import MODELS, RUNNERS
from fsd.runner import Runner

@pytest.fixture(autouse=True)
def seed():
    seed_everything(2024)
    

cfgs = ['fsd/configs/bevformer/bevformer_base.py']
@pytest.mark.parametrize('cfg', cfgs)
def test_bevformer(cfg):
    # scope
    init_default_scope('fsd')
    # cfg
    cfg = Config.fromfile(cfg)
    assert cfg is not None
        
    # forward pass
    #outputs = model(sample)
    runner = RUNNERS.build(cfg)
    #runner.train()
    runner.test()    
    

#test_bevformer(cfgs[0])
pytest.main(['-s', 'tests/models/detectors/test_bevformer.py'])    
    

