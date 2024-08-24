import pytest 
import torch 

from mmengine.config import Config
from mmengine.registry import init_default_scope

from fsd.utils import seed_everything
from fsd.registry import MODELS, AGENTS
from fsd.runner import Runner

@pytest.fixture(autouse=True)
def seed():
    seed_everything(2024)


# test three modes
cfgs = ['fsd/agents/InterFuser/configs/interfuser_r50_carla.py']
init_default_scope('fsd')

@pytest.mark.parametrize('cfg', cfgs)
def test_agent(cfg):
    # cfg
    cfg = Config.fromfile(cfg)
    assert cfg is not None

    init_default_scope('fsd')
    # separate building
    dataloader = Runner.build_dataloader(cfg.train_dataloader)
    data_preprocessor = MODELS.build(cfg.model.data_preprocessor)
    agent = AGENTS.build(cfg.model)    

    # get one sample
    sample = next(iter(dataloader))
    sample = data_preprocessor(sample)

    ## forward pass
    outputs = agent(**sample, mode='predict')
    object_density = outputs['object_density']
    junction = outputs['junction']
    stop_sign = outputs['stop_sign']
    traffic_light = outputs['traffic_light']
    waypoints = outputs['waypoints']

    # check the output shapes
    assert object_density.shape == (2, 400, 7)
    assert stop_sign.shape == (2, 1, 2)
    assert junction.shape == (2, 1, 2)
    assert traffic_light.shape == (2, 1, 2)
    assert waypoints.shape == (2, 10, 2)


    # loss calculation    
    loss = agent(**sample, mode='loss')
    assert loss.keys() == {'loss_object_density', 
                           'loss_junction', 
                           'loss_stop_sign', 
                           'loss_traffic_light', 
                           'loss_waypoints'}
     
pytest.main(['-s', 'tests/agents/InterFuser/test_interfuser.py'])