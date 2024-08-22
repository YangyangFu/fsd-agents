import pytest 
import torch 

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData, BaseDataElement

from fsd.utils import get_agent_cfg, seed_everything
from fsd.structures import PlanningDataSample
from fsd.registry import RUNNERS, DATASETS, MODELS, AGENTS
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

    # separate building
    dataloader = Runner.build_dataloader(cfg.train_dataloader)
    agent = AGENTS.build(cfg.model)    

    # get one sample
    sample = next(iter(dataloader))
    

    ## forward pass
    outputs = agent(sample, mode='predict')
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

    print(junction)
    
    # loss calculation

    
    
    #losses = agent(inputs, batch_target_dict=targets, mode='loss')

    #print(losses)

    #predictions = agent(inputs, batch_target_dict=None, mode='predict')
    #print(predictions['junction'])
    
    # test train_step
    
    # test val_step
    #out = agent.val_step(inputs)
    
test_agent(cfgs[0])   
#pytest.main(['-s', 'tests/agents/InterFuser/test_interfuser.py'])