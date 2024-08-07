import pytest 
import torch 

from fsd.utils import get_agent_cfg, seed_everything
from fsd.registry import AGENTS

@pytest.fixture(autouse=True)
def seed():
    seed_everything(2024)


# test three modes
cfgs = ['fsd/agents/InterFuser/configs/interfuser_r50_carla.py']

@pytest.mark.parametrize('cfg', cfgs)
def test_agent(cfg):
    # build agent
    cfg = get_agent_cfg(cfg)
    assert cfg is not None
    agent = AGENTS.build(cfg)    
    
    #4 camera images
    imgs = torch.randn(2, 4, 3, 224, 224)
    # 1 lidar bev image
    pts = torch.randn(2, 3, 224, 224)
    # goal points
    goal_points = torch.randn(2, 2)
    inputs = dict(imgs=imgs, pts=pts, goal_points=goal_points)

    ## forward pass
    outputs = agent(inputs, mode='tensor')
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
    targets = {
        'waypoints': torch.randn(2, 10, 2),
        'object_density': torch.randn(2, 400, 7),
        'junction': torch.randint(0, 2, (2, 1, 2)),
        'stop_sign': torch.randint(0, 2, (2, 1, 2)),
        'traffic_light': torch.randint(0, 2, (2, 1, 2)),
    }
    losses = agent(inputs, data_samples=None, mode='loss', targets = targets)

    print(losses)

    predictions = agent(inputs, data_samples=None, mode='predict')
    print(predictions['junction'])
    
    # test train_step
    
    # test val_step
    out = agent.val_step((inputs, None))
    
    
pytest.main(['-s', 'tests/agents/test_interfuser.py'])