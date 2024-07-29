import torch 
from fsd.registry import AGENTS
from mmengine.config import Config

cfg = Config.fromfile('interfuser_r50_carla.py')
agent = AGENTS.build(cfg.model)

## generate some fake inputs
# 4 camera images
imgs = torch.randn(2, 4, 3, 224, 224)
# 1 lidar bev image
pts = torch.randn(2, 3, 224, 224)
# goal points
goal_points = torch.randn(2, 2)
inputs = dict(imgs=imgs, pts=pts, goal_points=goal_points)

## forward pass
output_dec, density_map, stop_sign, is_junction, traffic_light, waypoints = agent(inputs)
print(stop_sign.shape, is_junction.shape, traffic_light.shape, waypoints.shape)
# check the output shapes
assert output_dec.shape == (2, 411, 256)
assert density_map.shape == (2, 400, 7)
assert stop_sign.shape == (2, 1, 2)
assert is_junction.shape == (2, 1, 2)
assert traffic_light.shape == (2, 1, 2)
assert waypoints.shape == (2, 10, 2)

