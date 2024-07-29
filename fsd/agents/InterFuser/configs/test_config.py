import torch 
from fsd.registry import AGENTS
from mmengine.config import Config

cfg = Config.fromfile('/home/yyf/github/FSDagents/fsd/agents/InterFuser/configs/interfuser_r50_carla.py')
agent = AGENTS.build(cfg.model)

## generate some fake inputs
# 4 camera images
imgs = torch.randn(2, 4, 3, 224, 224)
# 1 lidar bev image
pts = torch.randn(2, 3, 224, 224)
inputs = dict(imgs=imgs, pts=pts)

## forward pass
outputs = agent(inputs)

# check the output shapes
print(outputs.shape)