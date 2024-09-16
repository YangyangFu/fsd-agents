from mmengine import Config
from mmengine.registry import init_default_scope
from fsd.registry import RUNNERS, AGENTS

#cfg = Config.fromfile('fsd/agents/InterFuser/configs/interfuser_r50_carla.py')
cfg = Config.fromfile('fsd/configs/InterFuser/interfuser_r50_carla.py')
init_default_scope('fsd')
print(cfg.default_scope)

runner = RUNNERS.build(cfg)

runner.train()