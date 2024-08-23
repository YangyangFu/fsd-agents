from mmengine import Config
from mmengine.registry import init_default_scope
from fsd.registry import RUNNERS, AGENTS

cfg = Config.fromfile('fsd/agents/InterFuser/configs/interfuser_r50_carla.py')
init_default_scope('fsd')

runner = RUNNERS.build(cfg)

runner.train()