from mmengine.config import Config
from mmengine.registry import init_default_scope

from fsd.registry import MODELS 
init_default_scope('fsd')

cfg = Config.fromfile('tests/agents/VAD/config.py')

def test_registry():
    model = MODELS.build(cfg.model)
    assert model is not None
    

test_registry()