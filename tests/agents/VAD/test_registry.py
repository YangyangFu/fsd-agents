from mmengine.config import Config
from mmengine.registry import init_default_scope
from fsd.registry import MODELS, RUNNERS
from fsd.runner import Runner

init_default_scope('fsd')

cfg = Config.fromfile('tests/agents/VAD/config.py')

def test_model_registry():
    model = MODELS.build(cfg.model)
    assert model is not None
    

def test_dataloader_registry():
    dataset = Runner.build_dataloader(cfg.train_dataloader)
    assert dataset is not None


def test_dataloader():
    dataloader = Runner.build_dataloader(cfg.train_dataloader)
    
    sample = next(iter(dataloader))
    assert sample is not None



#test_model_registry()
#test_dataloader_registry()
test_dataloader()