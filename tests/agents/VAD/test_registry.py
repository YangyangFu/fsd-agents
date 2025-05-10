from mmengine.config import Config
from mmengine.registry import init_default_scope
from fsd.registry import MODELS, RUNNERS
from fsd.runner import Runner

init_default_scope('fsd')

cfg = Config.fromfile('fsd/configs/VAD/test_only.py')

def test_model_registry():
    model = MODELS.build(cfg.model)
    assert model is not None
    

def test_dataloader_registry():
    dataset = Runner.build_dataloader(cfg.train_dataloader)
    assert dataset is not None


def test_dataloader():
    dataloader = Runner.build_dataloader(cfg.test_dataloader)
    for i, data in enumerate(dataloader):
        if i > 10:
            break
        print(len(data['data_samples']))
        #assert data is not None

def test_train():
    runner = RUNNERS.build(cfg)
    #runner.train()
    runner.test()
    
#test_model_registry()
#test_dataloader_registry()
#test_dataloader()
test_train()