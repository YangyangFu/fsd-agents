import os
import torch 

from mmengine.config import Config
from mmengine.registry import init_default_scope

from fsd.runner import Runner
from fsd.registry import MODELS, AGENTS 

file_path = os.path.dirname(os.path.realpath(__file__))

cfg = Config.fromfile(os.path.join(os.path.dirname(file_path), 
                                   "configs/interfuser_r50_carla.py"))
init_default_scope('fsd')

# separate building
dataloader = Runner.build_dataloader(cfg.train_dataloader)
data_preprocessor = MODELS.build(cfg.model.data_preprocessor)
# TODO: data_processor should be part of the model. 
# But here it seems I need build data_preprocessor first 
# to get the sample shape.
agent = AGENTS.build(cfg.model)    

# save model weights
agent.load_state_dict(torch.load(os.path.join(file_path,'interfuser.pth.tar')))

# get one sample
for sample in dataloader:
    sample = data_preprocessor(sample)
    out = agent(**sample, mode='predict')

    print(out)
    
    break