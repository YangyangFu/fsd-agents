import os
import json

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.analysis import get_model_complexity_info 
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

# get one sample
sample = next(iter(dataloader))
sample = data_preprocessor(sample)

# Seems not support multi-modality inputs
inputs_dict = sample['inputs']

# The original inputs_dict has keys for meta data, which cannot be passed for complexity analysis
# This will not work as inputs_dict will be flattend before forward call, which 
inputs_dict= {'img': inputs_dict['img'], 
              'pts': inputs_dict['pts'], 
              'goal_points': inputs_dict['goal_points']}

inputs = (inputs_dict, None, 'predict')
#inputs = ((inputs_dict['img'], inputs_dict['pts'], inputs_dict['goal_points']), None, 'predict')


model_complexity_info = get_model_complexity_info(agent, inputs=inputs)

print(model_complexity_info.keys())

out = model_complexity_info['out_table']
model_arch = model_complexity_info['out_arch']

with open(os.path.join(file_path, 'model_complexity.json'), 'w') as f:
    json.dump(out, f)

with open(os.path.join(file_path, 'model_arch.json'), 'w') as f:
    json.dump(model_arch, f)