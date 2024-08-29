"""
Script to conver the original InterFuser checkpoint to the format used in the FSD codebase.
To use this script:
1. Download the original checkpoint from "http://43.159.60.142/s/p2CN" to the `checkpoints` folder
2. Rename the downloaded checkpoint to `ori_interfuser.pth.tar`
3. RUn this script, which will save the converted checkpoint as `interfuser.pth.tar` in the same folder

"""

import os
import torch 

from mmengine.config import Config
from mmengine.registry import init_default_scope
from fsd.runner import Runner
from fsd.registry import MODELS, AGENTS

# create agent model and output model architecture
file_path = os.path.dirname(os.path.realpath(__file__))
agent_dir = os.path.dirname(file_path)
checkpoint_dir = os.path.join(agent_dir, 'ckpts')
config_path = os.path.join(agent_dir, 'configs/interfuser_r50_carla.py')
cfg = Config.fromfile(config_path)

init_default_scope('fsd')
agent = AGENTS.build(cfg.model)
agent_state = agent.state_dict()

# load original interfuser checkpoint
raw_state = torch.load(os.path.join(checkpoint_dir, "ori_interfuser.pth.tar"))
raw_state = raw_state['state_dict']


# create mapping rules
# ori_interfuser -> agent
custom_mapping = {
            'rgb_patch_embed.proj.weight': 'img_neck.conv.weight',
            'rgb_patch_embed.proj.bias': 'img_neck.conv.bias',
            'lidar_patch_embed.proj.weight': 'pts_neck.conv.weight',
            'lidar_patch_embed.proj.bias': 'pts_neck.conv.bias',
            'global_embed': 'multi_view_mean_encoding.weight',
            'view_embed': 'multi_view_encoding.weight',
            'query_pos_embed': 'query_positional_encoding.weight',
            'query_embed': 'query_embedding.weight',
            'waypoints_generator.gru.weight_ih_l0': 'heads.waypoints_head.gru.weight_ih_l0',
            'waypoints_generator.gru.weight_hh_l0': 'heads.waypoints_head.gru.weight_hh_l0',
            'waypoints_generator.gru.bias_ih_l0':  'heads.waypoints_head.gru.bias_ih_l0',
            'waypoints_generator.gru.bias_hh_l0': 'heads.waypoints_head.gru.bias_hh_l0',
            'waypoints_generator.encoder.weight': 'heads.waypoints_head.linear1.weight',
            'waypoints_generator.encoder.bias': 'heads.waypoints_head.linear1.bias',
            'waypoints_generator.decoder.weight': 'heads.waypoints_head.linear2.weight',
            'waypoints_generator.decoder.bias': 'heads.waypoints_head.linear2.bias',
            'junction_pred_head.weight': 'heads.junction_head.linear.weight',
            'junction_pred_head.bias': 'heads.junction_head.linear.bias',
            'traffic_light_pred_head.weight': 'heads.traffic_light_head.linear.weight',
            'traffic_light_pred_head.bias': 'heads.traffic_light_head.linear.bias',
            'stop_sign_head.weight': 'heads.stop_sign_head.linear.weight',
            'stop_sign_head.bias': 'heads.stop_sign_head.linear.bias',
            'traffic_pred_head.0.weight': 'heads.object_density_head.mlp.0.weight',
            'traffic_pred_head.0.bias': 'heads.object_density_head.mlp.0.bias',
            'traffic_pred_head.2.weight': 'heads.object_density_head.mlp.2.weight',
            'traffic_pred_head.2.bias': 'heads.object_density_head.mlp.2.bias'
}

reverse_custom_mapping = {v: k for k, v in custom_mapping.items()}


# transformer key mapping: agent -> ori_interfuser
def transformer_mapping(query_keys):
    transformer_keys = {}
    for key in query_keys:
        if key.startswith('encoder') or key.startswith('decoder'):
            if 'attentions.0.attn' in key:
                newkey = key.replace('attentions.0.attn', 'self_attn')
            elif 'attentions.1.attn' in key:
                newkey = key.replace('attentions.1.attn', 'multihead_attn')
            elif 'ffns.0.layers.0.0' in key:
                newkey = key.replace('ffns.0.layers.0.0', 'linear1')
            elif 'ffns.0.layers.1' in key:
                newkey = key.replace('ffns.0.layers.1', 'linear2')
            elif 'norms.0' in key:
                newkey = key.replace('norms.0', 'norm1')
            elif 'norms.1' in key:
                newkey = key.replace('norms.1', 'norm2')
            elif 'norms.2' in key:
                newkey = key.replace('norms.2', 'norm3')
            elif 'decoder_norm' in key:
                newkey = key.replace('decoder_norm', 'decoder.norm')
            else:
                newkey = key
                
            transformer_keys[key] = newkey

    return transformer_keys

transformer_keys = transformer_mapping(agent_state.keys())

# backbone resnet key mapping: no need as pretrained weights are directly used
resnet_keys = {}
for key in agent_state.keys():
    if key.startswith('img_backbone') or key.startswith('pts_backbone'):
        resnet_keys[key] = key

# assemble all mappings
all_mappings = {**reverse_custom_mapping, **transformer_keys}

converted_state = {}
for key in all_mappings.keys():
    converted_state[key] = raw_state[all_mappings[key]].to('cpu')
for key in resnet_keys.keys():
    converted_state[key] = agent_state[key].to('cpu')

# reshape some weights
for key in converted_state.keys():
    if key in ['multi_view_encoding.weight', 'multi_view_mean_encoding.weight',
        'query_positional_encoding.weight']:
        converted_state[key] = converted_state[key].squeeze().permute(1, 0)
        print(converted_state[key].shape)

    if key == 'query_embedding.weight':
        converted_state[key] = converted_state[key].squeeze()
        print(converted_state[key].shape)

# save converted weights
out = {}
out['state_dict'] = converted_state
torch.save(out, os.path.join(checkpoint_dir, 'interfuser.pth.tar'))