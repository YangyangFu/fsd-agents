import pytest

import torch
from mmdet3d.structures import LiDARInstance3DBoxes
from fsd.agents import InterFuserDensityMap
from fsd.registry import AGENT_TRANSFORMS, TRANSFORMS
from mmengine.registry import init_default_scope

bboxes = torch.rand(10, 9)
bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9, with_yaw=True)

def test_density_map():
    
    dmap_cfg = dict(
        type="InterFuserDensityMap",
        bev_range=[0, 20, -10, 10],
        pixels_per_meter=8
    )
    init_default_scope('fsd')
    dmap = AGENT_TRANSFORMS.build(dmap_cfg)
    
    inputs = {}
    inputs['anno_info'] = {'gt_bboxes_3d': bboxes}
    results = dmap(inputs)
    
    print(results['anno_info']['gt_grid_density'])

pytest.main(['-q', 'tests/agents/InterFuser/test_density_map.py'])