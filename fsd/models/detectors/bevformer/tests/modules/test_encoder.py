import torch
import numpy as np

from fsd.registry import MODELS
from mmengine.registry import init_default_scope
#from fsd.models.detectors.bevformer.modules.encoder import BEVFormerEncoder

def test_bevformer_encoder_forward():

    embed_dims = 256
    num_levels = 4
    ffn_dim = embed_dims * 2
    
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    
    init_default_scope('fsd')
    
    cfg=dict(
    type='BEVFormerEncoder',
    num_layers=6,
    pc_range=point_cloud_range,
    num_points_in_pillar=4,
    return_intermediate=False,
    transformerlayers=dict(
        type='BEVFormerLayer',
        attn_cfgs=[
            dict(
                type='TemporalSelfAttention',
                embed_dims=embed_dims,
                num_levels=1),
            dict(
                type='SpatialCrossAttention',
                pc_range=point_cloud_range,
                deformable_attention=dict(
                    type='MultiScaleDeformableAttention3D',
                    embed_dims=embed_dims,
                    num_points=8,
                    num_levels=num_levels),
                embed_dims=embed_dims,
            )
        ],
        feedforward_channels=ffn_dim,
        ffn_dropout=0.1,
        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                            'ffn', 'norm')))
    
    encoder = MODELS.build(cfg)
    
    print(encoder)
    
    h = [50, 20, 10, 5]
    w = [50, 20, 10, 5]
    bev_h, bev_w = 200, 200
    spatial_shapes = torch.tensor([(hh, ww) for hh, ww in zip(h, w)]).view(-1, 2)
    level_start_index = [0] * num_levels
    for i in range(1, num_levels):
        level_start_index[i] = level_start_index[i - 1] + h[i - 1] * w[i - 1]
    
    print(spatial_shapes, level_start_index)
    num_keys = sum([hh * ww for hh, ww in zip(h, w)])
    num_query = bev_h * bev_w
    bs = 2
    num_cams = 6
    bev_query = torch.randn(num_query, bs, embed_dims)   # (num_query, bs, embed_dims)
    key = torch.randn(num_cams, num_keys, bs, embed_dims)      # (num_cam, num_value, bs, embed_dims)
    bev_pos = torch.randn(num_query, bs, embed_dims)  # (num_query, bs, embed_dims)

    # dummy image meta: each entry has at least 'img_shape' and 'lidar2img'
    lidar2img = np.eye(4)
    img_shapes = [(928, 1600, 3) for _ in range(num_cams)]  # single camera shape
    lidar2imgs = [lidar2img for _ in range(num_cams)]
    
    img_metas = [
        {
            'img_shape': img_shapes,  # camera shape
            'lidar2img': lidar2imgs     # identity transform
        } for _ in range(bs)             # batch size of 2
    ]
    
    shift = torch.randn(bs, 2)
    output = encoder(
        bev_query,
        key,
        key,
        bev_h=bev_h,
        bev_w=bev_w,
        bev_pos=bev_pos,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        img_metas=img_metas,
        shift=shift
    )
    
    # Since return_intermediate=True, output should have shape:
    # (num_layers, num_query, bs, embed_dims)
    assert len(output.shape) == 3
    assert output.shape == (bs, num_query, embed_dims)


if __name__ == '__main__':
    test_bevformer_encoder_forward()
    print("All tests passed!")