import pytest
import torch 
import copy 

from fsd.models import (DETRLayer, DETRLayerSequence, \
    DeformableDETRLayer,DeformableDETRLayerSequence)

def test_layer_cpu():
    embed_dims = 256
    batch_first = True
    attn_cfgs = dict(
        type='MultiheadAttention',
        embed_dims=embed_dims,
        num_heads=8,
        attn_drop=0.,
        proj_drop=0.,
        dropout_layer=dict(type='Dropout', drop_prob=0.),
        batch_first=batch_first
    )
    
    ffn_cfgs = dict(
        type='FFN',
        embed_dims=embed_dims,
        feedforward_channels=1024,
        num_fcs=2,
        ffn_drop=0.1,
        act_cfg=dict(type='ReLU', inplace=True)
    )
    
    operation_order = ['self_attn', 'norm', 'ffn', 'norm']
    
    layer = DETRLayer(
        attn_cfgs=attn_cfgs,
        ffn_cfgs=ffn_cfgs,
        operation_order=operation_order,
        batch_first=batch_first
    )
    
    assert layer is not None
    assert layer.batch_first is True 
    assert len(layer.ffns) == 1
    assert layer.ffns[0].feedforward_channels == 1024

    feedforward_channels = 2048
    layer = DETRLayer(
        attn_cfgs=attn_cfgs,
        ffn_cfgs=ffn_cfgs,
        operation_order=operation_order,
        batch_first=batch_first,
        feedforward_channels=feedforward_channels
    )
    
    assert layer is not None
    assert layer.batch_first is True
    assert len(layer.ffns) == 1
    assert layer.ffns[0].feedforward_channels == 2048
    
    query = torch.randn(2, 4, embed_dims)
    outputs = layer(query)
    assert outputs.shape == (2, 4, embed_dims)
    
    query = torch.randn(2, 8, embed_dims)
    key = torch.randn(2, 4, embed_dims)
    outputs = layer(query, key)
    assert outputs.shape == (2, 8, embed_dims)

@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_layer_cuda():
    operation_order = ['self_attn', 'norm', 'ffn', 'norm']
    embed_dims = 256
    batch_first = True
    layer = DETRLayer(
        attn_cfgs=dict(
            type='MultiheadAttention',
            embed_dims=embed_dims,
            num_heads=8,
            attn_drop=0.,
            proj_drop=0.,
            dropout_layer=dict(type='Dropout', drop_prob=0.),
            batch_first=batch_first
        ),
        operation_order=operation_order,
        batch_first=batch_first
    )
    
    seq = torch.nn.ModuleList([copy.deepcopy(layer) for _ in range(6)])
    seq.to('cuda')
    
    x = torch.randn(2, 4, embed_dims).to('cuda')
    for l in seq:
        x = l(x)
        assert x.shape == (2, 4, embed_dims)

if __name__ == '__main__':
    test_layer_cpu()
    test_layer_cuda()
    