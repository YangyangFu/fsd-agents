from torch.nn.modules.activation import ReLU

#_base_ = [
#    '../../../configs/_base_/datasets/coco_detection.py',
#    '../../../configs/_base_/default_runtime.py'
#]

EMBED_DIMS = 256
BATCH_FIRST = True

model = dict(
    type='InterFuser',
    num_queries=411,
    embed_dims=EMBED_DIMS,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=[3],
        deep_stem=True,
        frozen_stages=4,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    pts_backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=[3],
        deep_stem=True,
        frozen_stages=4,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')
    ),
    img_neck=dict(
        type='InterFuserNeck',
        in_channels=2048,
        out_channels=EMBED_DIMS
    ),
    pts_neck=dict(
        type='InterFuserNeck',
        in_channels=512,
        out_channels=EMBED_DIMS
    ),
    encoder = dict( # DetrTransformerEncoder
        type='DETRLayerSequence',
        num_layers=6,
        layer_cfgs=dict(
            type='DETRLayer',
            attn_cfgs=dict( # MultiheadAttention
                type='MultiheadAttention',
                embed_dims=EMBED_DIMS,
                num_heads=8,
                attn_drop=0.,
                proj_drop=0.
            ),
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=EMBED_DIMS,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)
            ),
            operation_order=['self_attn', 'norm', 'ffn', 'norm'],
            batch_first=BATCH_FIRST
        )
    ),       
    decoder = dict(  # DetrTransformerDecoder
        type='DETRLayerSequence',
        num_layers=6,
        layer_cfgs=dict(
            type='DETRLayer',
            attn_cfgs=dict( # MultiheadAttention
                type='MultiheadAttention',
                embed_dims=EMBED_DIMS,
                num_heads=8,
                attn_drop=0.,
                proj_drop=0.,
            ),
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=EMBED_DIMS,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)
            ),
            operation_order=['self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'],
            batch_first=BATCH_FIRST
        )
    ),
    
    positional_encoding=dict(
        num_feats=EMBED_DIMS//2,
        normalize=True
    ), 
    multi_view_encoding=dict(
        num_embeddings=5,
        embedding_dim=EMBED_DIMS
    )
)
