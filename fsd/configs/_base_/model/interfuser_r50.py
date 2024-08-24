EMBED_DIMS = 256
BATCH_FIRST = True
PLANNING_STEPS = 10

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
        type='Conv1d',
        in_channels=2048,
        out_channels=EMBED_DIMS
    ),
    pts_neck=dict(
        type='Conv1d',
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
    heads=dict(
        type='interfuser_heads',
        num_waypoints_queries=PLANNING_STEPS,
        num_traffic_rule_queries=1,
        num_object_density_queries=400,
        waypoints_head=dict(
            type='interfuser_gru_waypoint',
            num_waypoints=10,
            input_size=EMBED_DIMS,
            hidden_size=64,
            num_layers=1,
            dropout=0.,
            #batch_first=True,
            loss_cfg=dict(
                type='SmoothL1Loss',
                _scope_='mmdet',
                beta=1.0,
                reduction='mean',
                loss_weight=1.0
            ),
            waypoints_weights=[
                0.1407441030399059,
                0.13352157985305926,
                0.12588535273178575,
                0.11775496498388233,
                0.10901991343009122,
                0.09952110967153563,
                0.08901438656870617,
                0.07708872007078788,
                0.06294267636589287,
                0.04450719328435308,
            ]),
        object_density_head=dict(
            type='interfuser_object_density',
            input_size=EMBED_DIMS,
            hidden_size=64,
            output_size=7,
            loss_cfg=dict(
                type='L1Loss',
                _scope_='mmdet',
                reduction='mean',
                loss_weight=1.0
            )
        ),
        junction_head=dict(
            type='interfuser_traffic_rule',
            input_size=EMBED_DIMS,
            output_size=2,
            loss_cfg=dict(
                type='CrossEntropyLoss',
                _scope_='mmdet',
                use_sigmoid=True, # binary classification
                reduction='mean',
                loss_weight=1.0
            )
        ),
        stop_sign_head=dict(
            type='interfuser_traffic_rule',
            input_size=EMBED_DIMS,
            output_size=2,
            loss_cfg=dict(
                type='CrossEntropyLoss',
                _scope_='mmdet',
                use_sigmoid=True, # binary classification
                reduction='mean',
                loss_weight=1.0
            )
        ),
        traffic_light_head=dict(
            type='interfuser_traffic_rule',
            input_size=EMBED_DIMS,
            output_size=2,
            loss_cfg=dict(
                type='CrossEntropyLoss',
                _scope_='mmdet',
                use_sigmoid=True, # binary classification
                reduction='mean',
                loss_weight=1.0
            )
        )
    ),        
    positional_encoding=dict(
        num_feats=EMBED_DIMS//2,
        normalize=True
    ), 
    multi_view_encoding=dict(
        num_embeddings=5,
        embedding_dim=EMBED_DIMS
    ),
    data_preprocessor=dict(
        type="InterFuserDataPreprocessor")
)