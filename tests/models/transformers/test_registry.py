from fsd.registry import TRANSFORMERS


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


cfg = dict(
    type='DETRLayer',
    attn_cfgs=attn_cfgs,
    ffn_cfgs=ffn_cfgs,
    operation_order=operation_order,
    batch_first=batch_first
)

model = TRANSFORMERS.build(cfg)
print(model)


