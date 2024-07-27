import torch
import torch.nn as nn
from fsd.registry import HEADS 

# head config
cfg = dict(
    type='GRUWaypointHead',
    input_size=2048,
    hidden_size=256,
    num_layers=2,
    dropout=0.3
)

head = HEADS.build(cfg=cfg)

# inputs
inputs = torch.randn(2, 10, 2048)

# run forward
outputs = head(inputs)

print(outputs.shape)