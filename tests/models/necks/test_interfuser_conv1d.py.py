
import torch 
from fsd.registry import NECKS

# image backbone features (3, 224, 224)
feats = torch.randn(2, 2048, 7, 7)


# model config
cfg = dict(
    type='InterFuserNeck',
    in_channels=2048,
    out_channels=256
)

# build model
model = NECKS.build(cfg=cfg)

# run forward
outputs = model(feats)

print(outputs.shape)


