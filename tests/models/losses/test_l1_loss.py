import torch
from fsd.registry import MODELS

preds = torch.ones(2, 2)
targets = 2*torch.ones(2, 2)
mask = torch.ones(2, 2)

loss_cfg = dict(type='MaskedL1Loss',
                reduction='mean',
                loss_weight=1.0)

loss_fcn = MODELS.build(loss_cfg)
loss = loss_fcn(preds, targets, mask=mask)

assert loss - 1.0 < 1e-6

mask = torch.tensor([[1, 0], [1, 1]])
loss = loss_fcn(preds, targets, mask=mask)
assert loss - 1.0 < 1e-6

mask = torch.tensor([[1, 0], [0, 0]])
loss = loss_fcn(preds, targets, mask=mask)
assert loss - 1.0 < 1e-6