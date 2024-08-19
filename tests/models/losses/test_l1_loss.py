import torch
import pytest
from fsd.registry import MODELS


def test_masked_l1_loss():
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


def test_masked_smooth_l1_loss():
    preds = torch.ones(2, 2)
    targets = 2*torch.ones(2, 2)

    loss_cfg = dict(type='MaskedSmoothL1Loss',
                    beta=1.0,
                    reduction='mean',
                    loss_weight=1.0)

    loss_fcn = MODELS.build(loss_cfg)
    loss = loss_fcn(preds, targets)

    assert loss - 0.5 < 1e-6

    mask = torch.tensor([[1, 0], [1, 1]])
    loss = loss_fcn(preds, targets, mask=mask)
    assert loss - 0.5 < 1e-6
    