# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.models.losses.utils import weighted_loss
from fsd.registry import MODELS

def masked_loss(loss_func: Callable) -> Callable:
    """Decorator to make a loss function support mask.
        mask (Tensor, optional): The mask of loss for each prediction. 
        If any element is masked, the loss of this element will be calculated.
        Defaults to None.

    Args:
        loss_func (Callable): The loss function to be wrapped.

    Returns:
        Callable: The wrapped loss function.
    """

    def _inner(pred: Tensor, target: Tensor, mask: Optional[Tensor] = None, *args, **kwargs) -> Tensor:
        if mask is not None:
            pred = pred * mask
            target = target * mask
        return loss_func(pred, target, *args, **kwargs)

    return _inner


@weighted_loss
@masked_loss
def smooth_l1_loss(pred: Tensor, target: Tensor, beta: float = 1.0) -> Tensor:
    """Smooth L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


@weighted_loss
@masked_loss
def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Masked L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0
    
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    
    return loss


@MODELS.register_module()
class MaskedSmoothL1Loss(nn.Module):
    """Masked Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 beta: float = 1.0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                mask: Optional[Tensor] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            mask (Tensor, optional): The mask of loss for each prediction.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # masks
        if mask is None:
            mask = torch.ones_like(pred)
        
        # masked average factor
        if avg_factor is None:
            avg_factor = mask.sum() 
        else:
            avg_factor = avg_factor * mask.sum()

        # smooth l1 loss
        loss = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            mask = mask,
            **kwargs)
        return loss


@MODELS.register_module()
class MaskedL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                mask: Optional[Tensor] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            mask (Tensor, optional): The mask of loss for each prediction.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        
        if mask is None:
            mask = torch.ones_like(pred)
        
        # masked average factor
        if avg_factor is None:
            avg_factor = mask.sum() 
        else:
            avg_factor = avg_factor * mask.sum()
        
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor, mask=mask)
        return loss
