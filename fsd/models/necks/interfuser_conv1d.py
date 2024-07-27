
import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmdet3d.utils import OptConfigType
from fsd.registry import NECKS

@NECKS.register_module()
class InterFuserNeck(BaseModule):
    """Interfuser feature neck.

        A simple 1x1 convolutional layer to project the input features to a given dimension.
    """
    def __init__(self, in_channels: int, out_channels: int, init_cfg: OptConfigType = None):
        super(InterFuserNeck, self).__init__(init_cfg=init_cfg)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): with shape (N, in_channels, H, W)

        Returns:
            torch.Tensor: with shape (N, out_channels, H, W)
        """
        
        return self.conv(x)
