"""GRU head for predicting waypoints
"""
import torch
import torch.nn as nn

from mmengine.model import BaseModule
from fsd.registry import HEADS
from fsd.utils import ConfigType, OptConfigType

@HEADS.register_module()
class GRUWaypointHead(BaseModule):
    """GRU head for predicting waypoints
    """
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int, 
                 dropout: float, 
                 init_cfg: OptConfigType = None):
        super(GRUWaypointHead, self).__init__(init_cfg=init_cfg)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): with shape (N, T, input_size)

        Returns:
            torch.Tensor: with shape (N, 2)
        """
        output, _ = self.gru(x)
        return self.fc(output[-1])