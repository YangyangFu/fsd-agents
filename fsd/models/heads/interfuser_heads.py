"""GRU head for predicting waypoints
"""
import torch
import torch.nn as nn

from mmengine.model import BaseModule
from fsd.registry import HEADS
from fsd.utils import ConfigType, OptConfigType

@HEADS.register_module('interfuser_gru_waypoint')
class GRUWaypointHead(BaseModule):
    """GRU head for predicting waypoints torwards a goal point.
    The 2D goal point from global planner is first projected to a high-dimension (e.g, 64), which is then used as initial hidden state of GRU.
    The GRU input is the queried feature from transformers.
    The GRU output is the latent representation of the differential displacement between waypoints, which is then projected to 2D displacement, 
    and accumulated to get the waypoint.  
    
    Implementation from:
        `Safety-Enhanced Autonomous Driving using Interpretable Sensor Fusion Transformer`
        
    """
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int=64, 
                 num_layers: int = 1, 
                 dropout: float = 0., 
                 batch_first: bool = True,
                 init_cfg: OptConfigType = None):
        super(GRUWaypointHead, self).__init__(init_cfg=init_cfg)
        self.linear1 = nn.Linear(2, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=batch_first)
        self.linear2 = nn.Linear(hidden_size, 2)
        self.batch_first = batch_first
        
    def forward(self, x: torch.Tensor, goal_point: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): with shape (B, L, input_size) if batch_first else (L, B, input_size)
            goal_point (torch.Tensor): with shape (B, 2)

        Returns:
            torch.Tensor: with shape (B, L, 2) if batch_first else (L, B, 2)
        """
        # (B, 2) -> (B, hidden_size) -> (1, B, hidden_size)
        z = self.linear1(goal_point).unsqueeze(0)
        # (B, L, hidden_size)
        output, _ = self.gru(x, z)
        # (B, L, 2) or (L, B, 2)
        output = self.linear2(output)
        
        # accumulate the displacement
        if self.batch_first:
            output = torch.cumsum(output, dim=1)
        else:
            output = torch.cumsum(output, dim=0)
        
        return output

@HEADS.register_module('interfuser_density_map')
class DensityMapHead(BaseModule):
    """Head to predict density map from the output of transformer.

        The paper simply uses a 3-layer MLP to predict the 7 outputs
    """
    # TODO: this seems strange, e.g., only two layers here
    def __init__(self, input_size: int, 
                 hidden_size: int = 64, 
                 output_size: int = 7, 
                 init_cfg: OptConfigType = None):
        super(DensityMapHead, self).__init__(init_cfg=init_cfg)
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(), 
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): with shape (B, L, input_size) if batch_first else (L, B, input_size)

        Returns:
            torch.Tensor: with shape (B, L, output_size) if batch_first else (L, B, output_size)
        """
        return self.mlp(x)

@HEADS.register_module('interfuser_traffic_rule')
class TrafficRuleHead(BaseModule):
    """Traffic rule head to predict traffic rule from the output of transformer.

        The paper simply uses a 1-layer linear layer to predict 2 outputs for each traffic rule (i.e., stop sign, traffic light, and is_junction)
    """

    def __init__(self, input_size: int, 
                 output_size: int = 2,
                 init_cfg: OptConfigType = None):
        super(TrafficRuleHead, self).__init__(init_cfg=init_cfg)
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): with shape (B, input_size)

        Returns:
            torch.Tensor: with shape (B, output_size)
        """
        return self.linear(x)