"""GRU head for predicting waypoints
"""
from typing import List, Optional
import torch
import torch.nn as nn
import warnings 

from mmengine.model import BaseModule
from fsd.registry import HEADS, MODELS
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
                 num_waypoints: int, 
                 input_size: int, 
                 hidden_size: int=64, 
                 num_layers: int = 1, 
                 dropout: float = 0., 
                 batch_first: bool = True,
                 loss_cfg: ConfigType = dict(
                     type='mmdet.SmoothL1Loss', 
                     beta=1.0, 
                     reduction='mean', 
                     loss_weight=1.0),
                 waypoints_weights: Optional[List[float]] = None,
                 init_cfg: OptConfigType = None):
        super(GRUWaypointHead, self).__init__(init_cfg=init_cfg)
        self.linear1 = nn.Linear(2, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=batch_first)
        self.linear2 = nn.Linear(hidden_size, 2)
        self.batch_first = batch_first
        
        # loss fcn
        self.loss_fcn = MODELS.build(loss_cfg)
        # weights for each waypoint
        self.num_waypoints = num_waypoints
        self.waypoints_weights = waypoints_weights
        if self.waypoints_weights is not None:
            assert self.num_waypoints == len(self.waypoints_weights)
            # (L, ) -> (L, 1, 1) or (1, L, 1)
            self.waypoints_weights = torch.Tensor(self.waypoints_weights)
            if self.batch_first:
                self.waypoints_weights = self.waypoints_weights.unsqueeze(0).unsqueeze(-1)
            else:
                self.waypoints_weights = self.waypoints_weights.unsqueeze(-1).unsqueeze(-1)
            
    def forward(self, 
                hidden_states: torch.Tensor, 
                goal_points: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            hidden_states (torch.Tensor): Features from transformer decoder with
                shape (B, L, input_size) if batch_first else (L, B, input_size)
            goal_point (torch.Tensor): with shape (B, 2)

        Returns:
            torch.Tensor: with shape (B, L, 2) if batch_first else (L, B, 2)
        """
        assert hidden_states.dim() == 3, f"hidden_states must have 3 dimensions, got {hidden_states.dim()}"
        assert goal_points.dim() == 2, f"goal_points must have 2 dimensions, got {goal_points.dim()}"
        
        L = hidden_states.size(1) if self.batch_first else hidden_states.size(0)
        assert L == self.num_waypoints, f"Number of waypoints {L} must be equal to the number of waypoints {self.num_waypoints}"
        
        # (B, 2) -> (B, hidden_size) -> (1, B, hidden_size)
        z = self.linear1(goal_points).unsqueeze(0)
        # (B, L, hidden_size)
        output, _ = self.gru(hidden_states, z)
        # (B, L, 2) or (L, B, 2)
        output = self.linear2(output)
        
        # accumulate the displacement
        if self.batch_first:
            output = torch.cumsum(output, dim=1)
        else:
            output = torch.cumsum(output, dim=0)
        
        return output


    def loss(self, hidden_states: torch.Tensor, 
             goal_points: torch.Tensor, 
             target_waypoints: torch.Tensor) -> torch.Tensor:
        pred_waypoints = self(hidden_states, goal_points)
        if self.batch_first:
            B, L, _ = pred_waypoints.size()
            weight = self.waypoints_weights.repeat(B, 1, 2)
        else:
            L, B, _ = pred_waypoints.size()
            weight = self.waypoints_weights.repeat(1, B, 2)
        weight = weight.to(pred_waypoints.device)
        return self.loss_fcn(pred_waypoints, target_waypoints, weight=weight)
    

class ObjectDensityLoss(BaseModule):
    """Loss for object density map prediction in InterFuser

        Density map has a shape of (R, R, 7). The 7 channels are:
            - 0: occupancy at the current grid
            - 1-2: 2-D offset to grid center
            - 3-4: 2-D bounding box
            - 5: heading angle
            - 6: velocity
    """
    def __init__(self, 
                 loss_cfg: ConfigType = dict(
                     type='mmdet.L1Loss', 
                     reduction='mean', 
                     loss_weight=1.0),
                 init_cfg: OptConfigType = None):
        super(ObjectDensityLoss, self).__init__(init_cfg=init_cfg)
        self.loss_fcn = MODELS.build(loss_cfg)
        
    def forward(self, pred: torch.Tensor, 
                target: torch.Tensor, 
                weights: torch.Tensor = torch.Tensor([0.25, 0.25, 0.02])) -> torch.Tensor:
        target_1_mask = target[:, :, 0].ge(0.01)
        target_0_mask = target[:, :, 0].le(0.01)
        target_prob_1 = torch.masked_select(target[:, :, 0], target_1_mask)
        output_prob_1 = torch.masked_select(pred[:, :, 0], target_1_mask)
        target_prob_0 = torch.masked_select(target[:, :, 0], target_0_mask)
        output_prob_0 = torch.masked_select(pred[:, :, 0], target_0_mask)
        if target_prob_1.numel() == 0:
            loss_prob_1 = 0
        else:
            loss_prob_1 = self.loss_fcn(output_prob_1, target_prob_1)
        if target_prob_0.numel() == 0:
            loss_prob_0 = 0
        else:
            loss_prob_0 = self.loss_fcn(output_prob_0, target_prob_0)
        loss_1 = 0.5 * loss_prob_0 + 0.5 * loss_prob_1

        output_1 = pred[target_1_mask][:][:, 1:6]
        target_1 = target[target_1_mask][:][:, 1:6]
        if target_1.numel() == 0:
            loss_2 = 0
        else:
            loss_2 = self.loss_fcn(target_1, output_1)

        # speed pred loss
        output_2 = pred[target_1_mask][:][:, 6]
        target_2 = target[target_1_mask][:][:, 6]
        if target_2.numel() == 0:
            loss_3 = 0
        else:
            loss_3 = self.loss_fcn(target_2, output_2)
        
        # (3, ) * (3, ) -> (3, ) -> ()
        return loss_1*weights[0] + loss_2*weights[1] + loss_3*weights[2]
        
@HEADS.register_module('interfuser_object_density')
class ObjectDensityHead(BaseModule):
    """Head to predict density map from the output of transformer.

        The paper simply uses a 3-layer MLP to predict the 7 outputs
    """
    # TODO: this seems strange, e.g., only two layers here
    def __init__(self, input_size: int, 
                 hidden_size: int = 64, 
                 output_size: int = 7, 
                 loss_cfg: OptConfigType = dict(
                     type='mmdet.L1Loss', 
                     reduction='mean', 
                     loss_weight=1.0
                 ),
                 init_cfg: OptConfigType = None):
        super(ObjectDensityHead, self).__init__(init_cfg=init_cfg)
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(), 
        )

        self.loss_fcn = ObjectDensityLoss(loss_cfg)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): with shape (B, L, input_size) if batch_first else (L, B, input_size)

        Returns:
            torch.Tensor: with shape (B, L, output_size) if batch_first else (L, B, output_size)
        """
        return self.mlp(x)

    def loss(self, x: torch.Tensor, 
             target: torch.Tensor) -> torch.Tensor:
        preds = self(x)
        return self.loss_fcn(preds, target)
    
#TODO: No sigmoid in original paper ???
@HEADS.register_module('interfuser_traffic_rule')
class ClassificationHead(BaseModule):
    """Traffic rule head to predict traffic rule from the output of transformer.

        The paper simply uses a 1-layer linear layer to predict 2 outputs for each traffic rule (i.e., stop sign, traffic light, and is_junction)
    """

    def __init__(self, input_size: int, 
                 output_size: int = 2,
                 loss_cfg: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss', 
                     use_sigmoid=True, 
                     reduction='mean',
                     loss_weight=1.0),
                 init_cfg: OptConfigType = None):
        super(ClassificationHead, self).__init__(init_cfg=init_cfg)
        self.linear = nn.Linear(input_size, output_size)
        
        # loss 
        self.loss_fcn = MODELS.build(loss_cfg)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): with shape (B, input_size)

        Returns:
            torch.Tensor: with shape (B, output_size)
        """
        return self.linear(hidden_states)

    def loss(self, 
             hidden_states: torch.Tensor, 
             target: torch.Tensor) -> torch.Tensor:
        preds = self(hidden_states)
        return self.loss_fcn(preds, target)

@HEADS.register_module('interfuser_heads')     
class InterfuserHead(BaseModule):
    def __init__(self,
                 num_waypoints_queries: int, 
                 num_traffic_rule_queries: int,
                 num_object_density_queries: int,
                 waypoints_head: ConfigType,
                 object_density_head: ConfigType,
                 junction_head: ConfigType,
                 stop_sign_head: ConfigType,
                 traffic_light_head: ConfigType,
                 batch_first: bool = True,
                 init_cfg: OptConfigType = None):
        super(InterfuserHead, self).__init__(init_cfg=init_cfg)

        # number of queries
        self.num_waypoints_queries = num_waypoints_queries
        self.num_traffic_rule_queries = num_traffic_rule_queries
        self.num_object_density_queries = num_object_density_queries
        self.num_queries = self.num_object_density_queries + self.num_traffic_rule_queries + self.num_waypoints_queries
        
        assert waypoints_head['num_waypoints']== self.num_waypoints_queries, \
            f"waypoints_head num_waypoints {waypoints_head['num_waypoints']} must \
                be equal to InterfuserHead num_waypoints_queries {self.num_waypoints_queries}"
        
        
        # heads
        self.waypoints_head = HEADS.build(waypoints_head)
        self.object_density_head = HEADS.build(object_density_head)
        self.junction_head = HEADS.build(junction_head)
        self.stop_sign_head = HEADS.build(stop_sign_head)
        self.traffic_light_head = HEADS.build(traffic_light_head)

        self.batch_first = batch_first
        # if waypoints_head batch_first is not equal to InterfuserHead batch_first, 
        # set to InterfuserHead batch_first
        if not getattr(self.waypoints_head, 'batch_first') or self.waypoints_head.batch_first != self.batch_first:
            warnings.warn(f"waypoints_head batch_first {self.waypoints_head.batch_first} is not equal to InterfuserHead batch_first {self.batch_first}, and will be set to {self.batch_first}")
            self.waypoints_head.batch_first = self.batch_first
     
    def forward(self, hidden_states: torch.Tensor,
                goal_point: torch.Tensor) -> dict:
        """
        Args:
            hidden_states (torch.Tensor): with shape (B, L, input_size)
            goal_point (torch.Tensor): with shape (B, 2)

        Returns:
            dict: with keys:
                - object_density (torch.Tensor): with shape (B, L, 7)
                - junction (torch.Tensor): with shape (B, 2)
                - stop_sign (torch.Tensor): with shape (B, 2)
                - traffic_light (torch.Tensor): with shape (B, 2)
                - waypoints (torch.Tensor): with shape (B, L, 2)
        """
        L = hidden_states.size(1) if self.batch_first else hidden_states.size(0)
        assert L == self.num_queries, f"Number of queries {L} must be equal to the number of queries {self.num_queries}"
        
        if self.batch_first:
            object_density = self.object_density_head(hidden_states[:, :self.num_object_density_queries, :])
            junction = self.junction_head(hidden_states[:, self.num_object_density_queries: self.num_object_density_queries+self.num_traffic_rule_queries, :])
            stop_sign = self.stop_sign_head(hidden_states[:, self.num_object_density_queries: self.num_object_density_queries+self.num_traffic_rule_queries, :])
            traffic_light = self.traffic_light_head(hidden_states[:, self.num_object_density_queries: self.num_object_density_queries+self.num_traffic_rule_queries, :])
            waypoints = self.waypoints_head(hidden_states[:, -self.num_waypoints_queries:, :], goal_point)
            
        else:
            object_density = self.object_density_head(hidden_states[:self.num_object_density_queries, :, :])
            junction = self.junction_head(hidden_states[self.num_object_density_queries: self.num_object_density_queries+self.num_traffic_rule_queries, :, :])
            stop_sign = self.stop_sign_head(hidden_states[self.num_object_density_queries: self.num_object_density_queries+self.num_traffic_rule_queries, :, :])
            traffic_light = self.traffic_light_head(hidden_states[self.num_object_density_queries: self.num_object_density_queries+self.num_traffic_rule_queries, :, :])
            waypoints = self.waypoints_head(hidden_states[-self.num_waypoints_queries:, :, :], goal_point)
        
        return dict(
            object_density=object_density,
            junction=junction,
            stop_sign=stop_sign,
            traffic_light=traffic_light,
            waypoints=waypoints
        )
    
    def loss(self, hidden_states: torch.Tensor, 
                goal_point: torch.Tensor, 
                targets: dict) -> dict:
        
        L = hidden_states.size(1) if self.batch_first else hidden_states.size(0)
        assert L == self.num_queries, f"Number of queries {L} must be equal to the number of queries {self.num_queries}"
        
        # TODO: better way to switch between batch_first and not batch_first
        if self.batch_first:
            loss_density = self.object_density_head.loss(hidden_states[:, :self.num_object_density_queries, :], targets['object_density'])
            loss_junction = self.junction_head.loss(hidden_states[:, self.num_object_density_queries: self.num_object_density_queries+self.num_traffic_rule_queries, :], targets['junction'])
            loss_stop_sign = self.stop_sign_head.loss(hidden_states[:, self.num_object_density_queries: self.num_object_density_queries+self.num_traffic_rule_queries, :], targets['stop_sign'])
            loss_traffic_light = self.traffic_light_head.loss(hidden_states[:, self.num_object_density_queries: self.num_object_density_queries+self.num_traffic_rule_queries, :], targets['traffic_light'])
            loss_waypoints = self.waypoints_head.loss(hidden_states[:, -self.num_waypoints_queries:, :], goal_point, targets['waypoints'])
        else:
            loss_density = self.object_density_head.loss(hidden_states[:self.num_object_density_queries, :, :], targets['object_density'])
            loss_junction = self.junction_head.loss(hidden_states[self.num_object_density_queries: self.num_object_density_queries+self.num_traffic_rule_queries, :, :], targets['junction'])
            loss_stop_sign = self.stop_sign_head.loss(hidden_states[self.num_object_density_queries: self.num_object_density_queries+self.num_traffic_rule_queries, :, :], targets['stop_sign'])
            loss_traffic_light = self.traffic_light_head.loss(hidden_states[self.num_object_density_queries: self.num_object_density_queries+self.num_traffic_rule_queries, :, :], targets['traffic_light'])
            loss_waypoints = self.waypoints_head.loss(hidden_states[-self.num_waypoints_queries:, :, :], goal_point, targets['waypoints'])
        
        loss = dict(
            loss_object_density=loss_density,
            loss_junction=loss_junction,
            loss_stop_sign=loss_stop_sign,
            loss_traffic_light=loss_traffic_light,
            loss_waypoints=loss_waypoints
        )
        
        return loss
    
    def predict(self, hidden_states: torch.Tensor, 
                goal_point: torch.Tensor) -> dict:
        """
        Args:
            hidden_states (torch.Tensor): with shape (B, L, input_size)
            goal_point (torch.Tensor): with shape (B, 2)

        Returns:
            dict: with keys:
                - object_density (torch.Tensor): with shape (B, L, 7)
                - junction (torch.Tensor): with shape (B, 2)
                - stop_sign (torch.Tensor): with shape (B, 2)
                - traffic_light (torch.Tensor): with shape (B, 2)
                - waypoints (torch.Tensor): with shape (B, L, 2)
        """
        return self(hidden_states, goal_point)