from typing import Mapping, Optional, Sequence, Union
import torch 

from mmengine.model import BaseDataPreprocessor
from .utils import generate_density_map
from fsd.registry import MODELS

@MODELS.register_module()
class InterFuserDataPreprocessor(BaseDataPreprocessor):
    """Preprocess data for interfuser model by doing the following preprocssing:
    - generate object density map
    - generate target dictionary
        - density map
        - waypoints
        - traffici light, stop sign, junction
    
    """
    def __init__(self, 
                 bev_range: Sequence[float], 
                 pixels_per_meter: Optional[int] = 8,
                 non_blocking: Optional[bool] = False):
        """_summary_

        Args:
            bev_range (Sequence[float]): [xmin, xmax, ymin, ymax]
            pixels_per_meter (Optional[int], optional): Defaults to 8.
            non_blocking (Optional[bool], optional): Whether block current process
                when transferring data to device. New in version v0.3.0. Defaults to False.
        """
        super().__init__(non_blocking=non_blocking)
        self.bev_range = bev_range
        self.pixels_per_meter = pixels_per_meter
        
    def forward(self, data):
        data_samples = data['data_sample']
        
        # get object density map, waypoints and others
        density_maps = []
        waypoints = []
        waypoints_masks = []
        affected_by_lights = []
        affected_by_signs = []
        affected_by_junctions = []
        goal_points = []
        
        for sample in data_samples:
            gt_instances = sample.gt_instances_3d
            density_map = generate_density_map(gt_instances, self.bev_range, self.pixels_per_meter)
            density_maps.append(density_map)
            
            gt_ego = sample.gt_ego
            waypoints.append(gt_ego.gt_ego_future_traj.xy.squeeze(0)) # (2, L)
            waypoints_masks.append(gt_ego.gt_ego_future_traj.mask.squeeze(0)) # (L)
            goal_points.append(gt_ego.gt_ego_future_traj.xy[..., -1].squeeze(0)) # (2)
            #affected_by_lights.append(1 if sample['gt_ego']['affected_by_traffic_light'] else 0)
            #affected_by_signs.append(1 if sample['gt_ego']['affected_by_stop_sign'] else 0)
            #affected_by_junctions.append(1 if sample['gt_ego']['affected_by_junction'] else 0)

            affected_by_lights.append(0)
            affected_by_signs.append(0)
            affected_by_junctions.append(0)
            
        data.update({
            'gt_density_maps': torch.stack(density_maps),
            'gt_waypoints': torch.stack(waypoints).permute(0, 2, 1), # (B, L, 2)
            'gt_waypoints_masks': torch.stack(waypoints_masks),
            'gt_affected_by_lights': torch.tensor(affected_by_lights),
            'gt_affected_by_signs': torch.tensor(affected_by_signs),
            'gt_affected_by_junctions': torch.tensor(affected_by_junctions),
            'goal_points': torch.stack(goal_points)
        })
        
        return data