import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch 

from mmengine.evaluator import BaseMetric
from fsd.models.losses import MaskedL1Loss
from fsd.registry import METRICS

@METRICS.register_module()
class TrajectoryMetric(BaseMetric):
    """Metric for single modal trajecotry prediction.
    """    
    
    # NOTE: the datasample is processed as dict before sending to the metric in the val loop
    def process(self, 
                data_batch: dict,
                data_samples: Sequence[dict]) -> None:
        
        for data_sample in data_samples:
            result = dict()
            if 'traj' in data_sample['pred_ego']:
                planning_steps = data_sample['gt_ego']['traj']['num_future_steps']
                pred_traj_ego = data_sample['pred_ego']['traj']['data'].to('cpu')[-planning_steps:, :2]
                gt_traj_ego = data_sample['gt_ego']['traj']['data'].to('cpu')[-planning_steps:, :2]
                gt_traj_mask = data_sample['gt_ego']['traj']['mask'].to('cpu')[-planning_steps:].view(-1, 1).repeat(1, 2) if 'mask' in data_sample['gt_ego']['traj'] else None # [T, 2]
                result['pred_traj_ego'] = pred_traj_ego
                result['gt_traj_ego'] = gt_traj_ego
                if gt_traj_mask is not None:
                    result['gt_traj_ego_mask'] = gt_traj_mask
            
            # TODO: the eval loop seems didn't convert list of TrajectoryData to dict
            if 'traj' in data_sample['pred_instances']:
                planning_steps = data_sample['gt_instances']['traj'][0].num_future_steps
                pred_traj_instances = torch.cat([traj.data.to('cpu')[-planning_steps:, :2] for traj in data_sample['pred_instances']['traj']], axis=0)
                gt_traj_instances = torch.cat([traj.data.to('cpu')[-planning_steps:, :2] for traj in data_sample['gt_instances']['traj']], axis=0)
                if data_sample['gt_instances']['traj'][0].mask is not None:
                    gt_traj_mask = torch.cat([traj.mask.to('cpu')[-planning_steps:].view(-1, 1).repeat(1, 2) for traj in data_sample['gt_instances']['traj']], axis=0)
                else:
                    gt_traj_mask = None
                    
                result['pred_traj_instances'] = pred_traj_instances
                result['gt_traj_instances'] = gt_traj_instances
                if gt_traj_mask is not None:
                    result['gt_traj_instances_mask'] = gt_traj_mask 
                
            self.results.append(result)
        
    def compute_metrics(self, results: List[dict]) -> dict:
        """Compute the metrics.
        """
        metrics = dict()
        if results and 'pred_traj_ego' in results[0]:
            pred_traj_ego = torch.cat([res['pred_traj_ego'] for res in results], dim=0)
            gt_traj_ego = torch.cat([res['gt_traj_ego'] for res in results], dim=0)
            if 'gt_traj_ego_mask' in results[0]:
                gt_traj_ego_mask = torch.cat([res['gt_traj_ego_mask'] for res in results], dim=0)
            ego_metrics = self._compute_traj_metrics(pred_traj_ego, gt_traj_ego, gt_traj_ego_mask)
            metrics['ego_traj'] = ego_metrics
            
        if results and 'pred_traj_instances' in results[0]:
            pred_traj_instances = torch.cat([res['pred_traj_instances'] for res in results], dim=0)
            gt_traj_instances = torch.cat([res['gt_traj_instances'] for res in results], dim=0)
            if 'gt_traj_instances_mask' in results[0]:
                gt_traj_instances_mask = torch.cat([res['gt_traj_instances_mask'] for res in results], dim=0)
            instances_metrics = self._compute_traj_metrics(pred_traj_instances, gt_traj_instances, gt_traj_instances_mask)
            metrics['instances_traj'] = instances_metrics
            
        return metrics
    
    def _compute_traj_metrics(self, 
                              pred_traj: torch.Tensor, 
                              gt_traj: torch.Tensor,
                              gt_traj_mask: Optional[torch.Tensor]=None) -> Dict[str, Union[float, torch.Tensor]]:
        """Compute the trajectory metrics.
        """
        l1 = MaskedL1Loss()
        loss = l1(pred_traj, gt_traj, mask=gt_traj_mask)
        return loss