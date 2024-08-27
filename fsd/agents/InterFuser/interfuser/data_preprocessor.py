from typing import Union, List

import torch
import numpy as np
from fsd.models import PlanningDataPreprocessor, stack_batch
from fsd.registry import MODELS 

@MODELS.register_module()
class InterFuserDataPreprocessor(PlanningDataPreprocessor):
    def __init__(self, 
                 pixels_per_meter=8, 
                 bev_range=[0, 28, -14, 14],
                 max_hist_per_pixel=5,
                 below_threshold=-2.0,
                 *args, 
                 **kwargs):
        """_summary_

        """
        super().__init__(*args, **kwargs)
        self.pixels_per_meter = pixels_per_meter
        self.bev_range = bev_range
        self.max_hist_per_pixel = max_hist_per_pixel
        self.below_threshold = below_threshold
    
    def forward(self,
                data: Union[dict, List[dict]],
                training: bool = False) -> Union[dict, List[dict]]:
        """Perform data processing before sending to model:
            - cast data to the target device.
            - prepare model specific input data, such as ego goal points, ego velocity, etc
            - processing image data, such as generating images, etc
            - processing point data, such as generating voxels, 
            - stacking batch data etc.

        Args:
            data (dict or List[dict]): Data from dataloader. The dict contains
                the whole batch data, when it is a list[dict], the list
                indicates test time augmentation.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.
        """
        # generate goal points for goal-directed tasks
        data = self.get_goal_points(data)
        data = self.get_ego_velocity(data)
        
        # generate bin histogram from point cloud
        data = self.generate_pts_to_hist(data)
        
        return super().forward(data, training)

    def get_goal_points(self, data: dict):
        """Generate goal points for goal-directed tasks
        """
        
        gt_ego_future_traj = [sample.gt_ego.gt_ego_future_traj.xy[..., -1].squeeze(0) for sample in data['data_samples']]
        
        data['inputs']['goal_points'] = torch.stack(gt_ego_future_traj)
        
        return data
    
    def get_ego_velocity(self, data: dict):
        """Generate ego velocity for motion prediction tasks
        
        Returns:
            dict with ego velocity (B, 1)
        """
        
        v = [sample.gt_ego.ego_velocity for sample in data['data_samples']]
        v = [torch.sqrt(vx**2 + vy**2).unsqueeze(0).to(torch.float32) for vx, vy, _ in v]
        
        data['inputs']['ego_velocity'] = torch.stack(v)
        
        return data

    def generate_pts_to_hist(self, data: dict, view_ego_coord: bool = True):
        """Generate bin histogram from point cloud
        """
        # batched data
        points = data['inputs']['pts']

        points_ = []
        for pts in points:
            below_mask = pts.tensor[:, 2] <= self.below_threshold
            above_mask = ~below_mask
            below = pts[below_mask]
            above = pts[above_mask]
            below_feat = self._2bin_histogram(below.numpy())
            above_feat = self._2bin_histogram(above.numpy())
            total_feat = below_feat + above_feat
            features = np.stack([below_feat, above_feat, total_feat], axis=0).astype(np.float32) # [C, H, W]
            
            # min-max normalization
            features = features / features.max() 
        
            # flip for visualization as image: x front -> -y in image
            if view_ego_coord:
                features = np.rot90(features, k=1, axes=(1,2))
            points_.append(torch.from_numpy(features.copy()))

        data['inputs']['pts'] = points_
        return data

    def _2bin_histogram(self, points):
        xbins = np.linspace(self.bev_range[0], self.bev_range[1]+1, (self.bev_range[1]-self.bev_range[0])*self.pixels_per_meter+1)
        ybins = np.linspace(self.bev_range[2], self.bev_range[3]+1, (self.bev_range[3]-self.bev_range[2])*self.pixels_per_meter+1)
        hist = np.histogramdd(points[:, :2], bins=(ybins, xbins))[0] # x in lidar -> y in image
        hist[hist > self.max_hist_per_pixel] = self.max_hist_per_pixel
        hist = hist / self.max_hist_per_pixel
        return hist

    def stack_batch_data(self, data: dict) -> dict:
        """
        Stack batched data for model input.
        
        Interfuser uses multi-view images with different sizes as model input. 
         Here no stack is performed on multi-view dimension.

        Args:
            data (dict): Data sampled from dataloader.

        Returns:
            dict: Data in the same format as the model input.
        """
        # [batched view1, batched view2, ...]
        if 'img' in data['inputs']:
            data['inputs']['img'] = [stack_batch(img) for img in data['inputs']['img']]
            
        if 'pts' in data['inputs']:
            data['inputs']['pts'] = stack_batch(data['inputs']['pts'])
            
        return data    
        