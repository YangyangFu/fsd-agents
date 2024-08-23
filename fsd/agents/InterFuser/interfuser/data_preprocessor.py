from typing import Union, List

import torch
from fsd.models import PlanningDataPreprocessor
from fsd.registry import MODELS 

@MODELS.register_module()
class InterFuserDataPreprocessor(PlanningDataPreprocessor):
    def __init__(self, 
                 *args, 
                 **kwargs):
        """_summary_

        """
        super().__init__(*args, **kwargs)

    def forward(self,
                data: Union[dict, List[dict]],
                training: bool = False) -> Union[dict, List[dict]]:
        """Perform data processing before sending to model:
            - cast data to the target device.
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
        
        return super().forward(data, training)

    def get_goal_points(self, data: dict):
        """Generate goal points for goal-directed tasks
        """
        
        gt_ego_future_traj = [sample.gt_ego.gt_ego_future_traj.xy[..., -1].squeeze(0) for sample in data['data_samples']]
        
        data['inputs']['goal_points'] = torch.stack(gt_ego_future_traj)
        
        return data
