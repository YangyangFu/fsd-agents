from typing import Mapping, Optional, Sequence, Union, Tuple, List

from fsd.registry import AGENT_TRANSFORMS
from .utils import generate_density_map

@AGENT_TRANSFORMS.register_module()
class InterFuserDensityMap(object):
    """Generate object density map data for interfuser model

    """
    def __init__(self, 
                 bev_range: Sequence[float], 
                 pixels_per_meter: Optional[int] = 8):
        """Initialize DensityMap
        Args:
            bev_range (Sequence[float]): [xmin, xmax, ymin, ymax]
            pixels_per_meter (Optional[int], optional): Defaults to 8.
        """
        super().__init__()
        self.bev_range = bev_range
        self.pixels_per_meter = pixels_per_meter
        
    def __call__(self, results):
        """Preprocess data for interfuser model
        Args:
            results (Mapping[str, Any]): Input data.

        """
        gt_bboxes = results['anno_info']['gt_bboxes_3d']
        density_map = generate_density_map(gt_bboxes, self.bev_range, self.pixels_per_meter)
        results['anno_info']['gt_grid_density'] = density_map
        return results

        
        