from mmengine.config import Config
from mmengine.registry import init_default_scope
from fsd.registry import MODELS, RUNNERS
from fsd.runner import Runner
from fsd.datasets.data_utils.vector_map import VectorizedLocalMap
from pyquaternion import Quaternion
import numpy as np 

init_default_scope('fsd')

cfg = Config.fromfile('tests/agents/VAD/config1.py')


def test_dataloader():
    # dataloader 
    dataloader = Runner.build_dataloader(cfg.train_dataloader)
    
    # map
    patch_h = cfg.point_cloud_range[4] - cfg.point_cloud_range[1]
    patch_w = cfg.point_cloud_range[3] - cfg.point_cloud_range[0]
    
    vector_map = VectorizedLocalMap(
        dataroot='./data/nuscenes',
        patch_size=(patch_h, patch_w),
        map_classes=['divider','ped_crossing','boundary'],
        line_classes=['road_divider', 'lane_divider'],
        ped_crossing_classes=['ped_crossing'],
        contour_classes=['road_segment', 'lane'],
        sample_dist=1,
        num_samples=250,
        padding=False,
        fixed_ptsnum_per_line=-1,
    )
    
    
    for i, data in enumerate(dataloader):
        data_samples = data['data_samples']
        for ds in data_samples:
            map_loc = 'singapore-onenorth'
            ego2global = np.array(ds.ego2global)
            translation = ego2global[:3, 3]
            rotation = list(Quaternion(matrix=ego2global[:3, :3], atol=1e-06).q)
            vec = vector_map.gen_vectorized_samples(map_loc, translation, rotation)
            
            print(vec)

#test_model_registry()
#test_dataloader_registry()
test_dataloader()