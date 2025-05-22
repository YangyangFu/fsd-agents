import numpy as np

from mmengine.config import Config
from mmengine.registry import init_default_scope
from fsd.runner import Runner
from fsd.registry import DATASETS, VISUALIZERS


def _convert_kitti_to_mmdet3d(data_sample):
    """Convert kitti coordinate to mmdet3d coordinate.
    
    Args:
        bboxes_3d (LiDARInstance3DBoxes): 3D boxes in kitti coordinate.

    Returns:
        LiDARInstance3DBoxes: 3D boxes in mmdet3d coordinate.
    """
    bboxes_3d = data_sample.gt_instances_3d.bbox
    bboxes_data = bboxes_3d.tensor.clone()
    bboxes_data[:, 3:6] = bboxes_data[:, [4, 3, 5]]
    bboxes_data[:, 6] = -bboxes_data[:, 6] - np.pi/2
    bboxes_3d = bboxes_3d.new_box(data=bboxes_data)
    
    data_sample.gt_instances_3d.bbox = bboxes_3d
    return data_sample

init_default_scope('fsd')
ds_cfg = Config.fromfile('fsd/configs/datasets/nuscenes_vad.py')
ds = Runner.build_dataloader(ds_cfg.test_dataloader)

vis_cfg = Config(dict(
    type='PlanningVisualizer',
    _scope_ = 'fsd',
    save_dir='./temp_dir',
    image_mode='rgb' if ds_cfg.to_rgb else 'bgr',
    vis_backends=[dict(type='LocalVisBackend')],
    name='vis')
)
vis = VISUALIZERS.build(vis_cfg) 
vis.dataset_meta = ds.dataset.metainfo

for i, item in enumerate(ds):
    data_inputs = item['inputs']
    data_samples = item['data_samples']
    data_samples = [_convert_kitti_to_mmdet3d(data_sample) for data_sample in data_samples]
    
    view_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
    for b, data_sample in enumerate(data_samples):
        data_input = {}
        data_input['img'] = data_inputs['img'][b]
        data_input['points'] = data_inputs['points'][b]
        
        data_sample = data_samples[b]
        vis.add_datasample(
            name='test',
            data_input = data_input,
            data_sample = data_sample,
            draw_gt=True,
            draw_pred=False,
            show=True,
            wait_time=0.05,
            step=b,
            vis_task='multi-modality_planning',
            show_pcd_rgb=False,
            multi_view_names=view_names,
            pcd_range=ds_cfg.point_cloud_range,
            map_format='polyline', #'fixed_num_pts',
            pixels_per_meter=10,
            to_mmdet3d_lidar=ds.dataset.to_mmdet3d_lidar,
        )

