import sys
sys.path.append('')

import os 
import cv2
import numpy as np
import torch 

from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmdet3d.structures import LiDARInstance3DBoxes, limit_period, Box3DMode
from fsd.registry import DATASETS, VISUALIZERS

ds_cfg = Config.fromfile('fsd/configs/_base_/dataset/carla_dataset.py')
vis_cfg = Config(dict(
    type='PlanningVisualizer',
    _scope_ = 'fsd',
    save_dir='temp_dir',
    vis_backends=[dict(type='LocalVisBackend')],
    name='vis')
)
init_default_scope('fsd')
ds = DATASETS.build(ds_cfg.train_dataset)
vis = VISUALIZERS.build(vis_cfg) 

for i, item in enumerate(ds):
    data_inputs = item['inputs']
    data_samples = item['data_samples']
    
    # need set lidar2img in data_samples
    lidar2world = data_samples.pts_metas['lidar2world']
    
    cams2world = data_samples.img_metas['cam2world']
    cams_intrinsics = data_samples.img_metas['cam_intrinsics']
    cams_intrinsics = [np.pad(cam, (0, 1), constant_values=0) for cam in cams_intrinsics]
    lidar2imgs = [cam_intrinsic @ np.linalg.inv(cam2world) @ lidar2world for cam_intrinsic, cam2world in zip(cams_intrinsics, cams2world)]
    
    data_samples.set_metainfo(dict(lidar2img = lidar2imgs))
    
    # save file with four digits
    out_file = os.path.join('tmp', f'{i:05d}')
    
    # add data and plot
    vis.add_datasample(name='test',
                       data_input=data_inputs,
                       data_sample=data_samples,
                       draw_gt=True,
                       draw_pred=False,
                       show=True,
                       vis_task='multi-modality_planning',
                       traj_img_idx = 1,
                       out_file = out_file,
    )
