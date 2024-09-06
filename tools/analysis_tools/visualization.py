import sys
sys.path.append('')

import os 
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
    name='vis')
)
init_default_scope('fsd')
ds = DATASETS.build(ds_cfg.train_dataset)
vis = VISUALIZERS.build(vis_cfg) 

for i, item in enumerate(ds):
    data_inputs = item['inputs']
    data_samples = item['data_samples']
    
    gt_instances = data_samples.gt_instances_3d
    bboxes_3d = gt_instances.gt_bboxes_3d 

    
    # display the data
    front_img = data_inputs['img'][0]
    pts = data_inputs['pts'].tensor.numpy()
    
    # draw 3d bboxes on point cloud 
    # pts data are in mmdet Lidar coord,
    #vis.set_points(pts, pcd_mode=0) # 0: lidar, 1: cam, 2: depth
    #bboxes_colors = [(0, 255, 0) for _ in range(len(bboxes_3d))] 
    #vis.draw_bboxes_3d(bboxes_3d, bboxes_colors)
    #vis.show()
    
    # draw bev bboxes
    #vis.set_bev_image()
    #vis.draw_bev_bboxes(bboxes_3d.convert_to(Box3DMode.DEPTH), edge_colors='orange')
    #vis.show()
    
    # draw 3d bboxes on image
    img = front_img.numpy().transpose(1, 2, 0)
    #rgb to bgr
    img = img[..., ::-1]
     
    vis.set_image(img)
    lidar2world = data_inputs['pts_metas']['lidar2world']
    cam_front2world = data_inputs['img_metas']['cam2world'][0]
    cam_front_intrinsic = data_inputs['img_metas']['cam_intrinsics'][0]
    cam_front_intrinsic = np.pad(cam_front_intrinsic, (0, 1), constant_values=0)
    lidar2img = cam_front_intrinsic @ np.linalg.inv(cam_front2world) @ lidar2world
    
    vis.draw_proj_bboxes_3d(bboxes_3d, 
                            input_meta = {'lidar2img': lidar2img},
                            edge_colors='orange',
                            face_colors=None,)
    gt_ego_traj = data_samples.gt_ego.gt_ego_future_traj
    gt_ego_traj_xy = gt_ego_traj.xy.numpy().squeeze(0).transpose(1, 0)
    gt_ego_traj_mask = gt_ego_traj.mask.numpy().squeeze(0)
    print(f"gt_ego_traj: {gt_ego_traj_xy}")
    vis.draw_trajectory_image(gt_ego_traj_xy, gt_ego_traj_mask, input_meta = {'lidar2img': lidar2img})
    
    vis.show()
    
    
    
