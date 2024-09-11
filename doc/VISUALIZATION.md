# Visualization

```python

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

# dataset config
ds_cfg = Config.fromfile('fsd/configs/_base_/dataset/carla_dataset.py')
vis_cfg = Config(dict(
    type='PlanningVisualizer',
    _scope_ = 'fsd',
    save_dir='temp_dir',
    vis_backends=[dict(type='LocalVisBackend')],
    name='vis')
)
# build dataset
init_default_scope('fsd')
ds = DATASETS.build(ds_cfg.train_dataset)
vis = VISUALIZERS.build(vis_cfg) 

```

## Draw 3d boxes on point cloud

```Python
# visualize sample
for i, item in enumerate(ds):
    data_inputs = item['inputs']
    data_samples = item['data_samples']

    # inputs and annotations
    imgs = data_inputs['img']
    imgs = [img.numpy().transpose(1, 2, 0) for img in imgs]
    pts = data_inputs['pts'].tensor.numpy()
    bboxes_3d = data_samples.instances.gt_bboxes_3d 

    # draw 3d boxes on point cloud
    vis.set_points(pts, pcd_mode=0) # 0: lidar, 1: cam, 2: depth
    bboxes_colors = [(0, 255, 0) for _ in range(len(bboxes_3d))] 
    vis.draw_bboxes_3d(bboxes_3d, bboxes_colors)
    vis.show()

```

## Draw ego future trajectory on camera images

```Python
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

# dataset config
ds_cfg = Config.fromfile('fsd/configs/_base_/dataset/carla_dataset.py')
vis_cfg = Config(dict(
    type='PlanningVisualizer',
    _scope_ = 'fsd',
    save_dir='temp_dir',
    vis_backends=[dict(type='LocalVisBackend')],
    name='vis')
)
# build dataset
init_default_scope('fsd')
ds = DATASETS.build(ds_cfg.train_dataset)
vis = VISUALIZERS.build(vis_cfg) 

# visualize sample
for i, item in enumerate(ds):
    data_inputs = item['inputs']
    data_samples = item['data_samples']

    # inputs and annotations
    imgs = data_inputs['img']
    imgs = [img.numpy().transpose(1, 2, 0) for img in imgs]
    pts = data_inputs['pts'].tensor.numpy()
    bboxes_3d = data_samples.instances.gt_bboxes_3d 

    # draw 3d boxes and ego annotated trajectory on front camera
    front_img = imgs[1]
    vis.set_image(front_img)
    lidar2world = data_inputs['pts_metas']['lidar2world']
    cam_front2world = data_inputs['img_metas']['cam2world'][1]
    cam_front_intrinsic = data_inputs['img_metas']['cam_intrinsics'][1]
    cam_front_intrinsic = np.pad(cam_front_intrinsic, (0, 1), constant_values=0)
    lidar2img = cam_front_intrinsic @ np.linalg.inv(cam_front2world) @ lidar2world
    # - 3d boxes
    vis.draw_proj_bboxes_3d(bboxes_3d, 
                            input_meta = {'lidar2img': lidar2img},
                            edge_colors='orange',
                            face_colors=None,)
    # - ego trajectory
    gt_ego_traj = data_samples.ego.gt_traj
    gt_ego_traj_xyr = gt_ego_traj.data.numpy()
    gt_ego_traj_mask = gt_ego_traj.mask.numpy()
    vis.draw_ego_trajectory_image(gt_ego_traj_xyr, gt_ego_traj_mask, input_meta = {'lidar2img': lidar2img})
    
    # front image with boxes and ego trajectory annotated
    front_img_annotated = vis.get_image()

    # save
    cv2.imwrite('front_img.png', cv2.cvtColor(front_img_traj, cv2.COLOR_RGB2BGR))

    # show
    vis.show()
```

## Draw multi-view with ego trajectory


## Draw trajectories on BEV

