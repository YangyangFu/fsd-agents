from mmengine.registry import init_default_scope
from mmdet3d.registry import DATASETS
from mmengine.runner import Runner
from mmdet3d.structures import Det3DDataSample, PointData

# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset.sampler import DefaultSampler
from mmengine.visualization.vis_backend import LocalVisBackend

from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.datasets.transforms.loading import (LoadAnnotations3D,
                                                 LoadPointsFromFile,
                                                 LoadPointsFromMultiSweeps)
from mmdet3d.datasets.transforms.test_time_aug import MultiScaleFlipAug3D
from mmdet3d.datasets.transforms.transforms_3d import (  # noqa
    GlobalRotScaleTrans, ObjectNameFilter, ObjectRangeFilter, PointShuffle,
    PointsRangeFilter, RandomFlip3D)
from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric
from mmdet3d.visualization.local_visualizer import Det3DLocalVisualizer

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-50, -50, -5, 50, 50, 3]
# Using calibration info convert the Lidar-coordinate point cloud range to the
# ego-coordinate point cloud range could bring a little promotion in nuScenes.
# point_cloud_range = [-50, -50.8, -5, 50, 49.2, 3]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
metainfo = dict(classes=class_names)
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(use_lidar=True, use_camera=True)
data_prefix = dict(pts='samples/LIDAR_TOP', 
                CAM_FRONT='samples/CAM_FRONT',
                CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
                CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
                CAM_BACK='samples/CAM_BACK',
                CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
                CAM_BACK_LEFT='samples/CAM_BACK_LEFT', 
                sweeps='sweeps/LIDAR_TOP')

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection3d/nuscenes/'

# Method 2: Use backend_args, file_client_args in versions before 1.1.0
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection3d/',
#          'data/': 's3://openmmlab/datasets/detection3d/'
#      }))
backend_args = None

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles',
         to_float32=True,
         num_views=6, ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    # Actually, 'GridMask' is not used here
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats'
        ])
]

dataloader = dict(
    batch_size = 1,
    num_workers = 1,
    dataset=dict(
        type=NuScenesDataset,
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=data_prefix,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        backend_args=backend_args),
        sampler=dict(type="DefaultSampler", _scope_="mmengine", shuffle=False),
)



init_default_scope('mmdet3d')
ds = Runner.build_dataloader(dataloader)

sample = next(iter(ds))

# sample
# sample['inputs']
# sample['data_samples']

print(sample['inputs']['points'].shape)
print(sample['inputs']['img'].shape)

# ['gt_instances', 'gt_instances_3d', 'gt_pts_seg', 'eval_ann_info']
print(isinstance(sample['data_samples'], Det3DDataSample))

# ['ori_cam2img', 'num_pts_feats', 'sample_idx', 'lidar2cam', 'cam2img', 'lidar_path', 'box_type_3d', 'img_path']
print(sample['data_samples'].metainfo.keys())

# gt_instances: empty
instance = sample['data_samples'].gt_instances
print(instance.metainfo, instance.keys())

# gt_instances_3d: 
instance = sample['data_samples'].gt_instances_3d
# ['labels_3d', 'bboxes_3d']
print(instance.keys())
print(instance.labels_3d.shape, instance.bboxes_3d.shape)
print(type(instance.labels_3d), type(instance.bboxes_3d))

# gt_pts_seg: empty
point = sample['data_samples'].gt_pts_seg
print(isinstance(point, PointData))

# eval_ann_info: empty
print(sample['data_samples'].eval_ann_info == None)