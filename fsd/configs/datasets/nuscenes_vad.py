# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# (x1, y1, z1, x2, y2, z2) in LiDAR coordinate system
point_cloud_range = [-50.0, -50.0, -2.0, 50.0, 50.0, 2.0]
voxel_size = [0.15, 0.15, 4]

to_rgb = False
#img_norm_cfg = dict(
#    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=to_rgb)
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[1, 1, 1], to_rgb=to_rgb)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
num_classes = len(class_names)

# map has classes: divider, ped_crossing, boundary
map_classes = ['divider', 'ped_crossing', 'boundary']
map_num_vec = 100
map_fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0
map_fixed_ptsnum_per_pred_line = 20
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = len(map_classes)

# plannign settings
past_steps = 2 # past trajectory length
agent_fut_steps = 6 # motion prediction length if any
ego_fut_steps = 6 # planning length

version = 'v1.0-mini'#'v1.0-trainval'
metainfo = dict(
    classes=class_names,
    map_classes=map_classes,
    version=version)

# camera
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

view_names = [
    'CAM_FRONT_LEFT', 
    'CAM_FRONT', 
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
]

bev_h_ = 200
bev_w_ = 200
queue_length = 4 # each sequence contains `queue_length` frames.
total_epochs = 10

# data
dataset_type = 'NuScenesDatasetVAD'#NuScenesDatasetPlan3D
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')
data_prefix = dict(
    pts='samples/LIDAR_TOP', 
    img='', # for single view 
    sweeps='sweeps/LIDAR_TOP',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', 
         _scope_='mmdet3d', 
         to_float32=True, 
         num_views=len(view_names)),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='LoadAnnotationsPlan3D', 
        with_bbox_3d=True, 
        with_label_3d=True, 
        with_ego_traj=True,
        with_instances_traj=True,
        with_instances_ids=True,
        with_vector_map=True,),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg, divider=1.0),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='Pack3DPlanInputs',
         keys=['img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes_traj', 'gt_bboxes_traj_mask', 'bboxes_context', 
               'gt_ego_traj', 'gt_ego_traj_mask', 'ego_command', 'ego_context', 'ego_history_traj', 'ego_history_mask', 
               'gt_map_vectors_pt', 'gt_map_vectors_label'])
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", _scope_="mmengine", shuffle=False),
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='nuscenes_infos_train.pkl',
        metainfo=metainfo,
        pipeline=train_pipeline,
        modality=input_modality,
        box_type_3d_original='LiDAR', # original box in nuscenes are acatually Depth box in mmdet3d. 
        box_type_3d='LiDAR',
        past_steps=past_steps, # past trajectory length
        prediction_steps=agent_fut_steps, # motion prediction length if any
        planning_steps=ego_fut_steps, # planning length
        test_mode=False,
        with_can_bus=True,
        point_cloud_range=point_cloud_range,
        bev_size=(bev_h_, bev_w_),
        bev_queue_length=queue_length,
        map_fixed_ptsnum_per_line = map_fixed_ptsnum_per_gt_line,
        map_sample_dist=1.0,
        map_sample_nums=250,
        )
)

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', _scope_='mmdet3d', to_float32=True, num_views=len(view_names)),
    dict(type='LoadPointsFromFile',
         _scope_='mmdet3d',
         coord_type='DEPTH',
         load_dim=5,
         use_dim=[0 ,1, 2]),
    dict(type='LoadAnnotationsPlan3D', 
        with_bbox_3d=True, 
        with_label_3d=True, 
        with_ego_traj=True,
        with_instances_traj=True,
        with_instances_ids=True,
        with_vector_map=True),
    #dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    #dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg, divider=1.0),
    dict(type='RandomScaleImageMultiViewImage', scales=[1.0]), # no scale at all but with lidar2img
    #dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='Pack3DPlanInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes_traj', 'gt_bboxes_traj_mask', 'bboxes_context', 
            'gt_ego_traj', 'gt_ego_traj_mask', 'ego_command', 'ego_context', 'ego_history_traj', 'ego_history_mask', 
            'gt_map_vectors_pt', 'gt_map_vectors_label'])
]

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", _scope_="mmengine", shuffle=False),
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='nuscenes_infos_val.pkl',
        metainfo=metainfo,
        pipeline=test_pipeline,
        modality=input_modality,
        box_type_3d_original='Depth', # original box in nuscenes are acatually Depth box in mmdet3d. 
        box_type_3d='LiDAR',
        past_steps=past_steps, # past trajectory length
        prediction_steps=agent_fut_steps, # motion prediction length if any
        planning_steps=ego_fut_steps, # planning length
        test_mode=True,
        with_can_bus=True,
        point_cloud_range=point_cloud_range,
        bev_size=(bev_h_, bev_w_),
        bev_queue_length=queue_length,
        map_fixed_ptsnum_per_line = map_fixed_ptsnum_per_gt_line,
        map_sample_dist=1.0,
        map_sample_nums=250,
        )
)

test_dataloader = val_dataloader

train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=total_epochs, 
    val_interval=1
)

val_cfg = dict(
    type='ValLoop'
)

test_cfg = dict(
    type='TestLoop'
    )
