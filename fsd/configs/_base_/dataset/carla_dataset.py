default_scope='fsd'
#custom_imports = dict(imports = ['fsd.agents', 'fsd.datasets'], allow_failed_imports = False)

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
patch_size = [102.4, 102.4]
class_names = [
'car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian','others'
]

input_modality = dict(
    use_lidar=True, 
    use_camera=True, 
    use_radar=False, 
    use_map=False, 
    use_external=True
)

camera_sensors = [
    'CAM_FRONT_LEFT', 
    'CAM_FRONT', 
    'CAM_FRONT_RIGHT', 
    'CAM_BACK_LEFT',
    'CAM_BACK', 
    'CAM_BACK_RIGHT'
]
lidar_sensors = ['LIDAR_TOP']

### dataset config
sample_nterval = 5 # sample interval # frames skiped per step
predict_steps = 12
past_steps = 4
planning_steps = 6

dataset_type = "CarlaDataset"
data_root = "data-mini/carla/" # v1
info_root = "data-mini/carla/infos"
map_root = "data-mini/carla/maps"
map_file = "data-mini/carla/infos/b2d_map_infos.pkl"
file_client_args = dict(backend="disk")
ann_file_train=info_root + f"/b2d_infos_train.pkl"
ann_file_val=info_root + f"/b2d_infos_val.pkl"
ann_file_test=info_root + f"/b2d_infos_val.pkl"

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", 
         channel_order = 'rgb', 
         color_type = 'color',
         to_float32=False
    ),
    dict(type="LoadPointsFromFileCarlaDataset", coord_type="LIDAR", load_dim=3, use_dim=[0, 1, 2]),
    dict(
        type="LoadAnnotations3DPlanning",
        with_bbox_3d=True,
        with_label_3d=True,
        with_name_3d=True, # class names
        with_instances_ids=True,  # instance ids 
        with_instances_traj=True, # future
        with_ego_status=True, # ego status
        with_grids=True, # density map
    ),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(
        type="Collect3D",
        keys=[],
    ),
    dict(type="DefaultFormatBundle3D")
]

train_dataset=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=ann_file_train,
    pipeline=train_pipeline,
    classes=class_names,
    modality=input_modality,
    camera_sensors=camera_sensors,
    lidar_sensors=lidar_sensors,
    box_type_3d_original='Depth',
    box_type_3d="LiDAR",
    filter_empty_gt = True,
    past_steps = past_steps, # past trajectory length
    prediction_steps = predict_steps, # motion prediction length if any
    planning_steps = planning_steps, # planning length
    sample_interval = sample_nterval, # sample interval # frames skiped per step
    test_mode = False
)

train_dataloader = dict(
    batch_size = 2,
    num_workers = 1,
    dataset = train_dataset,
    sampler=dict(type="DefaultSampler", shuffle=False),
)

# validataion
val_pipeline = train_pipeline

val_pipeline = dict()
val_dataloader = dict()
test_pipeline = val_pipeline
test_dataloader = val_dataloader

