from fsd.registry import DATASETS 
from mmengine.registry import init_default_scope

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-20.2, -20.2, -5.0, 20.2, 20.2, 3.0]
voxel_size = [0.2, 0.2, 8]
patch_size = [102.4, 102.4]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
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
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
_feed_dim_ = _ffn_dim_
_dim_half_ = _pos_dim_
canvas_size = (bev_h_, bev_w_)

# NOTE: You can change queue_length from 5 to 3 to save GPU memory, but at risk of performance drop.
queue_length = 5  # each sequence contains `queue_length` frames.

### traj prediction args ###
predict_steps = 12
predict_modes = 6
fut_steps = 4
past_steps = 4
use_nonlinear_optimizer = True

## occflow setting	
occ_n_future = 4	
occ_n_future_plan = 6
occ_n_future_max = max([occ_n_future, occ_n_future_plan])	

### planning ###
planning_steps = 6
use_col_optim = True


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
    dict(type="LoadMultiViewImageFromFilesInCeph", to_float32=True, file_client_args=file_client_args, img_root=data_root),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=3, use_dim=[0, 1, 2], file_client_args=file_client_args),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="LoadAnnotations3D_E2E",
        with_bbox_3d=True,
        with_label_3d=True,
        with_instances_future_traj=True,
        with_attr_label=False,
        with_vis_token=False,
        with_future_anns=False,  # occ_flow gt
        with_ins_inds_3d=True,  # ins_inds 
        ins_inds_add_1=True,    # ins_inds start from 1
    ),

    # dict(type='GenerateOccFlowLabels', grid_conf=occflow_grid_conf, ignore_index=255, only_vehicle=True, 
    #                                 filter_invisible=False),  # NOTE: Currently vis_token is not in pkl 

    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="Collect3D",
        keys=[
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_instances_ids",
            "gt_ego_future_traj",
            "gt_instances_future_traj",
            "img",
            "pts",
            #"frame_index",
            #"gt_fut_traj",
            #"gt_fut_traj_mask",
            #"gt_past_traj",
            #"gt_past_traj_mask",
            #"gt_sdc_bbox",
            #"gt_sdc_label",
            #"gt_sdc_fut_traj",
            #"gt_sdc_fut_traj_mask",
            #"gt_lane_labels",
            #"gt_lane_bboxes",
            #"gt_lane_masks",
             # Occ gt
            # "gt_segmentation",
            # "gt_instance", 
            # "gt_centerness", 
            # "gt_offset", 
            # "gt_flow",
            # "gt_backward_flow",
            # "gt_occ_has_invalid_frame",
            # "gt_occ_img_is_valid",
            # # gt future bbox for plan	
            # "gt_future_boxes",	
            # "gt_future_labels",	
            # # planning	
            # "sdc_planning",	
            # "sdc_planning_mask",	
            # "command",
        ],
    ),
]


dataset = dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        modality=input_modality,
        box_type_3d="LiDAR",
        test_mode=False,
        classes=class_names,
)

init_default_scope('fsd')
dataset = DATASETS.build(dataset)

# test dataset
sample = dataset.__getitem__(45)
print(sample.keys())