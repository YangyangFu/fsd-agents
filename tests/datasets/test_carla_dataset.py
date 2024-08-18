from fsd.registry import DATASETS, RUNNERS
from mmengine.registry import init_default_scope
#from mmengine.runner import Runner 
from mmengine.visualization import Visualizer
from fsd.runner import Runner
from mmengine.registry import count_registered_modules

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
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
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromFileCarlaDataset", coord_type="LIDAR", load_dim=3, use_dim=[0, 1, 2]),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="LoadAnnotations3DPlanning",
        with_bbox_3d=True,
        with_label_3d=True,
        with_instances_ids=True,  # instance ids 
        with_attr_label=False,
        with_instances_future_traj=True, # future
    ),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="Points2BinHistogramGenerator", pixels_per_meter=10, bev_range=[-40, 40, -25, 25]),
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
        ],
    ),
    dict(type="DefaultFormatBundle3D")
]

dataloader = dict(
    batch_size = 1,
    num_workers = 1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        box_type_3d="LiDAR",
        filter_empty_gt = True,
        past_steps = 4, # past trajectory length
        prediction_steps = 6, # motion prediction length if any
        planning_steps = 6, # planning length
        sample_interval = 5, # sample interval # frames skiped per step
        test_mode = False
    ),
    sampler=dict(type="DefaultSampler", _scope_="mmengine", shuffle=False),
    #sampler=dict(type="DistributedSampler"),
    #nonshuffler_sampler=dict(type="DistributedSampler"),
)

init_default_scope('fsd')
dl = Runner.build_dataloader(dataloader)
#dl = DATASETS.build_dataloader(dataloader)

# visualize the sample
vis = Visualizer()

for sample in dl:
    vis.set_image(sample['img'][0].data[0,...].numpy().transpose(1,2,0))
    vis.show(backend='cv2')    
    vis.set_image(sample['pts'][0].data.numpy().transpose(1,2,0)*255)
    vis.show(backend='cv2')
    break

    

