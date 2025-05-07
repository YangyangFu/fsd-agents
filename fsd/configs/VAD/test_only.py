_base_ = [
    '../_base_/default_runtime.py'
]
#
plugin = False
plugin_dir = 'fsd'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# (x1, y1, z1, x2, y2, z2) in LiDAR coordinate system
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
voxel_size = [0.15, 0.15, 4]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

cameras = [
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
    'CAM_FRONT', 
    'CAM_FRONT_LEFT', 
    'CAM_FRONT_RIGHT'
]

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 4 # each sequence contains `queue_length` frames.
total_epochs = 60

model = dict(
    type='VAD',
    use_grid_mask=True,
    video_test_mode=True,
    img_backbone=dict(
        type='ResNet',
        _scope_='mmdet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='FPN',
        _scope_='mmdet',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='VADHead',
        map_thresh=0.5,
        dis_thresh=0.2,
        pe_normalization=True,
        tot_epoch=total_epochs,
        use_traj_lr_warmup=False,
        query_thresh=0.0,
        query_use_fix_pad=False,
        ego_his_encoder=None,
        ego_lcf_feat_idx=None,
        valid_fut_ts=6,
        ego_agent_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        ego_map_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        motion_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        motion_map_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        use_pe=True,
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=300,
        num_classes=num_classes,
        embed_dims=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        map_num_vec=map_num_vec,
        map_num_classes=map_num_classes,
        map_num_pts_per_vec=map_fixed_ptsnum_per_pred_line,
        map_num_pts_per_gt_vec=map_fixed_ptsnum_per_gt_line,
        map_query_embed_type='instance_pts',
        map_transform_method='minmax',
        map_gt_shift_pts_pattern='v2',
        map_dir_interval=1,
        map_code_size=2,
        map_code_weights=[1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='VADPerceptionTransformer',
            map_num_vec=map_num_vec,
            map_num_pts_per_vec=map_fixed_ptsnum_per_pred_line,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MultiScaleDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    #type='mmdet.models.DetrTransformerDecoderLayer',
                    type='BaseTransformerLayer',
                    _scope_='mmdet',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='BEVMultiScaleDeformableAttention',
                            _scope_='fsd',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=_ffn_dim_,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU',
                                     inplace=True)),
                    norm_cfg=dict(type='LN'),
                    batch_first=False,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            map_decoder=dict(
                type='MapDetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    #type='mmdet.models.DetrTransformerDecoderLayer',
                    type='BaseTransformerLayer',
                    _scope_='mmdet',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='BEVMultiScaleDeformableAttention',
                            _scope_='fsd',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=_ffn_dim_,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU',
                                     inplace=True)),
                    norm_cfg=dict(type='LN'),
                    batch_first=False,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='CustomNMSFreeCoder',
            post_center_range=[-20, -35, -10.0, 20, 35, 10.0],
            pc_range=point_cloud_range,
            max_num=100,
            voxel_size=voxel_size,
            num_classes=num_classes),
        map_bbox_coder=dict(
            type='MapNMSFreeCoder',
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=point_cloud_range,
            max_num=50,
            voxel_size=voxel_size,
            num_classes=map_num_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            _scope_='mmdet',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            _scope_='mmdet',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', _scope_='mmdet', loss_weight=0.25),
        loss_traj=dict(type='L1Loss', _scope_='mmdet',loss_weight=0.2),
        loss_traj_cls=dict(
            type='FocalLoss',
            _scope_='mmdet',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.2),
        loss_iou=dict(type='GIoULoss', _scope_='mmdet', loss_weight=0.0),
        loss_map_cls=dict(
            type='FocalLoss',
            _scope_='mmdet',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_map_bbox=dict(type='L1Loss', _scope_='mmdet', loss_weight=0.0),
        loss_map_iou=dict(type='GIoULoss', _scope_='mmdet', loss_weight=0.0),
        loss_map_pts=dict(type='PtsL1Loss', loss_weight=1.0),
        loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
        loss_plan_reg=dict(type='L1Loss', _scope_='mmdet', loss_weight=1.0),
        loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=1.0, dis_thresh=1.0),
        loss_plan_col=dict(type='PlanCollisionLoss', loss_weight=1.0),
        loss_plan_dir=dict(type='PlanMapDirectionLoss', loss_weight=0.5),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', _scope_='mmdet', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', _scope_='mmdet', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range),
            map_assigner=dict(
                type='MapHungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', _scope_='mmdet', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
                pts_cost=dict(type='OrderedPtsL1Cost', weight=1.0),
                pc_range=point_cloud_range))
        ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4)
        )
    )


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
    dict(type='LoadMultiViewImageFromFiles', _scope_='mmdet3d', to_float32=True, num_views=len(cameras)),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='LoadAnnotationsPlan3D', 
        with_bbox_3d=True, 
        with_label_3d=True, 
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
    dict(type='LoadMultiViewImageFromFiles', _scope_='mmdet3d', to_float32=True, num_views=len(cameras)),
#    dict(type='LoadPointsFromFile',
#         _scope_='mmdet3d',
#         coord_type='LIDAR',
#         load_dim=5,
#         use_dim=5),
    dict(type='LoadAnnotationsPlan3D', 
        with_bbox_3d=True, 
        with_label_3d=True, 
        with_instances_traj=True,
        with_instances_ids=True,
        with_vector_map=True),
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
        box_type_3d_original='LiDAR', # original box in nuscenes are acatually Depth box in mmdet3d. 
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

#TODO: this is faked evaluator, need to be changed
val_evaluator = dict(
    type="NuScenesMetric",
    _scope_="mmdet3d",
    data_root=data_root,
    ann_file=data_root + '/nuscenes_infos_val.pkl',\
    modality=input_modality,
    metric='bbox',
    jsonfile_prefix='eval')

test_evaluator = val_evaluator

# seed
randomness = dict(seed=2024)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01)

# parameter-lever learning rate and weight decay settings
paramwise_cfg=dict(
    custom_keys={
        'img_backbone': dict(lr_mult=0.1),
    })

# optimizer wrapper 
optim_wrapper = dict(
    type="OptimWrapper",
    _scope_="mmdet",
    optimizer=optimizer,
    paramwise_cfg=paramwise_cfg,
    clip_grad=dict(
        max_norm=35, 
        norm_type=2
    )
)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

# eval
evaluation = dict(interval=1, pipeline=test_pipeline)

# default hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        save_begin=0,
        interval=1, 
        by_epoch=True,
        save_best='auto',
        max_keep_ckpts=3,
    ),
)

# training log
vis_backends = [
    dict(type='TensorboardVisBackend'),
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='Visualizer',
    vis_backends=vis_backends,
    name='visualizer',
)

load_from = './ckpts/vad_base.pth'
resume = False
