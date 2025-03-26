_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
work_dir='.'
#
plugin = False
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]


img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

version = 'v1.0-mini'#'v1.0-trainval'
metainfo = dict(
    classes=class_names,
    version=version)

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

model = dict(
    type='BEVFormer',
    use_grid_mask=True,
    video_test_mode=True,
    img_backbone=dict(
        type='ResNet',
        _scope_="mmdet",
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        _scope_="mmdet",
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEVFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer',
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
                    type='BaseTransformerLayer',
                    _scope_='mmdet',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            _scope_='mmdet',
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
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            num_classes=10),
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
        loss_bbox=dict(type='L1Loss',  _scope_='mmdet', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', _scope_='mmdet', loss_weight=0.0),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost3D',weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', _scope_='mmdet', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range)),
        ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,)))

dataset_type = 'NuScenesDatasetBEVFormer'
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
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', _scope_='mmdet3d', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg, divider=1.0),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='Pack3DDetInputs', _scope_='mmdet3d', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg, divider=1.0),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PadMultiViewImage', size_divisor=32),
        ]),
    dict(type='Pack3DDetInputs', _scope_='mmdet3d', keys=['img'])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='nuscenes_infos_train.pkl',
        metainfo=metainfo,
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    sampler=dict(type="DefaultSampler", _scope_="mmengine", shuffle=True),
    pin_memory=True,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(    
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='nuscenes_infos_val.pkl',
        metainfo=metainfo,
        pipeline=test_pipeline,  
        bev_size=(bev_h_, bev_w_),
        modality=input_modality,
        test_mode=True,),
    sampler=dict(type="DefaultSampler", _scope_="mmengine", shuffle=False),
    pin_memory=True,
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='nuscenes_infos_val.pkl',
        metainfo=metainfo,
        pipeline=test_pipeline, 
        bev_size=(bev_h_, bev_w_),
        modality=input_modality,
        test_mode=True,
        serialize_data=False),
    sampler=dict(type="DefaultSampler", _scope_="mmengine", shuffle=False),
    pin_memory=True,
)

val_evaluator = dict(
    type="NuScenesMetric",
    _scope_="mmdet3d",
    data_root=data_root,
    ann_file=data_root + '/nuscenes_infos_val.pkl',\
    modality=input_modality,
    metric='bbox',
    jsonfile_prefix='eval')
test_evaluator = val_evaluator


total_epochs = 24
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=total_epochs, 
    val_interval=1
)
val_cfg = dict()
test_cfg = dict()

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

optim_wrapper = dict(
    type="OptimWrapper",
    _scope_="mmdet",
    optimizer=optimizer,
    paramwise_cfg=paramwise_cfg,
    clip_grad=dict(
        max_norm=35, norm_type=2
    )
)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

evaluation = dict(interval=1, pipeline=test_pipeline)

load_from = 'ckpts/bevformer_r101_dcn_24ep.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)