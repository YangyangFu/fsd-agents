_dim_ = 256
_ffn_dim_ = 512
_num_levels_ = 1
_pos_dim_ = 128
bev_h_ = 150
bev_w_ = 150
cameras = [
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
]
checkpoint_config = dict(interval=1)
class_names = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]
data_prefix = dict(
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    img='',
    pts='samples/LIDAR_TOP',
    sweeps='sweeps/LIDAR_TOP')
data_root = 'data/nuscenes'
dataset_type = 'NuScenesDatasetBEVFormer'
default_hooks = dict(
    checkpoint=dict(by_epoch=True, interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'fsd'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
eval_pipeline = [
    dict(
        coord_type='LIDAR',
        file_client_args=dict(backend='disk'),
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(
        file_client_args=dict(backend='disk'),
        sweeps_num=10,
        type='LoadPointsFromMultiSweeps'),
    dict(
        class_names=[
            'car',
            'truck',
            'trailer',
            'bus',
            'construction_vehicle',
            'bicycle',
            'motorcycle',
            'pedestrian',
            'traffic_cone',
            'barrier',
        ],
        type='DefaultFormatBundle3D',
        with_label=False),
    dict(keys=[
        'points',
    ], type='Collect3D'),
]
evaluation = dict(
    interval=1,
    pipeline=[
        dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
        dict(
            divider=1.0,
            mean=[
                103.53,
                116.28,
                123.675,
            ],
            std=[
                1.0,
                1.0,
                1.0,
            ],
            to_rgb=False,
            type='NormalizeMultiviewImage'),
        dict(
            flip=False,
            img_scale=(
                1600,
                900,
            ),
            pts_scale_ratio=1,
            transforms=[
                dict(scales=[
                    0.8,
                ], type='RandomScaleImageMultiViewImage'),
                dict(size_divisor=32, type='PadMultiViewImage'),
            ],
            type='MultiScaleFlipAug3D'),
        dict(_scope_='mmdet3d', keys=[
            'img',
        ], type='Pack3DDetInputs'),
    ])
file_client_args = dict(backend='disk')
img_norm_cfg = dict(
    mean=[
        103.53,
        116.28,
        123.675,
    ], std=[
        1.0,
        1.0,
        1.0,
    ], to_rgb=False)
input_modality = dict(
    use_camera=True,
    use_external=True,
    use_lidar=False,
    use_map=False,
    use_radar=False)
load_from = 'ckpts/bevformer_small_epoch_24.pth'
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
    interval=50)
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr_config = dict(
    min_lr_ratio=0.001,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333)
metainfo = dict(
    classes=[
        'car',
        'truck',
        'construction_vehicle',
        'bus',
        'trailer',
        'barrier',
        'motorcycle',
        'bicycle',
        'pedestrian',
        'traffic_cone',
    ],
    version='v1.0-mini')
model = dict(
    img_backbone=dict(
        _scope_='mmdet',
        dcn=dict(deform_groups=1, fallback_on_stride=False, type='DCNv2'),
        depth=101,
        frozen_stages=1,
        norm_cfg=dict(requires_grad=False, type='BN2d'),
        norm_eval=True,
        num_stages=4,
        out_indices=(3, ),
        stage_with_dcn=(
            False,
            False,
            True,
            True,
        ),
        style='caffe',
        type='ResNet',
        with_cp=True),
    img_neck=dict(
        _scope_='mmdet',
        add_extra_convs='on_output',
        in_channels=[
            2048,
        ],
        num_outs=1,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=0,
        type='FPN'),
    pts_bbox_head=dict(
        as_two_stage=False,
        bbox_coder=dict(
            max_num=300,
            num_classes=10,
            pc_range=[
                -51.2,
                -51.2,
                -5.0,
                51.2,
                51.2,
                3.0,
            ],
            post_center_range=[
                -61.2,
                -61.2,
                -10.0,
                61.2,
                61.2,
                10.0,
            ],
            type='NMSFreeCoder'),
        bev_h=150,
        bev_w=150,
        loss_bbox=dict(_scope_='mmdet', loss_weight=0.25, type='L1Loss'),
        loss_cls=dict(
            _scope_='mmdet',
            alpha=0.25,
            gamma=2.0,
            loss_weight=2.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(_scope_='mmdet', loss_weight=0.0, type='GIoULoss'),
        num_classes=10,
        num_query=900,
        positional_encoding=dict(
            _scope_='mmdet',
            col_num_embed=150,
            num_feats=128,
            row_num_embed=150,
            type='LearnedPositionalEncoding'),
        sync_cls_avg_factor=True,
        train_cfg=dict(
            assigner=dict(
                cls_cost=dict(type='FocalLossCost3D', weight=2.0),
                iou_cost=dict(_scope_='mmdet', type='IoUCost', weight=0.0),
                pc_range=[
                    -51.2,
                    -51.2,
                    -5.0,
                    51.2,
                    51.2,
                    3.0,
                ],
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                type='HungarianAssigner3D')),
        transformer=dict(
            decoder=dict(
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    _scope_='mmdet',
                    attn_cfgs=[
                        dict(
                            _scope_='mmdet',
                            dropout=0.1,
                            embed_dims=256,
                            num_heads=8,
                            type='MultiheadAttention'),
                        dict(
                            _scope_='fsd',
                            embed_dims=256,
                            num_levels=1,
                            type='BEVMultiScaleDeformableAttention'),
                    ],
                    batch_first=False,
                    ffn_cfgs=dict(
                        act_cfg=dict(inplace=True, type='ReLU'),
                        embed_dims=256,
                        feedforward_channels=512,
                        ffn_drop=0.1,
                        num_fcs=2,
                        type='FFN'),
                    norm_cfg=dict(type='LN'),
                    operation_order=(
                        'self_attn',
                        'norm',
                        'cross_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='BaseTransformerLayer'),
                type='DetectionTransformerDecoder'),
            embed_dims=256,
            encoder=dict(
                num_layers=3,
                num_points_in_pillar=4,
                pc_range=[
                    -51.2,
                    -51.2,
                    -5.0,
                    51.2,
                    51.2,
                    3.0,
                ],
                return_intermediate=False,
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            embed_dims=256,
                            num_levels=1,
                            type='TemporalSelfAttention'),
                        dict(
                            deformable_attention=dict(
                                embed_dims=256,
                                num_levels=1,
                                num_points=8,
                                type='MultiScaleDeformableAttention3D'),
                            embed_dims=256,
                            pc_range=[
                                -51.2,
                                -51.2,
                                -5.0,
                                51.2,
                                51.2,
                                3.0,
                            ],
                            type='SpatialCrossAttention'),
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'cross_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='BEVFormerLayer'),
                type='BEVFormerEncoder'),
            rotate_prev_bev=True,
            type='PerceptionTransformer',
            use_can_bus=True,
            use_shift=True),
        type='BEVFormerHead',
        with_box_refine=True),
    train_cfg=dict(
        pts=dict(
            grid_size=[
                512,
                512,
                1,
            ],
            out_size_factor=4,
            point_cloud_range=[
                -51.2,
                -51.2,
                -5.0,
                51.2,
                51.2,
                3.0,
            ],
            voxel_size=[
                0.2,
                0.2,
                8,
            ])),
    type='BEVFormer',
    use_grid_mask=True,
    video_test_mode=True)
optim_wrapper = dict(
    _scope_='mmdet',
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.0002, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')
optimizer = dict(lr=0.0002, type='AdamW', weight_decay=0.01)
paramwise_cfg = dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1)))
plugin = False
plugin_dir = None
point_cloud_range = [
    -51.2,
    -51.2,
    -5.0,
    51.2,
    51.2,
    3.0,
]
queue_length = 3
randomness = dict(seed=2024)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        bev_size=(
            150,
            150,
        ),
        data_prefix=dict(
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            img='',
            pts='samples/LIDAR_TOP',
            sweeps='sweeps/LIDAR_TOP'),
        data_root='data/nuscenes',
        metainfo=dict(
            classes=[
                'car',
                'truck',
                'construction_vehicle',
                'bus',
                'trailer',
                'barrier',
                'motorcycle',
                'bicycle',
                'pedestrian',
                'traffic_cone',
            ],
            version='v1.0-mini'),
        modality=dict(
            use_camera=True,
            use_external=True,
            use_lidar=False,
            use_map=False,
            use_radar=False),
        pipeline=[
            dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
            dict(
                divider=1.0,
                mean=[
                    103.53,
                    116.28,
                    123.675,
                ],
                std=[
                    1.0,
                    1.0,
                    1.0,
                ],
                to_rgb=False,
                type='NormalizeMultiviewImage'),
            dict(
                flip=False,
                img_scale=(
                    1600,
                    900,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(
                        scales=[
                            0.8,
                        ], type='RandomScaleImageMultiViewImage'),
                    dict(size_divisor=32, type='PadMultiViewImage'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(_scope_='mmdet3d', keys=[
                'img',
            ], type='Pack3DDetInputs'),
        ],
        serialize_data=False,
        test_mode=True,
        type='NuScenesDatasetBEVFormer'),
    num_workers=1,
    pin_memory=True,
    sampler=dict(_scope_='mmengine', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    _scope_='mmdet3d',
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',
    data_root='data/nuscenes',
    jsonfile_prefix='eval',
    metric='bbox',
    modality=dict(
        use_camera=True,
        use_external=True,
        use_lidar=False,
        use_map=False,
        use_radar=False),
    type='NuScenesMetric')
test_pipeline = [
    dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
    dict(
        divider=1.0,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        std=[
            1.0,
            1.0,
            1.0,
        ],
        to_rgb=False,
        type='NormalizeMultiviewImage'),
    dict(
        flip=False,
        img_scale=(
            1600,
            900,
        ),
        pts_scale_ratio=1,
        transforms=[
            dict(scales=[
                0.8,
            ], type='RandomScaleImageMultiViewImage'),
            dict(size_divisor=32, type='PadMultiViewImage'),
        ],
        type='MultiScaleFlipAug3D'),
    dict(_scope_='mmdet3d', keys=[
        'img',
    ], type='Pack3DDetInputs'),
]
total_epochs = 24
train_cfg = dict(max_epochs=24, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='nuscenes_infos_train.pkl',
        bev_size=(
            150,
            150,
        ),
        box_type_3d='LiDAR',
        data_prefix=dict(
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            img='',
            pts='samples/LIDAR_TOP',
            sweeps='sweeps/LIDAR_TOP'),
        data_root='data/nuscenes',
        metainfo=dict(
            classes=[
                'car',
                'truck',
                'construction_vehicle',
                'bus',
                'trailer',
                'barrier',
                'motorcycle',
                'bicycle',
                'pedestrian',
                'traffic_cone',
            ],
            version='v1.0-mini'),
        modality=dict(
            use_camera=True,
            use_external=True,
            use_lidar=False,
            use_map=False,
            use_radar=False),
        pipeline=[
            dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(
                _scope_='mmdet3d',
                type='LoadAnnotations3D',
                with_attr_label=False,
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                point_cloud_range=[
                    -51.2,
                    -51.2,
                    -5.0,
                    51.2,
                    51.2,
                    3.0,
                ],
                type='ObjectRangeFilter'),
            dict(
                classes=[
                    'car',
                    'truck',
                    'construction_vehicle',
                    'bus',
                    'trailer',
                    'barrier',
                    'motorcycle',
                    'bicycle',
                    'pedestrian',
                    'traffic_cone',
                ],
                type='ObjectNameFilter'),
            dict(
                divider=1.0,
                mean=[
                    103.53,
                    116.28,
                    123.675,
                ],
                std=[
                    1.0,
                    1.0,
                    1.0,
                ],
                to_rgb=False,
                type='NormalizeMultiviewImage'),
            dict(scales=[
                0.8,
            ], type='RandomScaleImageMultiViewImage'),
            dict(size_divisor=32, type='PadMultiViewImage'),
            dict(
                _scope_='mmdet3d',
                keys=[
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                    'img',
                ],
                type='Pack3DDetInputs'),
        ],
        queue_length=3,
        test_mode=False,
        type='NuScenesDatasetBEVFormer',
        use_valid_flag=True),
    num_workers=1,
    pin_memory=True,
    sampler=dict(_scope_='mmengine', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        _scope_='mmdet3d',
        type='LoadAnnotations3D',
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True),
    dict(
        point_cloud_range=[
            -51.2,
            -51.2,
            -5.0,
            51.2,
            51.2,
            3.0,
        ],
        type='ObjectRangeFilter'),
    dict(
        classes=[
            'car',
            'truck',
            'construction_vehicle',
            'bus',
            'trailer',
            'barrier',
            'motorcycle',
            'bicycle',
            'pedestrian',
            'traffic_cone',
        ],
        type='ObjectNameFilter'),
    dict(
        divider=1.0,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        std=[
            1.0,
            1.0,
            1.0,
        ],
        to_rgb=False,
        type='NormalizeMultiviewImage'),
    dict(scales=[
        0.8,
    ], type='RandomScaleImageMultiViewImage'),
    dict(size_divisor=32, type='PadMultiViewImage'),
    dict(
        _scope_='mmdet3d',
        keys=[
            'gt_bboxes_3d',
            'gt_labels_3d',
            'img',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        bev_size=(
            150,
            150,
        ),
        data_prefix=dict(
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            img='',
            pts='samples/LIDAR_TOP',
            sweeps='sweeps/LIDAR_TOP'),
        data_root='data/nuscenes',
        metainfo=dict(
            classes=[
                'car',
                'truck',
                'construction_vehicle',
                'bus',
                'trailer',
                'barrier',
                'motorcycle',
                'bicycle',
                'pedestrian',
                'traffic_cone',
            ],
            version='v1.0-mini'),
        modality=dict(
            use_camera=True,
            use_external=True,
            use_lidar=False,
            use_map=False,
            use_radar=False),
        pipeline=[
            dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
            dict(
                divider=1.0,
                mean=[
                    103.53,
                    116.28,
                    123.675,
                ],
                std=[
                    1.0,
                    1.0,
                    1.0,
                ],
                to_rgb=False,
                type='NormalizeMultiviewImage'),
            dict(
                flip=False,
                img_scale=(
                    1600,
                    900,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(
                        scales=[
                            0.8,
                        ], type='RandomScaleImageMultiViewImage'),
                    dict(size_divisor=32, type='PadMultiViewImage'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(_scope_='mmdet3d', keys=[
                'img',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='NuScenesDatasetBEVFormer'),
    num_workers=1,
    pin_memory=True,
    sampler=dict(_scope_='mmengine', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    _scope_='mmdet3d',
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',
    data_root='data/nuscenes',
    jsonfile_prefix='eval',
    metric='bbox',
    modality=dict(
        use_camera=True,
        use_external=True,
        use_lidar=False,
        use_map=False,
        use_radar=False),
    type='NuScenesMetric')
version = 'v1.0-mini'
voxel_size = [
    0.2,
    0.2,
    8,
]
work_dir = '.'
