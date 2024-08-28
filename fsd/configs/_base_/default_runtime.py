# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmengine.hooks.iter_timer_hook import IterTimerHook
from mmengine.hooks.logger_hook import LoggerHook
from mmengine.runner.log_processor import LogProcessor



default_scope = 'fsd'

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(
        type=LoggerHook, 
        interval=50
    ),
    checkpoint=dict(
        type=CheckpointHook, 
        interval=10000, 
        by_epoch=False
    )
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type=LogProcessor, window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

# TODO: support auto scaling lr
