import pytest 

import torch
import torch.nn as nn
from fsd.registry import TASK_UTILS 
from fsd.utils import seed_everything
# seed everthing
@pytest.fixture(autouse=True)
def seed():
    seed_everything(2024)


def test_object_density_head():
    # head config
    cfg = dict(
        type='interfuser_object_density',
        input_size=2048,
        hidden_size=64,
        output_size=7,
        loss_cfg=dict(
            type='L1Loss',
            _scope_='mmdet',
            reduction='mean',
            loss_weight=1.0
        )
    )

    head = TASK_UTILS.build(cfg=cfg)
    head.init_weights()
    
    # inputs
    inputs = torch.randn(2, 400, 2048)
    
    # run forward
    outputs = head(inputs)

    
    targets = torch.randn(2, 400, 7)
    loss = head.loss(inputs, targets)

    assert outputs.shape == (2, 400, 7) 

def test_gru_waypoint_head():
    # head config
    cfg = dict(
        type='interfuser_gru_waypoint',
        num_waypoints=10,
        input_size=2048,
        hidden_size=256,
        num_layers=1,
        dropout=0.,
        batch_first=True,
        loss_cfg=dict(
            type='MaskedSmoothL1Loss',
            beta=1.0,
            reduction='mean',
            loss_weight=1.0
        ),
        waypoints_weights=[
            0.1407441030399059,
            0.13352157985305926,
            0.12588535273178575,
            0.11775496498388233,
            0.10901991343009122,
            0.09952110967153563,
            0.08901438656870617,
            0.07708872007078788,
            0.06294267636589287,
            0.04450719328435308,
        ])

    head = TASK_UTILS.build(cfg=cfg)

    # inputs
    inputs = torch.randn(2, 10, 2048)
    goal_points = torch.randn(2, 2)
    
    # run forward
    outputs = head(inputs, goal_points)
    
    targets = torch.randn(2, 10, 2)
    loss = head.loss(hidden_states = inputs, 
                     goal_points = goal_points,
                     target_waypoints = targets)
    print(loss)
    
    assert outputs.shape == (2, 10, 2)

@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_gru_waypoint_head_cuda():
    # head config
    cfg = dict(
        type='interfuser_gru_waypoint',
        num_waypoints=10,
        input_size=2048,
        hidden_size=256,
        num_layers=1,
        dropout=0.,
        batch_first=True,
        loss_cfg=dict(
            type='MaskedSmoothL1Loss',
            beta=1.0,
            reduction='mean',
            loss_weight=1.0
        ),
        waypoints_weights=[
            0.1407441030399059,
            0.13352157985305926,
            0.12588535273178575,
            0.11775496498388233,
            0.10901991343009122,
            0.09952110967153563,
            0.08901438656870617,
            0.07708872007078788,
            0.06294267636589287,
            0.04450719328435308,
        ])

    head = TASK_UTILS.build(cfg=cfg).to('cuda')

    # inputs
    inputs = torch.randn(2, 10, 2048).cuda()
    goal_points = torch.randn(2, 2).cuda()
    
    # run forward
    outputs = head(inputs, goal_points)
    
    targets = torch.randn(2, 10, 2).cuda()
    loss = head.loss(hidden_states = inputs, 
                     goal_points = goal_points,
                     target_waypoints = targets)
    print(loss)
    
    assert outputs.shape == (2, 10, 2)

def test_stop_sign_head():
    # head config
    cfg = dict(
        type='interfuser_traffic_rule',
        input_size=2048,
        output_size=2,
        loss_cfg=dict(
            type='CrossEntropyLoss',
            _scope_='mmdet',
            use_sigmoid=True, # binary classification
            reduction='mean',
            loss_weight=1.0
        )
    )

    head = TASK_UTILS.build(cfg=cfg)
    head.init_weights()
    
    # inputs
    inputs = torch.randn(2, 1, 2048)
    
    # run forward
    outputs = head(inputs)

    
    targets = torch.randn(2, 1, 2)
    loss = head.loss(inputs, targets)

    assert outputs.shape == (2, 1, 2)
    assert torch.abs(loss - 0.4489) <= 1e-4
    
    inputs = torch.randn(2, 2048)
    outputs = head(inputs)
    assert outputs.shape == (2, 2)

def test_junction():
    # head config
    cfg = dict(
        type='interfuser_traffic_rule',
        input_size=2048,
        output_size=2,
        loss_cfg=dict(
            type='CrossEntropyLoss',
            _scope_='mmdet',
            use_sigmoid=True, # binary classification
            reduction='mean',
            loss_weight=1.0
        )
    )

    head = TASK_UTILS.build(cfg=cfg)
    head.init_weights()
    
    # inputs
    inputs = torch.randn(2, 1, 2048)
    
    # run forward
    outputs = head(inputs)

    
    targets = torch.randn(2, 1, 2)
    loss = head.loss(inputs, targets)

    assert outputs.shape == (2, 1, 2)

    inputs = torch.randn(2, 2048)
    outputs = head(inputs)
    assert outputs.shape == (2, 2)

def test_traffic_light():
    # head config
    cfg = dict(
        type='interfuser_traffic_rule',
        input_size=2048,
        output_size=2,
        loss_cfg=dict(
            type='CrossEntropyLoss',
            _scope_='mmdet',
            use_sigmoid=True, # binary-class classification
            reduction='mean',
            loss_weight=1.0
        )
    )

    head = TASK_UTILS.build(cfg=cfg)
    head.init_weights()
    
    # inputs
    inputs = torch.randn(2, 1, 2048)
    outputs = head(inputs)

    targets = torch.randn(2, 1, 2)
    loss = head.loss(inputs, targets)

    assert outputs.shape == (2, 1, 2)

    inputs = torch.randn(2, 2048)
    outputs = head(inputs)
    assert outputs.shape == (2, 2)


def test_interfuser_heads():
    cfg=dict(
        type='interfuser_heads',
        num_waypoints_queries=10,
        num_traffic_rule_queries=1,
        num_object_density_queries=400,
        waypoints_head=dict(
            type='interfuser_gru_waypoint',
            num_waypoints=10,
            input_size=2048,
            hidden_size=256,
            num_layers=1,
            dropout=0.,
            #batch_first=True,
            loss_cfg=dict(
                type='SmoothL1Loss',
                _scope_='mmdet',
                beta=1.0,
                reduction='mean',
                loss_weight=1.0
            ),
            waypoints_weights=[
                0.1407441030399059,
                0.13352157985305926,
                0.12588535273178575,
                0.11775496498388233,
                0.10901991343009122,
                0.09952110967153563,
                0.08901438656870617,
                0.07708872007078788,
                0.06294267636589287,
                0.04450719328435308,
            ]),
        object_density_head=dict(
            type='interfuser_object_density',
            input_size=2048 + 32,
            hidden_size=64,
            output_size=7,
            loss_cfg=dict(
                type='L1Loss',
                _scope_='mmdet',
                reduction='mean',
                loss_weight=1.0
            )
        ),
        junction_head=dict(
            type='interfuser_traffic_rule',
            input_size=2048,
            output_size=2,
            loss_cfg=dict(
                type='CrossEntropyLoss',
                _scope_='mmdet',
                use_sigmoid=True, # binary classification
                reduction='mean',
                loss_weight=1.0
            )
        ),
        stop_sign_head=dict(
            type='interfuser_traffic_rule',
            input_size=2048,
            output_size=2,
            loss_cfg=dict(
                type='CrossEntropyLoss',
                _scope_='mmdet',
                use_sigmoid=True, # binary classification
                reduction='mean',
                loss_weight=1.0
            )
        ),
        traffic_light_head=dict(
            type='interfuser_traffic_rule',
            input_size=2048,
            output_size=2,
            loss_cfg=dict(
                type='CrossEntropyLoss',
                _scope_='mmdet',
                use_sigmoid=True, # binary classification
                reduction='mean',
                loss_weight=1.0
            )
        )
    )
    
    heads = TASK_UTILS.build(cfg=cfg)
    inputs = torch.randn(2, 411, 2048)
    goal_points = torch.randn(2, 2)
    ego_velocity = torch.randn(2, 1)
    
    outputs = heads(inputs, goal_points, ego_velocity) 
    
    assert outputs['waypoints'].shape == (2, 10, 2)
    assert outputs['object_density'].shape == (2, 400, 7)
    assert outputs['junction'].shape == (2, 2)
    assert outputs['stop_sign'].shape == (2, 2)
    assert outputs['traffic_light'].shape == (2, 2)

# run pytest 
#test_interfuser_heads()
pytest.main(["-v", "--tb=line", __file__])
