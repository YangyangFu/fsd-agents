import pytest

import torch
from mmengine.structures import BaseDataElement, InstanceData, PixelData
from fsd.structures import PlanningDataSample, TrajectoryData
from fsd.utils import seed_everything
@pytest.fixture(autouse=True)
def seed():
    seed_everything(2024)

def test_PlanningDataSample():
    # test gt_ego
    data_sample = PlanningDataSample()
    gt_ego = BaseDataElement()
    gt_ego.goal_point = torch.rand(2)
    gt_ego.ego2world = torch.rand(4, 4)
    data_sample.gt_ego = gt_ego
    assert 'gt_ego' in data_sample
    assert data_sample.gt_ego == gt_ego
    del data_sample.gt_ego
    assert 'gt_ego' not in data_sample

    # test gt_instances_3d
    past_traj = TrajectoryData(xy=torch.rand(5, 2, 4), mask=torch.randint(0, 2, (5, 4)))
    future_traj = TrajectoryData(xy=torch.rand(5, 2, 8), mask=torch.randint(0, 2, (5, 8)))
    
    gt_instances_3d_data = dict(
        metainfo=dict(timestamp=1.0),
        bboxes=torch.rand(5, 7),
        labels=torch.randint(0, 10, (5,)),
        ids=torch.randint(0, 10, (5,)),
        bboxes_mask=torch.randint(0, 2, (5,)),
        bboxes2world=torch.rand(5, 4, 4),
        past_traj=past_traj,
        future_traj=future_traj,
    )
    gt_instances_3d = InstanceData(**gt_instances_3d_data)
    data_sample.gt_instances_3d = gt_instances_3d
    assert 'gt_instances_3d' in data_sample
    assert 'bboxes' in data_sample.gt_instances_3d
    assert data_sample.gt_instances_3d == gt_instances_3d
    del data_sample.gt_instances_3d
    assert 'gt_instances_3d' not in data_sample

    # test gt_pts_seg
    data_sample = PlanningDataSample()
    gt_pts_seg_data = dict(
        pts_instance_mask=torch.rand(100, 2),
        pts_semantic_mask=torch.rand(100, 2),
    )
    gt_pts_seg = PixelData(**gt_pts_seg_data)
    data_sample.gt_pts_seg = gt_pts_seg
    assert 'gt_pts_seg' in data_sample
    assert data_sample.gt_pts_seg == gt_pts_seg
    del data_sample.gt_pts_seg
    assert 'gt_pts_seg' not in data_sample


pytest.main(['-sv', 'tests/structures/test_fsd_data_sample.py'])
