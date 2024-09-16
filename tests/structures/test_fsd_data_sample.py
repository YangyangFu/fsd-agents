import pytest

import torch
from mmengine.structures import BaseDataElement, InstanceData, PixelData
from mmdet3d.structures import LiDARInstance3DBoxes
from fsd.structures import PlanningDataSample, TrajectoryData, Ego, Instances, Grids
from fsd.utils import seed_everything

@pytest.fixture(autouse=True)
def seed():
    seed_everything(2024)

def test_PlanningDataSample():
    # test gt_ego
    data_sample = PlanningDataSample()
    gt_ego = Ego()
    gt_ego.goal_point = torch.rand(2)
    gt_ego.pose = torch.rand(4, 4)
    data_sample.gt_ego = gt_ego
    assert 'gt_ego' in data_sample
    assert data_sample.gt_ego == gt_ego
    del data_sample.gt_ego
    assert 'gt_ego' not in data_sample

    # test gt_instances_3d
    gt_traj = [TrajectoryData(data=torch.rand(4,2), mask=torch.randint(0, 2, (4, ))) for i in range(5)]
    gt_bboxes_3d = LiDARInstance3DBoxes(torch.rand(5, 7))
    gt_instances_3d_data = dict(
        metainfo=dict(timestamp=1.0),
        bboxes_3d=gt_bboxes_3d,
        labels=torch.randint(0, 10, (5,)),
        ids=torch.randint(0, 10, (5,)),
        bboxes_mask=torch.randint(0, 2, (5,)),
        poses=torch.rand(5, 4, 4),
        traj = gt_traj
    )
    gt_instances_3d = Instances(**gt_instances_3d_data)
    data_sample.gt_instances = gt_instances_3d
    assert 'gt_instances' in data_sample
    assert 'bboxes_3d' in data_sample.gt_instances
    assert data_sample.gt_instances == gt_instances_3d
    del data_sample.gt_instances
    assert 'gt_instances' not in data_sample

    # TODO: test gt_pts_seg


pytest.main(['-sv', 'tests/structures/test_fsd_data_sample.py'])
