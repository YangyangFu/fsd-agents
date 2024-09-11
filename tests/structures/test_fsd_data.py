import pytest
import torch
from fsd.structures import TrajectoryData, MultiModalTrajectoryData
from fsd.utils import seed_everything
# seed everthing
@pytest.fixture(autouse=True)
def seed():
    seed_everything(2024)

    
def test_TrajectoryData():
    
    # test len()
    meta = {'time': 0,
            'time_step': 0.1,
            'num_past_steps': 2, 
            'num_future_steps': 2
            }
    traj = TrajectoryData(metainfo=meta)
    assert len(traj) == 0
    
    xyzr = torch.rand((5, 4))
    mask = torch.rand((5,))

    traj = TrajectoryData(data=xyzr, mask=mask)
    assert len(traj) == 5
    assert not hasattr(traj, 'num_past_steps')
    assert not hasattr(traj, 'num_future_steps')
    
    # TODO: no check of dimension consistency
    traj = TrajectoryData(metainfo = meta,
                        data=torch.rand((6, 4)))
    assert meta['num_past_steps'] + meta['num_future_steps'] + 1 != len(traj)
    
    with pytest.raises(AssertionError):
        traj = TrajectoryData(data = torch.rand((5, 2, 4)))
    
    # test __getitem__
    xy = torch.arange(10).reshape(5, 2)
    mask = torch.tensor([0, 1, 0, 1, 0])
    traj = TrajectoryData(metainfo=meta,
                          data=xy, 
                          mask=mask)
    # int index
    assert traj[0].data.tolist() == [[0, 1]] and traj[0].mask.tolist() == [0]
    assert traj[1].data.tolist() == [[2, 3]] and traj[1].mask.tolist() == [1]
    assert traj[2].data.tolist() == [[4, 5]] and traj[2].mask.tolist() == [0]
    assert traj[3].data.tolist() == [[6, 7]] and traj[3].mask.tolist() == [1]
    assert traj[4].data.tolist() == [[8, 9]] and traj[4].mask.tolist() == [0]
    with pytest.raises(IndexError):
        traj[5]
    
    # slice
    assert traj[1:3].data.tolist() == [[2, 3], [4, 5]] and traj[1:3].mask.tolist() == [1, 0]
    assert traj[3:8].data.tolist() == [[6, 7], [8, 9]] and traj[3:8].mask.tolist() == [1, 0]


def test_MultiModalTrajectoryData():
    # meta
    meta = {'time': 0,
            'time_step': 0.1,
            'num_past_steps': 2, 
            'num_future_steps': 2
            }
    # no data
    traj = MultiModalTrajectoryData(metainfo=meta)
    assert len(traj) == 0
    assert hasattr(traj, 'num_past_steps')
    assert hasattr(traj, 'num_future_steps')
    
    # modality
    xyzr = torch.rand((5, 2, 4))
    mask = torch.rand((5,))

    traj = MultiModalTrajectoryData(data=xyzr, mask=mask)
    assert len(traj) == 5
    assert traj.num_modalities == 2
    assert not hasattr(traj, 'num_past_steps')
    assert not hasattr(traj, 'num_future_steps')
    
    # index
    assert traj[0].data.shape == (1, 2, 4)
    
pytest.main(['-sv', 'tests/structures/test_fsd_data.py'])