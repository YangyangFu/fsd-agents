import pytest
import torch
from fsd.structures import TrajectoryData
from fsd.utils import seed_everything
# seed everthing
@pytest.fixture(autouse=True)
def seed():
    seed_everything(2024)

    
def test_TrajectoryData():
    
    # test len()
    traj = TrajectoryData()
    assert len(traj) == 0
    
    xy = torch.rand((5, 2))
    mask = torch.rand((5,))
    timestamps = torch.rand((5,))
    
    traj = TrajectoryData(xy=xy, mask=mask, timestamps=timestamps)
    assert len(traj) == 5
    
    # catch wrong dimension
    with pytest.raises(AssertionError):
        traj = TrajectoryData(xy=torch.rand((5, 3)))
    with pytest.raises(AssertionError):
        traj = TrajectoryData(mask=torch.rand((5, 2)))
    with pytest.raises(AssertionError):
        traj = TrajectoryData(timestamps=torch.rand((5, 2)))    
    
    # test __getitem__
    xy = torch.arange(10).reshape(5, 2)
    mask = torch.tensor([0, 1, 0, 1, 0])
    timestamps = torch.arange(5)
    traj = TrajectoryData(xy=xy, mask=mask, timestamps=timestamps)
    # int index
    assert traj[0].xy.tolist() == [[0, 1]] and traj[0].mask.tolist() == [0] and traj[0].timestamps.tolist() == [0]
    assert traj[1].xy.tolist() == [[2, 3]] and traj[1].mask.tolist() == [1] and traj[1].timestamps.tolist() == [1]
    assert traj[2].xy.tolist() == [[4, 5]] and traj[2].mask.tolist() == [0] and traj[2].timestamps.tolist() == [2]
    assert traj[3].xy.tolist() == [[6, 7]] and traj[3].mask.tolist() == [1] and traj[3].timestamps.tolist() == [3]
    assert traj[4].xy.tolist() == [[8, 9]] and traj[4].mask.tolist() == [0] and traj[4].timestamps.tolist() == [4]
    with pytest.raises(IndexError):
        traj[5]
    
    # slice
    assert traj[1:3].xy.tolist() == [[2, 3], [4, 5]] and traj[1:3].mask.tolist() == [1, 0] and traj[1:3].timestamps.tolist() == [1, 2]
    assert traj[3:8].xy.tolist() == [[6, 7], [8, 9]] and traj[3:8].mask.tolist() == [1, 0] and traj[3:8].timestamps.tolist() == [3, 4]
    
    # test sort 
    xy = torch.rand((5, 2))
    mask = torch.rand((5,))
    timestamps = torch.rand((5,))
    traj = TrajectoryData(xy=xy, mask=mask, timestamps=timestamps)
    traj_sorted = traj[traj.timestamps.sort().indices]

    assert traj_sorted.timestamps.tolist() == sorted(traj.timestamps.tolist())
    assert traj_sorted.xy.tolist() == [traj.xy[i].tolist() for i in traj.timestamps.sort().indices]
    assert traj_sorted.mask.tolist() == [traj.mask[i].tolist() for i in traj.timestamps.sort().indices]
    
    # test overloaded arithmetic operators
    traj1 = TrajectoryData(xy=torch.tensor([[1,2]]), mask=torch.ones((1,)), timestamps=torch.ones((1,)))
    traj2 = TrajectoryData(xy=torch.arange(3, 11).reshape(4, 2), mask=torch.tensor([1, 1, 0, 0]), timestamps=torch.arange(4))
    traj = traj2 - traj1 
    assert traj.xy.tolist() == [[2, 2], [4, 4], [6, 6], [8, 8]]
    assert traj.mask.tolist() == [1, 1, 0, 0]
    assert traj.timestamps.tolist() == [-1, 0, 1, 2]

    
pytest.main(['-sv', 'test_fsd_data.py'])