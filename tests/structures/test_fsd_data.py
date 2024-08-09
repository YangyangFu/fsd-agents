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

    traj = TrajectoryData(xy=xy, mask=mask)
    assert len(traj) == 5
    
    # catch wrong dimension
    with pytest.raises(AssertionError):
        traj = TrajectoryData(xy=torch.rand((5, 3)))
    with pytest.raises(AssertionError):
        traj.mask = torch.rand((5, 2))

    # test __getitem__
    xy = torch.arange(10).reshape(5, 2)
    mask = torch.tensor([0, 1, 0, 1, 0])
    traj = TrajectoryData(xy=xy, mask=mask)
    # int index
    assert traj[0].xy.tolist() == [[0, 1]] and traj[0].mask.tolist() == [0]
    assert traj[1].xy.tolist() == [[2, 3]] and traj[1].mask.tolist() == [1]
    assert traj[2].xy.tolist() == [[4, 5]] and traj[2].mask.tolist() == [0]
    assert traj[3].xy.tolist() == [[6, 7]] and traj[3].mask.tolist() == [1]
    assert traj[4].xy.tolist() == [[8, 9]] and traj[4].mask.tolist() == [0]
    with pytest.raises(IndexError):
        traj[5]
    
    # slice
    assert traj[1:3].xy.tolist() == [[2, 3], [4, 5]] and traj[1:3].mask.tolist() == [1, 0]
    assert traj[3:8].xy.tolist() == [[6, 7], [8, 9]] and traj[3:8].mask.tolist() == [1, 0]
    
    # test sort 
    
    # test overloaded arithmetic operators
    traj1 = TrajectoryData(xy=torch.tensor([[1,2]]), mask=torch.ones((1,))) # one instance
    traj2 = TrajectoryData(xy=torch.arange(3, 11).reshape(4, 2), mask=torch.tensor([1, 1, 0, 0])) # 4 instances
    traj = traj2 - traj1 
    assert traj.xy.tolist() == [[2, 2], [4, 4], [6, 6], [8, 8]]
    assert traj.mask.tolist() == [1, 1, 0, 0]

    # test cat multiple instances
    traj = TrajectoryData()
    assert len(traj) == 0
    
    traj = traj.cat([traj1])
    assert len(traj) == 1
    assert traj.xy.shape == (1, 2)
    
    traj = traj.cat([traj, traj2])
    assert len(traj) == 5
    assert traj.num_instances == 5
    assert traj.xy.shape == (5, 2)
    
    with pytest.raises(AssertionError):
        traj1 = TrajectoryData(xy=torch.arange(12).reshape(2, 2, 3), mask=torch.tensor([[1, 0, 0], [1, 0, 0]])) # two instance, three steps
        traj = traj.cat([traj, traj1])
    
    # test stack multiple trajectories with different time steps
    traj = TrajectoryData()
    traj1 = TrajectoryData(xy=torch.tensor([[1,2], [3,4]]), mask=torch.tensor([1, 0])) # two instance, one step
    traj = traj.stack([traj1])
    assert len(traj) == 2
    assert traj.num_steps == 1
    assert traj.xy.shape == (2, 2)
    
    traj = traj.stack([traj, traj1])
    assert len(traj) == 2 
    assert traj.num_steps == 2
    assert traj.xy.shape == (2, 2, 2)
    
    with pytest.raises(AssertionError):
        traj = traj.stack([traj, traj2])
    
pytest.main(['-sv', 'tests/structures/test_fsd_data.py'])