import torch 

from mmengine.registry import init_default_scope
from fsd.structures import PlanningDataSample, Ego, Instances, TrajectoryData
from fsd.registry import METRICS

def _init_data_samples(bsize: int) -> list:
   # generate random data
    data_samples = []
    bsize = 2
    for _ in range(bsize):
        data_sample = PlanningDataSample()
        
        # ego data
        ego = Ego()
        gt_traj = TrajectoryData(metainfo={'num_planning_steps': 10})
        gt_traj.data = torch.randn(10, 2)
        gt_traj.mask = torch.ones(10)
        pred_traj = gt_traj.clone()
        pred_traj.data = torch.randn(10, 2)
        
        ego.gt_traj = gt_traj
        ego.pred_traj = pred_traj
        data_sample.ego = ego
        
        # random 4 instances traj
        num_instances = 4
        gt_trajs = []
        pred_trajs = []
        for _ in range(num_instances):
            traj = TrajectoryData(metainfo={'num_planning_steps': 10})
            traj.data = torch.randn(10, 2)
            traj.mask = torch.ones(10)
            gt_trajs.append(traj)
            
            traj = traj.clone()
            traj.data = torch.randn(10, 2)
            pred_trajs.append(traj)
        
        instances = Instances()
        instances.gt_traj = gt_trajs
        instances.pred_traj = pred_trajs
        data_sample.instances = instances
        
        data_samples.append(data_sample)
    
    return data_samples    

def test_TrajectoryMetric():
    cfg = dict(
        type='TrajectoryMetric',
    )
    init_default_scope('fsd')
    traj_metric = METRICS.build(cfg)
    
    # init data samples
    data_samples = _init_data_samples(2)
    
    # check metrics
    traj_metric.process(None, data_samples)
    metrics = traj_metric.compute_metrics(traj_metric.results)
    
    print(metrics)


test_TrajectoryMetric()