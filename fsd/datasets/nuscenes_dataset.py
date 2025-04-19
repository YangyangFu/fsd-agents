# Nuscenes dataset for planning tasks
from os import path as osp
from typing import Callable, List, Union, Optional
import numpy as np 
from pyquaternion import Quaternion

from mmengine.fileio import load
from mmdet3d.structures import limit_period, CameraInstance3DBoxes, LiDARInstance3DBoxes
from mmdet3d.datasets import Det3DDataset
from fsd.datasets import BasePlanDataset
from fsd.datasets.convert_utils import nus_categories, NuScenesNameMapping
from fsd.structures import TrajectoryData
from fsd.utils import one_hot_encoding
from fsd.registry import DATASETS

# FOR NUSCENES
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

@DATASETS.register_module()
class NuscenesDatasetPlan3D(BasePlanDataset):
    """NuScenes dataset for 3D planning tasks.
    
    """
    # transformation matrix from dataset lidar coordinate to mmdet3d lidar
    # now assumes no transformation
    TO_MMDET3D_LIDAR = np.eye(4)
    
    METAINFO = {
        'name': 'nuscenes',
        'classes': nus_categories,
        'version': 'v1.0-trainval',
        'palette': [
            (255, 158, 0),  # Orange
            (255, 99, 71),  # Tomato
            (255, 140, 0),  # Darkorange
            (255, 127, 80),  # Coral
            (233, 150, 70),  # Darksalmon
            (220, 20, 60),  # Crimson
            (255, 61, 99),  # Red
            (0, 0, 230),  # Blue
            (47, 79, 79),  # Darkslategrey
            (112, 128, 144),  # Slategrey
        ]
    }
    
    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(pts='velodyne', img=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=False, use_camera=True),
                 camera_sensors: List[str] = ['CAM_FRONT'],
                 lidar_sensors: List[str] = ['LIDAR_TOP'],
                 box_type_3d_original: str = 'Depth', # box cooridnate in the original annotation file
                 box_type_3d: str = 'LiDAR', # targeted box coordinate for the dataset
                 filter_empty_gt: bool = True,
                 past_steps: int = 4, # past trajectory length
                 prediction_steps: int = 6, # motion prediction length if any
                 planning_steps: int = 6, # planning length
                 sample_interval: int = 1, # sample interval # frames skiped per step
                 FPS: int = 2, # frame per second
                 test_mode: bool = False,
                 load_eval_anns: bool = True,
                 show_ins_var: bool = False,
                 **kwargs) -> None:
        
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            camera_sensors=camera_sensors,
            lidar_sensors=lidar_sensors,
            box_type_3d_original=box_type_3d_original,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            past_steps=past_steps,
            prediction_steps=prediction_steps,
            planning_steps=planning_steps,
            sample_interval=sample_interval,
            FPS=FPS,
            test_mode=test_mode,
            load_eval_anns=load_eval_anns,
            show_ins_var=show_ins_var,
            **kwargs)
        
        # add nuscenes 
        self.nusc = NuScenes(version=self.metainfo['version'], dataroot=self.data_root, verbose=False)
    
    def _get_nuscenes_box_pose(self, sample_annotation):
        """Get the pose of the box in the world coordinate system.
        """
        box_pose = np.eye(4)
        rotation = Quaternion(sample_annotation['rotation']).rotation_matrix
        translation = np.array(sample_annotation['translation'])
        box_pose[:3, :3] = rotation
        box_pose[:3, 3] = translation
        
        return box_pose
    
    def _generate_past_future_instances_trajectory1(self, index, curr_info):
        """Generate past and future trajectories for instances, 
            centered at the lidar coords in the current frame.

        Args:
            index (_type_): _description_
            info (_type_): _description_
        
        Returns:
            TrajectoryData: Trajectory data for N instances, with a length of (past_steps + 1 + planning_steps)
        """
        index_list = range(index - self.past_steps * self.sample_interval, 
                           index + self.planning_steps * self.sample_interval + 1, 
                           self.sample_interval)
        instances_ids = curr_info['ann_info']['gt_bboxes_id']
        lidar2ego = curr_info['lidar_points']['lidar2ego']
        ego2world = curr_info['ego2global']
        world2lidar_curr = np.linalg.inv(np.array(ego2world) @ np.array(lidar2ego))
        ann_tokens_curr = curr_info['ann_info']['gt_bboxes_anno_token']
        
        # initialize the trajectory data
        trajs = []
                
        # for each instance in the current frame, find its past and future trajectory
        for i, _ in enumerate(instances_ids):
            xyr = np.zeros((self.past_steps + 1 + self.planning_steps, 3)) # (T, 3)
            mask = np.zeros((self.past_steps + 1 + self.planning_steps,)) # (T,)    
            
            # box to lidar_curr
            instance2lidar_curr = world2lidar_curr @ curr_info['ann_info']['gt_bboxes_pose'][i] # (4, 4)
            xy_curr = instance2lidar_curr[:2, 3]
            r_curr = np.arctan2(instance2lidar_curr[1, 0], instance2lidar_curr[0, 0]) # [-pi, pi]
            # current xyr 
            xyr[self.past_steps, :2] = xy_curr
            xyr[self.past_steps, 2] = r_curr
            mask[self.past_steps] = 1
            
            # sample annotation 
            ann_token_curr = ann_tokens_curr[i]
            ann_curr = self.nusc.get('sample_annotation', ann_token_curr)
            
            # history traj
            _ann = ann_curr
            for j in range(self.past_steps-1, -1, -1):
                # sample interval
                for k in range(self.sample_interval):
                    if _ann['prev'] == '':
                        break
                    # get the prev sample annotation
                    _ann = self.nusc.get('sample_annotation', _ann['prev'])
                # extract trajectory
                # global coord
                box_pose = self._get_nuscenes_box_pose(_ann)
                # box to lidar_curr
                adj2curr = world2lidar_curr @ box_pose
                # save to the trajectory
                xyr[j, :2] = adj2curr[:2, 3]
                xyr[j, 2] = np.arctan2(adj2curr[1, 0], adj2curr[0, 0]) # [-pi, pi]
                mask[j] = 1
                
            # future traj
            _ann = ann_curr
            for j in range(self.past_steps+1, self.past_steps + self.planning_steps+1):
                # sample interval
                for k in range(self.sample_interval):
                    if _ann['next'] == '':
                        break
                    # get the next sample annotation
                    _ann = self.nusc.get('sample_annotation', _ann['next'])
                    
                # extract trajectory
                # global coord
                box_pose = self._get_nuscenes_box_pose(_ann)
                # box to lidar_curr
                adj2curr = world2lidar_curr @ box_pose
                
                # save to the trajectory
                xyr[j, :2] = adj2curr[:2, 3]
                xyr[j, 2] = np.arctan2(adj2curr[1, 0], adj2curr[0, 0]) # [-pi, pi]
                mask[j] = 1
            
            traj = TrajectoryData(
                    metainfo=dict(
                        mode = 'accumulated',
                        num_past_steps=self.past_steps, 
                        num_future_steps=self.planning_steps,
                        time_step=self.sample_interval/self.FPS)
            )
            traj.data=xyr.astype(np.float32)
            traj.mask=mask.astype(np.bool_)
            
            # to diff mode
            traj.convert_to_mode('difference')
            # save
            trajs.append(traj)
            
        return trajs
          

    
    
    