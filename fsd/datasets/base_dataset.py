# Copyright (c) OpenMMLab. All rights reserved. 
import numpy as np
import tempfile
import warnings
from os import path as osp
from torch.utils.data import Dataset

#from mmcv.datasets.builder import DATASETS
from fsd.registry import DATASETS

#from ..core.bbox import get_box_type
from mmdet3d.structures import get_box_type, LiDARInstance3DBoxes, DepthInstance3DBoxes, CameraInstance3DBoxes
from fsd.datasets.utils import extract_result_dict, get_loading_pipeline
from mmengine.fileio import load, dump, list_from_file
from mmengine.dataset import Compose
from mmengine.structures import InstanceData, BaseDataElement
from fsd.structures import TrajectoryData, PlanningDataSample

@DATASETS.register_module()
class Planning3DDataset(Dataset):
    """Customized 3D dataset.

    This is the base dataset of SUNRGB-D, ScanNet, nuScenes, and KITTI
    dataset.
    
    dataset pipelines:
    - prepare anno info to a standard strucuture
    - get data info: input data path and anno info
        - _get_pts_info
        - _get_imgs_info
        - _get_ann_info
    - pre_pipeline: prepare data before data processing
    - run_pipeline: run the pipeline for data transformation
    - prepare_train_data: prepare data for training
    
    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d_original (str, optional): Type of 3D box of the original annotation file.
            Defaults to 'Depth'. Available options includes
            - 'LiDAR': Box in LiDAR coordinates, e.g., x-front, y-left, z-up.
            - 'Depth': Box in depth coordinates, e.g., x-right, y-front, z-up as in NuScenes.
            - 'Camera': Box in camera coordinates, e.g., x-right, y-down, z-front.
        box_type_3d (str, optional): Type of 3D box of the dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR'. Available options includes
            - 'LiDAR': Box in LiDAR coordinates, e.g., x-front, y-left, z-up.
            - 'Depth': Box in depth coordinates, e.g., x-right, y-front, z-up.
            - 'Camera': Box in camera coordinates, e.g., x-right, y-down, z-front.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """
    # transformation matrix from dataset lidar coordinate to mmdet3d lidar
    # default is identity matrix
    TO_MMDET3D_LIDAR = np.eye(4)

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline = None,
                 classes = None,
                 modality = None,
                 camera_sensors = ['CAM_FRONT'],
                 lidar_sensors = None,
                 box_type_3d_original = 'Depth', # box cooridnate in the original annotation file
                 box_type_3d = 'LiDAR', # targeted box coordinate for the dataset
                 filter_empty_gt = True,
                 past_steps = 4, # past trajectory length
                 prediction_steps = 6, # motion prediction length if any
                 planning_steps = 6, # planning length
                 sample_interval = 5, # sample interval # frames skiped per step
                 FPS = 10, # frame per second
                 test_mode = False):
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.camera_sensors = [sensor.upper() for sensor in camera_sensors] if camera_sensors is not None else None
        self.lidar_sensors = [sensor.upper() for sensor in lidar_sensors] if lidar_sensors is not None else None
        self.filter_empty_gt = filter_empty_gt
        
        # past and future frames
        self.past_steps = past_steps
        self.prediction_steps = prediction_steps
        self.planning_steps = planning_steps
        self.sample_interval = sample_interval
        self.FPS = FPS
        
        self.box_type_3d_original = box_type_3d_original
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        
        self.CLASSES = self.get_classes(classes)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_anno_files(self.ann_file)

        self.num_samples = len(self.data_infos)
        
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

    def load_anno_files(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        return load(ann_file)
    
    def _check_if_annotation_is_valid(self, anno_info):
        """Check if the annotation is valid.
        
        Args:
            anno_info (dict): Annotation information.
            
        Returns:
            bool: Whether the annotation is valid.
        """
        required_keys = ['folder', 'scene_token', 'frame_idx', 'ego', 'sensors', 
                         'gt_bboxes_3d', 'gt_instances_names', 
                         'gt_instances_ids', 'gt_bboxes_mask', \
                          'gt_instances_velocities']
        for key in required_keys:
            if key not in anno_info:
                return False
        
        # check ego info
        ego_keys = ['world2ego', 'translation', 'yaw', 
                    'velocity', 'acceleration', 'size', 
                    'brake', 'throttle', 'steering']
        for key in ego_keys:
            if key not in anno_info['ego']:
                return False
        
        # check sensors info
        for sensor in anno_info['sensors'].keys():
            sensor_keys = ['sensor2ego', 'sensor2world', 'intrinsic', 'data_path']
            for key in sensor_keys:
                if key not in anno_info['sensors'][sensor]:
                    return False
        
        return True

    def prepare_planning_info(self, index):
        """ Prepare annotation info in the raw dataset 
            to a standard structured design for planning and control only.
            
            The resulting structure should be a dictionary with the following required keys:
            - 'folder': data folder relative to data root, e.g., data_root/folder/camera_xxx/xxx.png
            - 'scene_token': scene token
            - 'frame_index': frame index
            - 'ego': dict contains ego information
                - 'size': ego size, 2 times extent of the ego vehicle in x, y, z
                - 'ego2world': world to ego transformation
                - 'translation': ego translation
                - 'rotation': in quaternion
                - 'yaw': ego yaw
                - 'velocity': ego velocity
                - 'acceleration': ego acceleration
            - 'sensors': dict contains sensor information
                - 'CAM_XX': dict contains camera information
                    - 'sensor2ego': camera to ego transformation
                    - 'sensor2world': camera to world transformation
                    - 'intrinsic': camera intrinsics
                    - 'data_path': camera image path
                - 'LIDAR_XX': dict contains lidar information
                    - 'sensor2ego': lidar to ego transformation
                    - 'sensor2world': lidar to world transformation
                    - 'data_path': lidar data path
            - 'gt_bboxes_3d': ground truth boxes in the format of [x, y, z, w, l, h, ry]
            - 'gt_instances_names': ground truth class names
            - 'gt_bboxes_mask': ground truth boxes mask
            - 'gt_instances_velocities': ground truth velocities for all instances
            - 'gt_instances_ids': ground truth instance ids
            - 'gt_instances2world': ground truth instance pose in world coordinates
            - 'box_center': default center of the boxes, e.g., [0.5, 0.5, 0.5] for center in Nuscene, [0.5, 0.5, 0] for bottom center in KITTI
        """
        raw_info= self.data_infos[index]
        if self._check_if_annotation_is_valid(raw_info):
            return raw_info
        
        else:
            raise ValueError(f"Annotation info at index {index} is invalid. \
                Please overwrite this function to generate annotation info that follows the standard annotation structure as illustrated in \
                    XXXX.")
    
    def _get_pts_info(self, info):
        """Get point cloud data info from the given info. 
        """
        pts_filenames = []
        pts_sensors = []
        lidar_poses_in_world = []
        for sensor in self.lidar_sensors:
            if 'LIDAR' in sensor and sensor in info['sensors']:
                pts_datapath = info['sensors'][sensor]['data_path']
                pts_filenames.append(osp.join(self.data_root, pts_datapath))
                pts_sensors.append(sensor)
                
                # lidar to world 
                lidar_poses_in_world.append(info['sensors'][sensor]['sensor2world'])
        
        assert len(pts_sensors) <= 1, "Only one lidar sensor is supported."
        
        return pts_filenames[0], pts_sensors[0], lidar_poses_in_world[0]
    
    def _get_imgs_info(self, info):
        img_filenames = []
        img_sensors = []
        cam_intrinsics = []
        cam_poses_in_world = []
        for sensor in self.camera_sensors:
            if 'CAM' in sensor and sensor in info['sensors']:
                img_datapath = info['sensors'][sensor]['data_path']
                img_filenames.append(osp.join(self.data_root, img_datapath))
                cam_intrinsics.append(info['sensors'][sensor]['intrinsic'])
                cam_poses_in_world.append(info['sensors'][sensor]['sensor2world'])
                
                img_sensors.append(sensor)
                
        return img_filenames, img_sensors, cam_intrinsics, cam_poses_in_world
    
    def _get_ann_info(self, info):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_instances_names (list[str]): Class names of ground truths.
        """
        ## Construct mmdet3d Box objects
        # -------------------------------------------------------------------

        gt_bboxes_mask = info.pop('gt_bboxes_mask')
        gt_bboxes_3d = info.pop('gt_bboxes_3d')
        gt_instances_names = info.pop('gt_instances_names')
        gt_instances_ids = info.pop('gt_instances_ids')
        gt_instances2world = info.pop('gt_instances2world')
        
        gt_labels_3d = []
        for cat in gt_instances_names:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1) # if not in the class list, set as -1
        gt_labels_3d = np.array(gt_labels_3d)

        # add velocity to gt_bboxes_3d
        # from (N, 7) to (N, 9)
        if self.with_velocity:
            gt_velocities = info.get('gt_instances_velocities', None)
            if gt_velocities is None:
                gt_velocities = np.zeros_like((gt_bboxes_3d.shape[0], 2))
                
            # fill nan with 0
            nan_mask = np.isnan(gt_velocities).any(axis=1)
            gt_velocities[nan_mask] = 0
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocities], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        if self.box_type_3d_original.lower() == 'depth':
            BoxInstance = DepthInstance3DBoxes
        elif self.box_type_3d_original.lower() == 'lidar':
            BoxInstance = LiDARInstance3DBoxes
        elif self.box_type_3d_original.lower() == 'camera':
            BoxInstance = CameraInstance3DBoxes
        else:
            raise ValueError(f"Unknown box type {self.box_type_3d_original}")
        
        gt_bboxes_3d = BoxInstance(#LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=info['box_center']).convert_to(self.box_mode_3d)
        
        # planning annotations
        gt_instances_traj = info.pop('gt_instances_traj')
        gt_ego_traj = info.pop('gt_ego_traj')
        
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_bboxes_mask=gt_bboxes_mask,
            gt_instances_names=gt_instances_names,
            gt_instances_ids=gt_instances_ids,
            gt_instances2world=gt_instances2world,
            gt_instances_traj=gt_instances_traj,
            gt_ego_traj=gt_ego_traj)
        
        return anns_results
    
    def _get_map_info(self, info):
        """Get map data info from the given info. 
        """
        pass
    
    def get_data_info(self, info):
        """Get data info according to the given index.
        - Points data
        - Multiview image data
        - Annotation data
            
        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:
                
                - pts_filename (list[str]): Filenames of point clouds if more than one lidar.
                - imgs_filename (list[str]): Filenames of images if more than one camera.
                - data_sample (DataSample): Annotation info.
        """
        input_dict = {}
        
        # get lidar data
        if self.modality and self.modality.get('use_lidar', False) and self.lidar_sensors:
            pts_filenames, pts_sensors, lidar_poses = self._get_pts_info(info)
            input_dict['pts_filename'] = pts_filenames
            input_dict['pts_sensor_name'] = pts_sensors
            input_dict['lidar2world'] = lidar_poses
            
        # get camera data file
        if self.modality and self.modality.get('use_camera', False) and self.camera_sensors:
            img_filenames, img_sensors, cam_intrinsics, cam_poses_in_world = self._get_imgs_info(info)
            input_dict['img_filename'] = img_filenames       
            input_dict['img_sensor_name'] = img_sensors
            input_dict['cam_intrinsics'] = cam_intrinsics
            input_dict['cam2world'] = cam_poses_in_world
                    
        # get ego and sensor info
        input_dict['ego'] = info['ego']
        input_dict['sensors'] = info['sensors']
        
        # get annotation info
        anno_info = self._get_ann_info(info)
        
        # save to standard info
        input_dict['anno_info'] = anno_info

        return input_dict

    def generate_past_future_info(self, index, curr_info):
        """Generate past/future annotation info, such as future trajectory.

            The coordinate system is in the local lidar coord at the current frame.
        """
        # make sure the required keys are in the info
        #required_keys = []
        #for key in required_keys:
        #    if key not in info:
        #        raise ValueError(f"Key {key} is required in the info.")
    
        # make sure the required attribues are set
        
        # generate ego past/future trajectory
        past_future_ego_traj = self._generate_past_future_ego_trajectory(index, curr_info)

        # generate instances past/future trajectory
        past_future_instances_traj = self._generate_past_future_instances_trajectory(index, curr_info)

        # add to the current info
        curr_info['gt_ego_traj'] = past_future_ego_traj
        curr_info['gt_instances_traj'] = past_future_instances_traj

        return curr_info
        
    def _generate_past_future_ego_trajectory(self, index, curr_info):
        """Generate past and future trajectories for ego vehicle, offset from the current frame.

        Args:
            index (_type_): _description_
            info (_type_): _description_
        
        Returns:
            TrajectoryData: Trajectory data for ego vehicle, with a length of (past_steps + 1 + planning_steps)
        """

        index_list = range(index - self.past_steps * self.sample_interval, index + self.planning_steps * self.sample_interval + 1, self.sample_interval)
        world2lidar_curr = np.linalg.inv(curr_info['sensors']['LIDAR_TOP']['sensor2world'])
        xyr = np.zeros((self.past_steps + 1 + self.planning_steps, 3)) # past + current + future
        mask = np.zeros((self.past_steps + 1 + self.planning_steps,)) 

        # current frame: 0 centered
        
        # past/future frames
        for i, idx in enumerate(index_list):
            # skip the current frame
            if idx == index:
                mask[i] = 1
                continue
            # check if index is within range
            if idx < 0 or idx >= self.num_samples:
                break 
            # check if the the frames are from the same scene
            adj_info = self.prepare_planning_info(idx)
            if curr_info['scene_token'] != adj_info['scene_token']:
                break
            
            world2lidar_adj = np.linalg.inv(adj_info['sensors']['LIDAR_TOP']['sensor2world'])
            # T12 = T2^-1 * T1
            adj2curr = world2lidar_curr @ np.linalg.inv(world2lidar_adj)
            xyr[i, :2] = adj2curr[:2, 3]
            xyr[i, 2] = np.arctan2(adj2curr[1, 0], adj2curr[0, 0]) # [-pi, pi]
            mask[i] = 1
            
        return TrajectoryData(metainfo=dict(num_past_steps=self.past_steps, 
                                            num_future_steps=self.planning_steps,
                                            time_step=self.sample_interval/self.FPS), 
                              data=xyr.astype(np.float32), 
                              mask=mask.astype(np.uint8))
            
    def _generate_past_future_instances_trajectory(self, index, curr_info):
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
        instances_ids = curr_info['gt_instances_ids']
        world2lidar_curr = np.linalg.inv(curr_info['sensors']['LIDAR_TOP']['sensor2world'])
        
        # initialize the trajectory data
        trajs = []
                
        # for each instance in the current frame, find its past and future trajectory
        for i, instance_id in enumerate(instances_ids):
            # TODO: should use accumulative points for the trajectory. if no more data, use the last point
            xyr = np.zeros((self.past_steps + 1 + self.planning_steps, 3)) # (T, 3)
            mask = np.zeros((self.past_steps + 1 + self.planning_steps,)) # (T,)    
            
            # 
            instance2lidar_curr = world2lidar_curr @ curr_info['gt_instances2world'][i]
            xyr[self.past_steps, :2] = instance2lidar_curr[:2, 3]
            xyr[self.past_steps, 2] = np.arctan2(instance2lidar_curr[1, 0], instance2lidar_curr[0, 0]) # [-pi, pi]
            
            for j, idx in enumerate(index_list):
                # skip the current frame
                if idx == index:
                    mask[j] = 1
                    continue
                
                # check if index is within range
                if idx < 0 or idx >= self.num_samples:
                    break
                # check if the the frames are from the same scene
                adj_info = self.prepare_planning_info(idx)
                if curr_info['scene_token'] != adj_info['scene_token']:
                    break
                # instance not found in the adjacent frame
                if instance_id not in adj_info['gt_instances_ids']:
                    continue
                # box index of the instance in the adjacent frame
                adj_idx = np.where(adj_info['gt_instances_ids'] == instance_id)[0][0]
                
                # these two should be the same
                #instance2lidar_adj = adj_info['sensors']['LIDAR_TOP']['world2sensor'] @ adj_info['gt_instance2world'][adj_idx]
                #adj2curr = instance2lidar_curr @ np.linalg.inv(instance2lidar_adj)
                # viewing instance in adj frame lidar coords from the current frame's lidar coord
                adj2curr = world2lidar_curr @ adj_info['gt_instances2world'][adj_idx]

                xyr[j, :2] = adj2curr[:2, 3]
                xyr[j, 2] = np.arctan2(adj2curr[1, 0], adj2curr[0, 0]) # [-pi, pi]
                mask[j] = 1
        
            trajs.append(TrajectoryData(metainfo=dict(num_past_steps=self.past_steps, 
                                                num_future_steps=self.planning_steps,
                                                time_step=self.sample_interval/self.FPS), 
                                    data=xyr.astype(np.float32),
                                    mask=mask.astype(np.uint8))
        )
        return trajs
                
    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.
                - img_fields (list[str]): Image fields, inlcuding 'img', 'img_filename', etc
                - pts_fields (list[str]): Point fields, including 'pts', 'pts_filename', etc
                - ego_fields (list[str]): Ego fields, including 
                        'gt_ego_traj', 'ego_world2ego', 'ego_velocity', 'ego_affected_by_lights', 'ego_affected_by_stop_sign', 'ego_is_at_junction', etc
                - bbox3d_fields (list[str]): 3D bbox fields, including 'gt_bboxes_3d', 'gt_labels_3d', 'gt_classes', etc
                - pts_seg_fields (list[str]): Point cloud segmentation fields.
                - grid_fields (list[str]): Grid fields, including "gt_grid_density", "gt_grid_occupancy", etc
        """
        
        results['img_fields'] = []
        results['pts_fields'] = []
        results['ego_fields'] = [] # ['gt_ego_traj']
        results['bbox3d_fields'] = []
        results['pts_seg_fields'] = []
        results['grid_fields'] = [] 
        results['bbox_fields'] = []
        results['img_seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        info = self.prepare_planning_info(index)
        # add past/future annotation info, such as future trajectory
        info = self.generate_past_future_info(index, info) 
        # assemble for data pipeline
        input_dict = self.get_data_info(info)
        if not input_dict:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['data_samples'].gt_instances.labels != -1).any()):
            return None
        return example

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        info = self.prepare_planning_info(index)
        # add past/future annotation info, such as future trajectory
        info = self.generate_past_future_info(index, info) 
        # assemble for data pipeline
        input_dict = self.get_data_info(info)
        if not input_dict:
            return None

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['data_samples'].instances.gt_labels != -1).any()):
            return None
        
        return example

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results, \
                tmp_dir is the temporal directory created for saving json \
                files when ``jsonfile_prefix`` is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
            out = f'{pklfile_prefix}.pkl'
        dump(outputs, out)
        return outputs, tmp_dir

    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str]): Metrics to be evaluated.
            iou_thr (list[float]): AP IoU thresholds.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        from mmcv.core.evaluation import indoor_eval
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'
        gt_annos = [info['annos'] for info in self.data_infos]
        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}
        ret_dict = indoor_eval(
            gt_annos,
            results,
            iou_thr,
            label2cat,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d)
        if show:
            self.show(results, out_dir, pipeline=pipeline)

        return ret_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        raise NotImplementedError('_build_default_pipeline is not implemented '
                                  f'for dataset {self.__class__.__name__}')

    def _get_pipeline(self, pipeline):
        """Get data loading pipeline in self.show/evaluate function.

        Args:
            pipeline (list[dict] | None): Input pipeline. If None is given, \
                get from self.pipeline.
        """
        if pipeline is None:
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                warnings.warn(
                    'Use default pipeline for data loading, this may cause '
                    'errors when data is on ceph')
                return self._build_default_pipeline()
            loading_pipeline = get_loading_pipeline(self.pipeline.transforms)
            return Compose(loading_pipeline)
        return Compose(pipeline)

    def _extract_data(self, index, pipeline, key, load_annos=False):
        """Load data using input pipeline and extract data according to key.

        Args:
            index (int): Index for accessing the target data.
            pipeline (:obj:`Compose`): Composed data loading pipeline.
            key (str | list[str]): One single or a list of data key.
            load_annos (bool): Whether to load data annotations.
                If True, need to set self.test_mode as False before loading.

        Returns:
            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
                A single or a list of loaded data.
        """
        assert pipeline is not None, 'data loading pipeline is not provided'
        # when we want to load ground-truth via pipeline (e.g. bbox, seg mask)
        # we need to set self.test_mode as False so that we have 'annos'
        if load_annos:
            original_test_mode = self.test_mode
            self.test_mode = False
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = pipeline(input_dict)

        # extract data items according to keys
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]
        if load_annos:
            self.test_mode = original_test_mode

        return data

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.data_infos)

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
