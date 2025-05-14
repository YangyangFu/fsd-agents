import os
import os.path as osp
import torch
import numpy as np
import laspy
import pycocotools.mask as maskUtils

from mmengine.fileio import FileClient
from mmengine import check_file_exist
from mmcv.image import imfrombytes, imread
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmdet3d.structures.points import BasePoints, get_points_type
# from mmcv.datasets.pipelines.loading import LoadAnnotations, LoadImageFromFile
from fsd.registry import TRANSFORMS as PIPELINES

@PIPELINES.register_module()
class LoadAnnotations3D(object):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_name_3d=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox_depth=False,
                 seg_3d_dtype='int',):
        super().__init__()
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_name_3d = with_name_3d
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmcv3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmcv3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d'].copy()
        results['depths'] = results['ann_info']['depths']
        results['bbox3d_fields'].extend(['centers2d', 'depths'])
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmcv3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d'].copy()
        results['bbox3d_fields'].append('gt_labels_3d')
        return results

    def _load_instances_names(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmcv3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_instances_name'] = results['ann_info']['gt_instances_names'].copy()
        results['bbox3d_fields'].append('gt_instances_name')
        
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmcv3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['anno_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int)
        except ConnectionError:
            check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.long)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmcv3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.long)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmcv3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_name_3d:
            results = self._load_instances_names(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_name_3d={self.with_name_3d}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        return repr_str

@PIPELINES.register_module()
class LoadAnnotationsPlan3D(LoadAnnotations3D):
    """Load Annotations3D for planning tasks.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
    # Planning:
        with_instances_traj (bool, optional): Whether to load trajectory for instances.
            Defaults to False.
        
    # 3D annotations inherited from LoadAnnotations3D
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
       
    # 2D annotations
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """
    def __init__(self,
                 with_ego_traj=False,
                 with_instances_traj=False,
                 with_instances_ids=False,
                 with_grids=False,
                 with_vector_map=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.with_ego_traj = with_ego_traj
        self.with_instances_traj = with_instances_traj
        self.with_instances_ids = with_instances_ids
        self.with_grids = with_grids
        self.with_vector_map = with_vector_map
        
    def _load_instances_ids(self, results):
        ann_gt_inds = results['ann_info']['gt_bboxes_id'].copy() 
        results['gt_bboxes_id'] = ann_gt_inds
        results['bbox3d_fields'].append('gt_bboxes_id')
        return results

    def _load_ego_traj(self, results):
        ego_traj = results['ann_info']['gt_ego_traj']
        ego_traj_mask = results['ann_info'].get('gt_ego_traj_mask', None)
        results['gt_ego_traj'] = ego_traj
        if ego_traj_mask is not None:
            results['gt_ego_traj_mask'] = ego_traj_mask
            results['ego_fields'].append('gt_ego_traj_mask')
        results['ego_fields'].append('gt_ego_traj')
        return results
    
    def _load_instances_traj(self, results):
        instances_traj = results['ann_info']['gt_bboxes_traj']
        instances_traj_mask = results['ann_info'].get('gt_bboxes_traj_mask', None)
        results['gt_bboxes_traj'] = instances_traj
        if instances_traj_mask is not None:
            results['gt_bboxes_traj_mask'] = instances_traj_mask
            results['bbox3d_fields'].append('gt_bboxes_traj_mask')
        results['bbox3d_fields'].append('gt_bboxes_traj')
        return results
    
    def _load_grids(self, results):
        """Load all types of grids if available in the dataset.

        Args:
            results (_type_): _description_
        """
        grids = ['gt_grid_density', 'gt_grid_occupancy']
        for grid in grids:
            if grid in results['anno_info']:
                results[grid] = results['anno_info'][grid]
                results['map_fields'].append(grid)
        return results
    
    def _load_vector_map(self, results):
        """Load vector map if available in the dataset.

        Args:
            results (_type_): _description_
        """
        keys = ['gt_map_vectors', 'gt_map_vectors_pt', 'gt_map_vectors_label']
        for key in keys:
            if key in results['ann_info']:
                results[key] = results['ann_info'][key]
                results['map_fields'].append(key)
            
        return results
    
    def __call__(self, results):
        results = super().__call__(results)
        
        # load ego related
        if self.with_ego_traj:
            results = self._load_ego_traj(results)
   
        # load future trajectory instances
        if self.with_instances_traj:
            results = self._load_instances_traj(results)
        if self.with_instances_ids:
            results = self._load_instances_ids(results)

        # load grids
        if self.with_grids:
            results = self._load_grids(results)
        
        # load vector map
        if self.with_vector_map:
            results = self._load_vector_map(results)
        
        return results

    def __repr__(self):
        repr_str = super().__repr__()
        indent_str = '    '
        repr_str += f'{indent_str}with_instances_traj={self.with_instances_traj}, '
        repr_str += f'{indent_str}with_instances_ids={self.with_instances_ids}, '
        repr_str += f'{indent_str}with_grids={self.with_grids})'
        
        return repr_str


def load_augmented_point_cloud(path, virtual=False, reduce_beams=32):
    # NOTE: following Tianwei's implementation, it is hard coded for nuScenes
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    # NOTE: path definition different from Tianwei's implementation.
    tokens = path.split("/")
    vp_dir = "_VIRTUAL" if reduce_beams == 32 else f"_VIRTUAL_{reduce_beams}BEAMS"
    seg_path = os.path.join(
        *tokens[:-3],
        "virtual_points",
        tokens[-3],
        tokens[-2] + vp_dir,
        tokens[-1] + ".pkl.npy",
    )
    assert os.path.exists(seg_path)
    data_dict = np.load(seg_path, allow_pickle=True).item()

    virtual_points1 = data_dict["real_points"]
    # NOTE: add zero reflectance to virtual points instead of removing them from real points
    virtual_points2 = np.concatenate(
        [
            data_dict["virtual_points"][:, :3],
            np.zeros([data_dict["virtual_points"].shape[0], 1]),
            data_dict["virtual_points"][:, 3:],
        ],
        axis=-1,
    )

    points = np.concatenate(
        [
            points,
            np.ones([points.shape[0], virtual_points1.shape[1] - points.shape[1] + 1]),
        ],
        axis=1,
    )
    virtual_points1 = np.concatenate(
        [virtual_points1, np.zeros([virtual_points1.shape[0], 1])], axis=1
    )
    # note: this part is different from Tianwei's implementation, we don't have duplicate foreground real points.
    if len(data_dict["real_points_indice"]) > 0:
        points[data_dict["real_points_indice"]] = virtual_points1
    if virtual:
        virtual_points2 = np.concatenate(
            [virtual_points2, -1 * np.ones([virtual_points2.shape[0], 1])], axis=1
        )
        points = np.concatenate([points, virtual_points2], axis=0).astype(np.float32)
    return points


def reduce_LiDAR_beams(pts, reduce_beams_to=32):
    # print(pts.size())
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    radius = torch.sqrt(pts[:, 0].pow(2) + pts[:, 1].pow(2) + pts[:, 2].pow(2))
    sine_theta = pts[:, 2] / radius
    # [-pi/2, pi/2]
    theta = torch.asin(sine_theta)
    phi = torch.atan2(pts[:, 1], pts[:, 0])

    top_ang = 0.1862
    down_ang = -0.5353

    beam_range = torch.zeros(32)
    beam_range[0] = top_ang
    beam_range[31] = down_ang

    for i in range(1, 31):
        beam_range[i] = beam_range[i - 1] - 0.023275
    # beam_range = [1, 0.18, 0.15, 0.13, 0.11, 0.085, 0.065, 0.03, 0.01, -0.01, -0.03, -0.055, -0.08, -0.105, -0.13, -0.155, -0.18, -0.205, -0.228, -0.251, -0.275,
    #                -0.295, -0.32, -0.34, -0.36, -0.38, -0.40, -0.425, -0.45, -0.47, -0.49, -0.52, -0.54]

    num_pts, _ = pts.size()
    mask = torch.zeros(num_pts)
    if reduce_beams_to == 16:
        for id in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
            beam_mask = (theta < (beam_range[id - 1] - 0.012)) * (
                theta > (beam_range[id] - 0.012)
            )
            mask = mask + beam_mask
        mask = mask.bool()
    elif reduce_beams_to == 4:
        for id in [7, 9, 11, 13]:
            beam_mask = (theta < (beam_range[id - 1] - 0.012)) * (
                theta > (beam_range[id] - 0.012)
            )
            mask = mask + beam_mask
        mask = mask.bool()
    # [?] pick the 14th beam
    elif reduce_beams_to == 1:
        chosen_beam_id = 9
        mask = (theta < (beam_range[chosen_beam_id - 1] - 0.012)) * (
            theta > (beam_range[chosen_beam_id] - 0.012)
        )
    else:
        raise NotImplementedError
    # points = copy.copy(pts)
    points = pts[mask]
    # print(points.size())
    return points.numpy()

@PIPELINES.register_module()
class LoadPointsFromMultiSweeps:
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results["pts"]
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"] / 1e6
        if self.pad_empty_sweeps and len(results["sweeps"]) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results["sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["sweeps"]))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                # NOTE: seems possible to load frame -11?
                if not self.load_augmented:
                    choices = np.random.choice(
                        len(results["sweeps"]), self.sweeps_num, replace=False
                    )
                else:
                    # don't allow to sample the earliest frame, match with Tianwei's implementation.
                    choices = np.random.choice(
                        len(results["sweeps"]) - 1, self.sweeps_num, replace=False
                    )
            for idx in choices:
                sweep = results["sweeps"][idx]
                points_sweep = self._load_points(sweep["data_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                # TODO: make it more general
                if self.reduce_beams and self.reduce_beams < 32:
                    points_sweep = reduce_LiDAR_beams(points_sweep, self.reduce_beams)

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep["timestamp"] / 1e6
                points_sweep[:, :3] = (
                    points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
                )
                points_sweep[:, :3] += sweep["sensor2lidar_translation"]
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results["pts"] = points
        results['pts_fileds'].append('pts')
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"

@PIPELINES.register_module()
class LoadPointsFromFileCarlaDataset:
    """Load Points From File used for carla dataset only.

    CarlaDataset save points data in .laz file, and 
    the points are in left-hand UE coordinates(e.g., x forward, y right, z up), and 
    the points are saved in ego coordinates.
    Here we load the points and convert them to right-hand MMDET3D lidar coordinates for consistency.
    
    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        load_augmented=None,
        reduce_beams=None,
        to_float32=True,
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams
        self.to_float32 = to_float32
        
    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        elif lidar_path.endswith(".laz"):
            assert self.load_dim <= 4, "Only support loading xyz and intensity in laz for now"
            with open(lidar_path, "rb") as f:
                las = laspy.read(f)
                if self.load_dim == 4:
                    points = np.vstack([las.x, las.y, las.z, las.intensity]).T
                elif self.load_dim == 3:
                    points = np.vstack([las.x, las.y, las.z]).T
                else: 
                    raise NotImplementedError
        else:
            points = np.fromfile(lidar_path)

        if self.to_float32:
            points = points.astype(np.float32)
        
        return points

    def __call__(self, results):
        """Call function to load points data from file,
        and convert them to right-hand mmdet3d lidar coordinates

        Carladataset left-hand ego coordinates: x-front, y-right, z-up
        MMDET3D right-hand lidar coordinates: x-front, y-left, z-up
        
        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        assert self.coord_type == "LIDAR", "Only support LIDAR type for now"
        
        lidar_path = results["pts_filename"]
        lidar_name = results["pts_sensor_name"]
        points = self._load_points(lidar_path)
        points = points.reshape(-1, self.load_dim)
        
        # This assumes the points are originally in lefthand system
        # and annotated sensor2ego is mmdet sensor to mmdet ego
        # convert from left-hand ego coord to right-hand mmdet3d lidar coord
        left2right = np.eye(4)
        left2right[1, 1] = -1
        # mmdet lidar coord to mmdet ego coord
        lidar2ego = results['sensors'][lidar_name]['sensor2ego']
        points_hom = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        
        # convert to mmdet3d lidar coord: ego2lidar_mmdet @ lefthand_ego2mmdet_ego
        points = (np.linalg.inv(lidar2ego) @ left2right @ points_hom.T).T 
        points = points[:, :3]
        
        # TODO: make it more general
        if self.reduce_beams and self.reduce_beams < 32:
            points = reduce_LiDAR_beams(points, self.reduce_beams)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )
        results["pts"] = points
        results['pts_num_features'] = points.points_dim
        results['pts_fields'].extend(['pts', 'pts_num_features'])

        return results
