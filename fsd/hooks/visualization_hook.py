# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence
import torch

import mmcv
import numpy as np
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer

from fsd.registry import HOOKS
from fsd.structures import PlanningDataSample
from fsd.visualization import PlanningVisualizer


@HOOKS.register_module()
class PlanningVisualizationHook(Hook):
    """Planning Visualization Hook. Used to visualize validation and testing
    process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``test_out_dir`` is specified, it means that the prediction results
        need to be saved to ``test_out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `test_out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        vis_task (str): Visualization task. Defaults to 'mono_det'.
        wait_time (float): The interval of show (s). Defaults to 0.
        draw_gt (bool): Whether to draw ground truth. Defaults to True.
        draw_pred (bool): Whether to draw prediction. Defaults to True.
        show_pcd_rgb (bool): Whether to show RGB point cloud. Defaults to
            False.
        test_out_dir (str, optional): directory where painted images
            will be saved in testing process.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 score_thr: float = 0.3,
                 show: bool = False,
                 vis_task: str = 'mono_det',
                 wait_time: float = 0.,
                 test_out_dir: Optional[str] = None,
                 draw_gt: bool = False,
                 draw_pred: bool = True,
                 show_pcd_rgb: bool = False,
                 backend_args: Optional[dict] = None,
                 view_first_only: Optional[bool] = True,
                 image_mode: Optional[str] = 'bgr',
                 multi_view_names: Optional[Sequence[str]] = None,
                 point_cloud_range: Optional[Sequence[float]] = None,
                 pixels_per_meter: Optional[float] = 10,
                 map_format: Optional[str] = 'fixed_num_pts',
                 ) -> None:
        """Initialize the visualization hook.
        Args:
            draw (bool): Whether to draw prediction results. If it is False,
                it means that no drawing will be done. Defaults to False.
            interval (int): The interval of visualization. Defaults to 50.
            score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            show (bool): Whether to display the drawn image. Default to False.
            vis_task (str): Visualization task. Defaults to 'mono_det'.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_gt (bool): Whether to draw ground truth. Defaults to True.
            draw_pred (bool): Whether to draw prediction. Defaults to True.
            show_pcd_rgb (bool): Whether to show RGB point cloud. Defaults to
                False.
            test_out_dir (str, optional): directory where painted images
                will be saved in testing process.
            backend_args (dict, optional): Arguments to instantiate the
                corresponding backend. Defaults to None.
            view_first_only (bool): Whether to only visualize the first
                sample in the batch. Defaults to True.
            image_mode (str): The image mode. Defaults to 'bgr'. Options are
                'bgr' and 'rgb'.
            multi_view_names (Sequence[str]): The names of the multi-view
                images. Defaults to None. 
            pixels_per_meter (float): The pixels per meter for the map.
                Defaults to 10.
            map_format (str): The format of the map. Defaults to 'fixed_num_pts'.
                Options are 'fixed_num_pts' and 'polyline'.
            
        """

        vis = PlanningVisualizer.get_instance(name='vis')
        self._visualizer: PlanningVisualizer = PlanningVisualizer(
            image_mode=image_mode,
            ).get_current_instance()
        self.interval = interval
        self.score_thr = score_thr
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')
        self.vis_task = vis_task

        if show and wait_time == -1:
            print_log(
                'Manual control mode, press [Right] to next sample.',
                logger='current')
        elif show:
            print_log(
                'Autoplay mode, press [SPACE] to pause.', logger='current')
        self.wait_time = wait_time
        self.backend_args = backend_args
        self.draw = draw
        self.test_out_dir = test_out_dir
        self._test_index = 0
        self.draw_gt = draw_gt
        self.draw_pred = draw_pred
        self.show_pcd_rgb = show_pcd_rgb
        # only view first data in the batch
        self.view_first_only = view_first_only 
        # the arrangment order of the multi-view images
        self.multi_view_names = multi_view_names
        
        # map visualization
        self.point_cloud_range = point_cloud_range
        self.pixels_per_meter = pixels_per_meter
        self.map_format = map_format
        
    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[PlanningDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.
            Same implementation as after_test_iter.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        pass

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[PlanningDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        # get dataset meta
        dataset_meta = runner.test_dataloader.dataset.metainfo
        self._visualizer.dataset_meta = dataset_meta
        
        # BGR or RGB after the pipeline
        image_mode = self._visualizer.image_mode
        
        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx
        
        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)
        out_file = o3d_save_path = None
            
        # add lidar2img to data_sample
        for b, data_sample in enumerate(outputs):
            self._test_index += 1

            data_input = dict()
            # load original images and pts
            # inputs from data_batch are from data pipeline, which may have been reshaped.
            if self.vis_task in [
                    'mono_det', 'multi-view_det', 'multi-modality_det', 'multi-modality_planning'
            ]:
                assert hasattr(data_sample, 'metainfo') and 'img_path' in data_sample.metainfo, \
                    "image path is not in data_sample.metainfo"
                    
                img_path = [img for img in data_sample.metainfo['img_path']]
                
                if isinstance(img_path, list):
                    img = []
                    for single_img_path in img_path:
                        img_bytes = get(
                            single_img_path, backend_args=self.backend_args)
                        single_img = mmcv.imfrombytes(
                            img_bytes, channel_order=image_mode.lower())
                        img.append(torch.from_numpy(single_img).permute(2, 0, 1))
                else:
                    img_bytes = get(img_path, backend_args=self.backend_args)
                    img = mmcv.imfrombytes(img_bytes, channel_order=image_mode.lower())
                    img = torch.from_numpy(img).permute(2, 0, 1)
                    
                data_input['img'] = img
                # save folder
                if self.test_out_dir is not None:
                    if isinstance(img_path, list):
                        img_path = img_path[0]
                    out_file = osp.basename(img_path)
                    out_file = osp.join(self.test_out_dir, out_file)
                
            # load pts in Lidar coord
            if self.vis_task in ['lidar_det', 'multi-modality_det', 'multi-modality_planning', 'lidar_seg']:
                assert hasattr(data_sample, 'metainfo') and 'lidar_path' in data_sample.metainfo, \
                    'lidar_path is not in data_sample.metainfo'
                lidar_path = data_sample.metainfo['lidar_path']
                
                # CARLA dataset lidar points
                if dataset_meta['name'] == 'carla':
                    from fsd.datasets.transforms import load_points_carla
                    
                    lidar2world = data_sample.pts_metas['lidar2world']
                    ego2world = data_sample.gt_ego.pose.cpu().numpy()
                    lidar2ego = np.linalg.inv(ego2world) @ lidar2world
                    points = load_points_carla(
                        lidar_path = lidar_path, 
                        input_meta = {'lidar2ego': lidar2ego}, 
                        coord_type = 'depth', 
                        num_features = 3, 
                        to_float32 = True
                    )
                elif 'nuscenes' in dataset_meta['name']:
                    # use data from pipeline
                    points = data_batch['inputs']['points'][b]
                    points = points.cpu()
                    # load from file
                    
                    
                else:
                    raise NotImplementedError('Only support CARLA dataset for now')
                
                data_input['points'] = points
                
                # save folder
                if self.test_out_dir is not None:
                    o3d_save_path = osp.basename(lidar_path).split(
                        '.')[0] + '.png'
                    o3d_save_path = osp.join(self.test_out_dir, o3d_save_path)                    
                    

            if total_curr_iter % self.interval == 0:
                # get lidar2img transform
                assert hasattr(data_sample, 'lidar2img'), \
                    'lidar2img is not in data_sample'
                
                # to cpu
                data_sample = data_sample.to('cpu')
                
                # some customized data in nuscenes_vad dataset
                # nuscenes vad dataset follows kitti box convention
                # change it back to mmdet3d box convention before visualization
                if 'nuscenes-vad' in dataset_meta['name']:
                    data_sample = self._change_kitti_box_to_mmdet3d(data_sample)
                
                # visualizer
                self._visualizer.add_datasample(
                    'test',
                    data_input,
                    data_sample=data_sample,
                    draw_gt=self.draw_gt,
                    draw_pred=self.draw_pred,
                    show=self.show,
                    vis_task=self.vis_task,
                    wait_time=self.wait_time,
                    pred_score_thr=self.score_thr,
                    out_file=out_file,
                    o3d_save_path=o3d_save_path,
                    step=self._test_index,
                    show_pcd_rgb=self.show_pcd_rgb,
                    multi_view_names=self.multi_view_names,
                    pcd_range = self.point_cloud_range,
                    map_format = self.map_format,
                    pixels_per_meter = self.pixels_per_meter,
                    to_mmdet3d_lidar = runner.test_dataloader.dataset.to_mmdet3d_lidar,
                )

            # first only
            if self.view_first_only:
                break
    
    def _change_kitti_box_to_mmdet3d(self, data_sample: PlanningDataSample):
        """Change the box format from kitti to mmdet3d."""
        # change the box format from kitti to mmdet3d
        # kitti: [x, y, z, w, l, h, yaw]
        # mmdet3d: [x, y, z, l, w, h, -yaww-pi/2]
        
        if data_sample.gt_instances_3d is None or data_sample.gt_instances_3d.bbox is None:
            return
        
        bbox = data_sample.gt_instances_3d.bbox
        data = bbox.tensor.clone()
        data[:, 3:6] = data[:, [4, 3, 5]]
        data[:, 6] = -data[:, 6] - np.pi / 2

        bbox_new = bbox.new_box(data = data)
        
        data_sample.gt_instances_3d.bbox = bbox_new
        
        return data_sample