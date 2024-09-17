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
                 index_front_camera: Optional[int] = 0):
        vis = PlanningVisualizer.get_instance(name='vis')
        self._visualizer: PlanningVisualizer = PlanningVisualizer.get_current_instance()
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
        
        # index of front camera in the multi-view data
        # draw traj on this image
        self.index_front_camera = index_front_camera
        
    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[PlanningDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        data_input = dict()

        # Visualize only the first data
        if self.vis_task in [
                'mono_det', 'multi-view_det', 'multi-modality_det', 'multi-modality_planning'
        ]:
            assert 'img_path' in outputs[0], 'img_path is not in outputs[0]'
            img_path = outputs[0].img_path
            if isinstance(img_path, list):
                img = []
                for single_img_path in img_path:
                    img_bytes = get(
                        single_img_path, backend_args=self.backend_args)
                    single_img = mmcv.imfrombytes(
                        img_bytes, channel_order='rgb')
                    img.append(single_img)
            else:
                img_bytes = get(img_path, backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            data_input['img'] = img

        if self.vis_task in ['lidar_det', 'multi-modality_det', 'multi-modality_planning', 'lidar_seg']:
            assert 'lidar_path' in outputs[
                0], 'lidar_path is not in outputs[0]'
            lidar_path = outputs[0].lidar_path
            num_pts_feats = outputs[0].num_pts_feats
            pts_bytes = get(lidar_path, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
            points = points.reshape(-1, num_pts_feats)
            data_input['pts'] = points

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                'val sample',
                data_input,
                data_sample=outputs[0],
                draw_gt=self.draw_gt,
                draw_pred=self.draw_pred,
                show=self.show,
                vis_task=self.vis_task,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                step=total_curr_iter,
                show_pcd_rgb=self.show_pcd_rgb)

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

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx
        
        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)

        # add lidar2img to data_sample
        data_inputs = data_batch['inputs']
        for b, data_sample in enumerate(outputs):
            self._test_index += 1

            data_input = dict()
            # load original images and pts
            # inputs from data_batch are from data pipeline, which may have been reshaped.
            if self.vis_task in [
                    'mono_det', 'multi-view_det', 'multi-modality_det', 'multi-modality_planning'
            ]:
                assert 'img_filename' in data_batch['inputs']['img_metas'], "image path is not in data_batch['inputs']"
                img_path = [fb[b] for fb in data_batch['inputs']['img_metas']['img_filename']]
                
                if isinstance(img_path, list):
                    img = []
                    for single_img_path in img_path:
                        img_bytes = get(
                            single_img_path, backend_args=self.backend_args)
                        single_img = mmcv.imfrombytes(
                            img_bytes, channel_order='rgb')
                        img.append(torch.from_numpy(single_img).permute(2, 0, 1))
                else:
                    img_bytes = get(img_path, backend_args=self.backend_args)
                    img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                    img = torch.from_numpy(img).permute(2, 0, 1)
                    
                data_input['img'] = img

            #TODO: need load pts from file instead of from data_batch
            # load pts in Lidar coord
            if self.vis_task in ['lidar_det', 'multi-modality_det', 'multi-modality_planning', 'lidar_seg']:
                assert 'pts_filename' in data_batch['inputs']['pts_metas'], 'lidar_path is not in outputs[0]'
                points = data_batch['inputs']['pts'][b]
                data_input['pts'] = points
                
            # save dirs
            out_file = o3d_save_path = None

            if total_curr_iter % self.interval == 0:
                # get lidar2img transform
                cams2world = [pose[b] for pose in data_inputs['img_metas']['cam2world']]
                cams_intrinsics = [intrinsic[b] for intrinsic in data_inputs['img_metas']['cam_intrinsics']]
                cams_intrinsics = [np.pad(cam, (0, 1), constant_values=0) for cam in cams_intrinsics]
                lidar2world = data_inputs['pts_metas']['lidar2world'][b]
                lidar2imgs = [cam_intrinsic @ np.linalg.inv(cam2world) @ lidar2world for cam_intrinsic, cam2world in zip(cams_intrinsics, cams2world)]
                data_sample.set_metainfo(dict(lidar2img=lidar2imgs))
                
                # to cpu
                data_sample = data_sample.to('cpu')
                
                # visualizer
                self._visualizer.add_datasample(
                    'test sample',
                    data_input,
                    data_sample=data_sample,
                    draw_gt=self.draw_gt,
                    draw_pred=self.draw_pred,
                    show=self.show,
                    vis_task=self.vis_task,
                    wait_time=self.wait_time,
                    pred_score_thr=self.score_thr,
                    out_file=self.test_out_dir,
                    o3d_save_path=self.test_out_dir,
                    step=self._test_index,
                    show_pcd_rgb=self.show_pcd_rgb,
                    traj_img_idx=self.index_front_camera)

            # first only
            if self.view_first_only:
                break
