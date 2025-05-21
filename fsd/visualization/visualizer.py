# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import os
import sys
import time
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import cv2
import mmcv
import numpy as np
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.colors as mcolors
from mmdet.visualization import get_palette
from mmengine.dist import master_only
from mmengine.logging import print_log
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer as MMENGINE_Visualizer
from mmengine.visualization.utils import (check_type, color_val_matplotlib,
                                      tensor2ndarray, wait_continue)
import torch
from torch import Tensor

from mmdet3d.structures import (BaseInstance3DBoxes, Box3DMode,
                                CameraInstance3DBoxes, Coord3DMode,
                                DepthInstance3DBoxes, DepthPoints,
                                Det3DDataSample, LiDARInstance3DBoxes,
                                PointData, points_cam2img)
from .vis_utils import (proj_camera_bbox3d_to_img, proj_depth_bbox3d_to_img,
                        proj_lidar_bbox3d_to_img, to_depth_mode)

try:
    import open3d as o3d
    from open3d import geometry
    from open3d.visualization import Visualizer
except ImportError:
    o3d = geometry = Visualizer = None

from fsd.structures import Trajectory
from fsd.registry import VISUALIZERS

@VISUALIZERS.register_module()
class PlanningVisualizer(MMENGINE_Visualizer):
    """Planning Visualizer.

    - 3D detection and segmentation drawing methods

      - draw_bboxes_3d: draw 3D bounding boxes on point clouds
      - draw_proj_bboxes_3d: draw projected 3D bounding boxes on image
      - draw_seg_mask: draw segmentation mask via per-point colorization
      - draw_

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        points (np.ndarray, optional): Points to visualize with shape (N, 3+C).
            Defaults to None.
        image (np.ndarray, optional): The origin image to draw. The format
            should be RGB. Defaults to None.
        pcd_mode (int): The point cloud mode (coordinates): 0 represents LiDAR,
            1 represents CAMERA, 2 represents Depth. Defaults to 0.
        vis_backends (List[dict], optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
            Defaults to None.
        bbox_color (str or Tuple[int], optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to None.
        text_color (str or Tuple[int]): Color of texts. The tuple of color
            should be in BGR order. Defaults to (200, 200, 200).
        mask_color (str or Tuple[int], optional): Color of masks. The tuple of
            color should be in BGR order. Defaults to None.
        line_width (int or float): The linewidth of lines. Defaults to 3.
        frame_cfg (dict): The coordinate frame config while Open3D
            visualization initialization.
            Defaults to dict(size=1, origin=[0, 0, 0]).
        alpha (int or float): The transparency of bboxes or mask.
            Defaults to 0.8.
        multi_imgs_col (int): The number of columns in arrangement when showing
            multi-view images.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from mmengine.structures import InstanceData
        >>> from mmdet3d.structures import (DepthInstance3DBoxes
        ...                                 Det3DDataSample)
        >>> from mmdet3d.visualization import Det3DLocalVisualizer

        >>> det3d_local_visualizer = Det3DLocalVisualizer()
        >>> image = np.random.randint(0, 256, size=(10, 12, 3)).astype('uint8')
        >>> points = np.random.rand(1000, 3)
        >>> gt_instances_3d = InstanceData()
        >>> gt_instances_3d.bboxes_3d = DepthInstance3DBoxes(
        ...     torch.rand((5, 7)))
        >>> gt_instances_3d.labels_3d = torch.randint(0, 2, (5,))
        >>> gt_det3d_data_sample = Det3DDataSample()
        >>> gt_det3d_data_sample.gt_instances_3d = gt_instances_3d
        >>> data_input = dict(img=image, points=points)
        >>> det3d_local_visualizer.add_datasample('3D Scene', data_input,
        ...                                       gt_det3d_data_sample)

        >>> from mmdet3d.structures import PointData
        >>> det3d_local_visualizer = Det3DLocalVisualizer()
        >>> points = np.random.rand(1000, 3)
        >>> gt_pts_seg = PointData()
        >>> gt_pts_seg.pts_semantic_mask = torch.randint(0, 10, (1000, ))
        >>> gt_det3d_data_sample = Det3DDataSample()
        >>> gt_det3d_data_sample.gt_pts_seg = gt_pts_seg
        >>> data_input = dict(points=points)
        >>> det3d_local_visualizer.add_datasample('3D Scene', data_input,
        ...                                       gt_det3d_data_sample,
        ...                                       vis_task='lidar_seg')
    """

    def __init__(
        self,
        name: str = 'visualizer',
        points: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        image_mode: Optional[str] = 'bgr',
        pcd_mode: int = 0,
        vis_backends: Optional[List[dict]] = None,
        save_dir: Optional[str] = None,
        bbox_color: Optional[Union[str, Tuple[int]]] = None,
        text_color: Union[str, Tuple[int]] = (200, 200, 200),
        mask_color: Optional[Union[str, Tuple[int]]] = None,
        line_width: Union[int, float] = 3,
        frame_cfg: dict = dict(size=1, origin=[0, 0, 0]),
        alpha: Union[int, float] = 0.8,
        multi_imgs_col: int = 3,
        multi_view_size: Optional[Tuple[int]] = (2400, 800),
        fig_show_cfg: dict = dict(figsize=(18, 12))
    ) -> None:
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir)

        self.image_mode = image_mode
        
        # color settings
        self.bbox_color = bbox_color
        self.text_color = text_color
        self.mask_color = mask_color
        self.line_width = line_width
        self.alpha = alpha

        # default data met
        # When calling
        # `PlanningVisualizer().dataset_meta=xxx`,
        # it will override the default value.
        self.dataset_meta = {}
        
        # points settings
        if points is not None:
            self.set_points(points, pcd_mode=pcd_mode, frame_cfg=frame_cfg)
        self.multi_imgs_col = multi_imgs_col
        self.multi_view_size = multi_view_size
        
        self.fig_show_cfg.update(fig_show_cfg)

        self.flag_pause = False
        self.flag_next = False
        self.flag_exit = False

    def _clear_o3d_vis(self) -> None:
        """Clear open3d vis."""

        if hasattr(self, 'o3d_vis'):
            del self.o3d_vis
            del self.points_colors
            del self.view_control
            if hasattr(self, 'pcd'):
                del self.pcd

    def _initialize_o3d_vis(self, show=True) -> Visualizer:
        """Initialize open3d vis according to frame_cfg.

        Args:
            frame_cfg (dict): The config to create coordinate frame in open3d
                vis.

        Returns:
            :obj:`o3d.visualization.Visualizer`: Created open3d vis.
        """
        if o3d is None or geometry is None:
            raise ImportError(
                'Please run "pip install open3d" to install open3d first.')
        glfw_key_escape = 256  # Esc
        glfw_key_space = 32  # Space
        glfw_key_right = 262  # Right
        o3d_vis = o3d.visualization.VisualizerWithKeyCallback()
        o3d_vis.register_key_callback(glfw_key_escape, self.escape_callback)
        o3d_vis.register_key_action_callback(glfw_key_space,
                                             self.space_action_callback)
        o3d_vis.register_key_callback(glfw_key_right, self.right_callback)
        if os.environ.get('DISPLAY', None) is not None and show:
            o3d_vis.create_window()
            self.view_control = o3d_vis.get_view_control()
        return o3d_vis

    @master_only
    def set_points(self,
                   points: np.ndarray,
                   pcd_mode: int = 0,
                   vis_mode: str = 'replace',
                   frame_cfg: dict = dict(size=1, origin=[0, 0, 0]),
                   points_color: Tuple[float] = (0.8, 0.8, 0.8),
                   points_size: int = 2,
                   mode: str = 'xyz') -> None:
        """Set the point cloud to draw.

        Args:
            points (np.ndarray): Points to visualize with shape (N, 3+C).
            pcd_mode (int): The point cloud mode (coordinates) for the given points:
                0 represents LiDAR, 1 represents CAMERA, 2 represents Depth. Defaults to 0.
            vis_mode (str): The visualization mode in Open3D:

                - 'replace': Replace the existing point cloud with input point
                  cloud.
                - 'add': Add input point cloud into existing point cloud.

                Defaults to 'replace'.
            frame_cfg (dict): The coordinate frame config for Open3D
                visualization initialization.
                Defaults to dict(size=1, origin=[0, 0, 0]).
            points_color (Tuple[float]): The color of points.
                Defaults to (1, 1, 1).
            points_size (int): The size of points to show on visualizer.
                Defaults to 2.
            mode (str): Indicate type of the input points, available mode
                ['xyz', 'xyzrgb']. Defaults to 'xyz'.
        """
        assert points is not None
        assert vis_mode in ('replace', 'add')
        check_type('points', points, np.ndarray)

        if not hasattr(self, 'o3d_vis'):
            self.o3d_vis = self._initialize_o3d_vis()

        # for now we convert points into depth mode for visualization
        if pcd_mode != Coord3DMode.DEPTH:
            points = Coord3DMode.convert(points, pcd_mode, Coord3DMode.DEPTH)

        if hasattr(self, 'pcd') and vis_mode != 'add':
            self.o3d_vis.remove_geometry(self.pcd)

        # set points size in Open3D
        render_option = self.o3d_vis.get_render_option()
        if render_option is not None:
            render_option.point_size = points_size
            render_option.background_color = np.asarray([0, 0, 0])

        points = points.copy()
        pcd = geometry.PointCloud()
        if mode == 'xyz':
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            points_colors = np.tile(
                np.array(points_color), (points.shape[0], 1))
        elif mode == 'xyzrgb':
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            points_colors = points[:, 3:6]
            # normalize to [0, 1] for Open3D drawing
            if not ((points_colors >= 0.0) & (points_colors <= 1.0)).all():
                points_colors /= 255.0
        else:
            raise NotImplementedError

        # create coordinate frame
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(**frame_cfg)
        self.o3d_vis.add_geometry(mesh_frame)

        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        self.o3d_vis.add_geometry(pcd)
        self.pcd = pcd
        self.points_colors = points_colors

    # TODO: assign 3D Box color according to pred / GT labels
    # We draw GT / pred bboxes on the same point cloud scenes
    # for better detection performance comparison
    def draw_bboxes_3d(
        self,
        bboxes_3d: BaseInstance3DBoxes,
        bbox_color: Tuple[float] = (0, 1, 0),
        points_in_box_color: Tuple[float] = (1, 0, 0),
        rot_axis: int = 2,
        center_mode: str = 'lidar_bottom',
        mode: str = 'xyz') -> None:
        """Draw bbox on visualizer and change the color of points inside
        bbox3d.

        Args:
            bboxes_3d (:obj:`BaseInstance3DBoxes`): 3D bbox
                (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            bbox_color (Tuple[float]): The color of 3D bboxes.
                Defaults to (0, 1, 0).
            points_in_box_color (Tuple[float]): The color of points inside 3D
                bboxes. Defaults to (1, 0, 0).
            rot_axis (int): Rotation axis of 3D bboxes. Defaults to 2.
            center_mode (str): Indicates the center of bbox is bottom center or
                gravity center. Available mode
                ['lidar_bottom', 'camera_bottom']. Defaults to 'lidar_bottom'.
            mode (str): Indicates the type of input points, available mode
                ['xyz', 'xyzrgb']. Defaults to 'xyz'.
        """
        # Before visualizing the 3D Boxes in point cloud scene
        # we need to convert the boxes to Depth mode
        check_type('bboxes', bboxes_3d, BaseInstance3DBoxes)

        if not isinstance(bboxes_3d, DepthInstance3DBoxes):
            bboxes_3d = bboxes_3d.convert_to(Box3DMode.DEPTH)

        # convert bboxes to numpy dtype
        bboxes_3d = tensor2ndarray(bboxes_3d.tensor)

        # in_box_color = np.array(points_in_box_color)

        for i in range(len(bboxes_3d)):
            center = bboxes_3d[i, 0:3]
            dim = bboxes_3d[i, 3:6]
            yaw = np.zeros(3)
            yaw[rot_axis] = bboxes_3d[i, 6]
            rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)

            if center_mode == 'lidar_bottom':
                # bottom center to gravity center
                center[rot_axis] += dim[rot_axis] / 2
            elif center_mode == 'camera_bottom':
                # bottom center to gravity center
                center[rot_axis] -= dim[rot_axis] / 2
            box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

            line_set = geometry.LineSet.create_from_oriented_bounding_box(
                box3d)
            line_set.paint_uniform_color(np.array(bbox_color[i]) / 255.)
            # draw bboxes on visualizer
            self.o3d_vis.add_geometry(line_set)

            # change the color of points which are in box
            if self.pcd is not None and mode == 'xyz':
                indices = box3d.get_point_indices_within_bounding_box(
                    self.pcd.points)
                self.points_colors[indices] = np.array(bbox_color[i]) / 255.

        # update points colors
        if self.pcd is not None:
            self.pcd.colors = o3d.utility.Vector3dVector(self.points_colors)
            self.o3d_vis.update_geometry(self.pcd)

    @master_only
    def set_image(self, 
                image: np.ndarray, 
                origin: str = 'upper') -> None:
        """Set the image to draw.

        Args:
            image (np.ndarray): The origin image to draw.
            origin (str): The origin [0, 0] index of the image array. Defaults to 'upper'.
                Options are 'upper' and 'lower'. 'upper' is typically used for camera images,
                and 'lower' is typically used for BEV images.
        """
        super().set_image(image)
        # overwrite show settings
        if origin.lower() == 'lower':
            self.ax_save.cla()
            self.ax_save.axis(False)
            self.ax_save.imshow(
                image, 
                origin=origin,
                interpolation='none')
    
    
    def draw_bev(
        self, 
        pcd_range,
        pixels_per_meter: int = 10,
        background_color: str = 'white'):
        
        dx, dy = pcd_range[3] - pcd_range[0], pcd_range[4] - pcd_range[1]
        
        # fill background colors\
        rgb_color = mcolors.to_rgb(background_color)
        color = (np.array(rgb_color) * 255).astype(np.uint8).reshape((1, 1, 3))

        img = color.repeat(int(dy * pixels_per_meter), axis=0).repeat(int(dx * pixels_per_meter), axis=1)

        self.set_image(img, origin='lower')
        return self.get_image()
            
    # TODO: Support bev point cloud visualization
    @master_only
    def draw_bboxes_on_bev(
        self,
        bbox_3d_ego: BaseInstance3DBoxes,
        bboxes_3d_instances: BaseInstance3DBoxes,
        scale: int = 15,
        edge_colors_ego: Union[str, Tuple[int],
                            List[Union[str, Tuple[int]]]] = 'r',
        edge_colors_instances: Union[str, Tuple[int],
                            List[Union[str, Tuple[int]]]] = 'b',
        line_styles_ego: Union[str, List[str]] = '-',
        line_styles_instances: Union[str, List[str]] = '-',
        line_widths: Union[int, float, List[Union[int,
                                                    float]]] = 1,
        face_colors: Union[str, Tuple[int],
                            List[Union[str,
                                        Tuple[int]]]] = 'none',
        alpha: Union[int, float] = 1) -> MMENGINE_Visualizer:
        """Draw projected 3D boxes on the image.

        Args:
            bbox_3d_ego (:obj:`BaseInstance3DBoxes`): 3D bbox of ego vehicle
            bboxes_3d_instances (:obj:`BaseInstance3DBoxes`): 3D bbox of other agents in the scene,
                (x, y, z, x_size, y_size, z_size, yaw).
            scale (dict): Value to scale the bev bboxes for better
                visualization, i.e., pixels per meter. Defaults to 15.
            edge_colors (str or Tuple[int] or List[str or Tuple[int]]):
                The colors of bboxes. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value, all
                the lines will have the same colors. Refer to `matplotlib.
                colors` for full list of formats that are accepted.
                Defaults to 'o'.
            line_styles (str or List[str]): The linestyle of lines.
                ``line_styles`` can have the same length with texts or just
                single value. If ``line_styles`` is single value, all the lines
                will have the same linestyle. Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (int or float or List[int or float]): The linewidth of
                lines. ``line_widths`` can have the same length with lines or
                just single value. If ``line_widths`` is single value, all the
                lines will have the same linewidth. Defaults to 2.
            face_colors (str or Tuple[int] or List[str or Tuple[int]]):
                The face colors. Defaults to 'none'.
            alpha (int or float): The transparency of bboxes. Defaults to 1.
        """
        
        if bbox_3d_ego is not None:
            self = self._draw_bboxes_on_bev(
                bboxes_3d=bbox_3d_ego, 
                scale=scale, 
                edge_colors=edge_colors_ego, 
                line_styles=line_styles_ego,
                line_widths=line_widths, 
                face_colors=face_colors, 
                alpha=alpha
            )
        
        if bboxes_3d_instances is not None:
            self = self._draw_bboxes_on_bev(
                bboxes_3d=bboxes_3d_instances, 
                scale=scale, 
                edge_colors=edge_colors_instances, 
                line_styles=line_styles_instances,
                line_widths=line_widths, 
                face_colors=face_colors, 
                alpha=alpha)

        return self
        
    def _draw_bboxes_on_bev(
        self,
        bboxes_3d: BaseInstance3DBoxes,
        scale: int = 15,
        edge_colors: Union[str, Tuple[int],
                            List[Union[str, Tuple[int]]]] = 'o',
        line_styles: Union[str, List[str]] = '-',
        line_widths: Union[int, float, List[Union[int,
                                                    float]]] = 1,
        face_colors: Union[str, Tuple[int],
                            List[Union[str,
                                        Tuple[int]]]] = 'none',
        alpha: Union[int, float] = 1) -> MMENGINE_Visualizer:
        """Draw projected 3D boxes on the image.

        Args:
            bboxes_3d (:obj:`BaseInstance3DBoxes`): 3D bbox
                (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            scale (dict): Value to scale the bev bboxes for better
                visualization. Defaults to 15.
            edge_colors (str or Tuple[int] or List[str or Tuple[int]]):
                The colors of bboxes. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value, all
                the lines will have the same colors. Refer to `matplotlib.
                colors` for full list of formats that are accepted.
                Defaults to 'o'.
            line_styles (str or List[str]): The linestyle of lines.
                ``line_styles`` can have the same length with texts or just
                single value. If ``line_styles`` is single value, all the lines
                will have the same linestyle. Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (int or float or List[int or float]): The linewidth of
                lines. ``line_widths`` can have the same length with lines or
                just single value. If ``line_widths`` is single value, all the
                lines will have the same linewidth. Defaults to 2.
            face_colors (str or Tuple[int] or List[str or Tuple[int]]):
                The face colors. Defaults to 'none'.
            alpha (int or float): The transparency of bboxes. Defaults to 1.
        """
        
        check_type('bboxes', bboxes_3d, BaseInstance3DBoxes)
        # Convert to Depth mode for visualization
        if not isinstance(bboxes_3d, DepthInstance3DBoxes):
            bboxes_3d = bboxes_3d.convert_to(Box3DMode.DEPTH)
        
        # convert rgb color to bgr if image is bgr
        if self.image_mode == 'bgr':
            edge_colors = self._rgb_to_bgr(edge_colors)
            if face_colors is not None and face_colors != 'none':
                face_colors = self._rgb_to_bgr(face_colors)
            
        bev_bboxes = tensor2ndarray(bboxes_3d.bev)
        # scale the bev bboxes for better visualization
        bev_bboxes[:, :4] *= scale
        ctr, w, h, theta = np.split(bev_bboxes, [2, 3, 4], axis=-1)
        cos_value, sin_value = np.cos(theta), np.sin(theta)
        vec1 = np.concatenate([w / 2 * cos_value, w / 2 * sin_value], axis=-1)
        vec2 = np.concatenate([-h / 2 * sin_value, h / 2 * cos_value], axis=-1)
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        poly = np.stack([pt1, pt2, pt3, pt4], axis=-2)

        # move lidar (0, 0) to the center of the image
        poly[:, :, 0] += self.width / 2
        poly[:, :, 1] += self.height / 2
        poly = [p for p in poly]
        
        # add arrows to indicate the orientation of the boxes
        # midpoints of the front edge 
        midpt_front = (pt1 + pt2) / 2
        direction = np.stack([midpt_front, ctr], axis=-2)
        direction[..., 0] += self.width / 2
        direction[..., 1] += self.height / 2
        
        self.draw_lines(x_datas=direction[..., 0],
                        y_datas=direction[..., 1],
                        colors=edge_colors,
                        line_styles=line_styles,
                        line_widths=line_widths) 
        
        return self.draw_polygons(
            poly,
            alpha=alpha,
            edge_colors=edge_colors,
            line_styles=line_styles,
            line_widths=line_widths,
            face_colors=face_colors)
 
    @master_only
    def draw_points_on_image(self,
                             points: Union[np.ndarray, Tensor],
                             pts2img: np.ndarray,
                             sizes: Union[np.ndarray, int] = 3,
                             max_depth: Optional[float] = None) -> None:
        """Draw projected points on the image.

        Args:
            points (np.ndarray or Tensor): Points to draw.
            pts2img (np.ndarray): The transformation matrix from the coordinate
                of point cloud to image plane.
            sizes (np.ndarray or int): The marker size. Defaults to 10.
            max_depth (float): The max depth in the color map. Defaults to
                None.
        """
        check_type('points', points, (np.ndarray, Tensor))
        points = tensor2ndarray(points)
        assert self._image is not None, 'Please set image using `set_image`'
        projected_points = points_cam2img(points, pts2img, with_depth=True)
        depths = projected_points[:, 2]
        # Show depth adaptively consideing different scenes
        if max_depth is None:
            max_depth = depths.max()
        colors = (depths % max_depth) / max_depth
        # use colormap to obtain the render color
        color_map = plt.get_cmap('jet')
        self.ax_save.scatter(
            projected_points[:, 0],
            projected_points[:, 1],
            c=colors,
            cmap=color_map,
            s=sizes,
            alpha=0.7,
            edgecolors='none')

    # TODO: set bbox color according to palette
    @master_only
    def draw_bboxes_3d_on_image(
            self,
            bboxes_3d: BaseInstance3DBoxes,
            input_meta: dict,
            edge_colors: Union[str, Tuple[int],
                               List[Union[str, Tuple[int]]]] = 'royalblue',
            line_styles: Union[str, List[str]] = '-',
            line_widths: Union[int, float, List[Union[int, float]]] = 2,
            alpha: Union[int, float] = 0.4,
            img_size: Optional[Tuple] = None):
        """Draw projected 3D boxes on image.

        Args:
            bboxes_3d (:obj:`BaseInstance3DBoxes`): 3D bbox
                (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            input_meta (dict): Input meta information.
            edge_colors (str or Tuple[int] or List[str or Tuple[int]]):
                The RGB colors of bboxes. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value, all
                the lines will have the same colors. Refer to `matplotlib.
                colors` for full list of formats that are accepted.
                Defaults to 'royalblue'.
            line_styles (str or List[str]): The linestyle of lines.
                ``line_styles`` can have the same length with texts or just
                single value. If ``line_styles`` is single value, all the lines
                will have the same linestyle. Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (int or float or List[int or float]): The linewidth of
                lines. ``line_widths`` can have the same length with lines or
                just single value. If ``line_widths`` is single value, all the
                lines will have the same linewidth. Defaults to 2.
            alpha (int or float): The transparency of bboxes. Defaults to 0.4.
            img_size (tuple, optional): The size (w, h) of the image.
        """

        check_type('bboxes', bboxes_3d, BaseInstance3DBoxes)

        if isinstance(bboxes_3d, DepthInstance3DBoxes):
            proj_bbox3d_to_img = proj_depth_bbox3d_to_img
        elif isinstance(bboxes_3d, LiDARInstance3DBoxes):
            proj_bbox3d_to_img = proj_lidar_bbox3d_to_img
        elif isinstance(bboxes_3d, CameraInstance3DBoxes):
            proj_bbox3d_to_img = proj_camera_bbox3d_to_img
        else:
            raise NotImplementedError('unsupported box type!')

        # convert to bgr color, the default color code is in rgb
        if self.image_mode == 'bgr':
            edge_colors = self._rgb_to_bgr(edge_colors)
        edge_colors_norm = color_val_matplotlib(edge_colors)

        corners_2d = proj_bbox3d_to_img(bboxes_3d, input_meta)
        if img_size is not None:
            # Filter out the bbox where half of stuff is outside the image.
            # This is for the visualization of multi-view image.
            valid_point_idx = (corners_2d[..., 0] >= 0) & \
                        (corners_2d[..., 0] <= img_size[0]) & \
                        (corners_2d[..., 1] >= 0) & (corners_2d[..., 1] <= img_size[1])  # noqa: E501
            valid_bbox_idx = valid_point_idx.sum(axis=-1) >= 4
            corners_2d = corners_2d[valid_bbox_idx]
            filter_edge_colors = []
            filter_edge_colors_norm = []
            for i, color in enumerate(edge_colors):
                if valid_bbox_idx[i]:
                    filter_edge_colors.append(color)
                    filter_edge_colors_norm.append(edge_colors_norm[i])
            edge_colors = filter_edge_colors
            edge_colors_norm = filter_edge_colors_norm

        lines_verts_idx = [0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 5, 1, 2, 6]
        lines_verts = corners_2d[:, lines_verts_idx, :]
        front_polys = corners_2d[:, 4:, :]
        codes = [Path.LINETO] * lines_verts.shape[1]
        codes[0] = Path.MOVETO
        pathpatches = []
        for i in range(len(corners_2d)):
            verts = lines_verts[i]
            pth = Path(verts, codes)
            pathpatches.append(PathPatch(pth))

        p = PatchCollection(
            pathpatches,
            facecolors='none',
            edgecolors=edge_colors_norm,
            linewidths=line_widths,
            linestyles=line_styles)

        self.ax_save.add_collection(p)

        # draw a mask on the front of project bboxes
        front_polys = [front_poly for front_poly in front_polys]
        return self.draw_polygons(
            front_polys,
            alpha=alpha,
            edge_colors=edge_colors,
            line_styles=line_styles,
            line_widths=line_widths,
            face_colors=edge_colors)
 
    def _rgb_to_bgr(self, rgb_color: Union[str, Tuple[int], Tuple[str], Tuple[Tuple[int]]]
                    ) -> Union[Tuple[int], Tuple[Tuple[int]]]:
        """Convert RGB color to BGR color.

        Args:
            color (str or Tuple[int] or Tuple[str] or Tuple[Tuple[int]]):
                The color to convert.

        Returns:
            Tuple[int]: The converted BGR color.
        """
        if isinstance(rgb_color, str):
            rgb = mcolors.to_rgb(rgb_color) # rgb in [0, 1]
            bgr_color = mcolors.to_hex(rgb[::-1])
            return bgr_color
        
        elif isinstance(rgb_color, (tuple, list)):
            if isinstance(rgb_color[0], str):
                bgr = [mcolors.to_hex(mcolors.to_rgb(color)[::-1]) for color in rgb_color ]
                return type(rgb_color)(bgr)
            
            elif isinstance(rgb_color[0], int):
                return rgb_color[::-1]

            elif isinstance(rgb_color[0], (tuple, list)):
                bgr = [color[::-1] for color in rgb_color]
                return type(rgb_color)(bgr)
        else:
            raise TypeError('color should be str or tuple')

    def color_map(self, data, cmap):
        """数值映射为颜色"""
        
        dmin, dmax = np.nanmin(data), np.nanmax(data)
        cmo = plt.cm.get_cmap(cmap)
        cs, k = list(), 256/cmo.N
        
        for i in range(cmo.N):
            c = cmo(i)
            for j in range(int(i*k), int((i+1)*k)):
                cs.append(c)
        cs = np.array(cs)
        data = np.uint8(255*(data-dmin)/(dmax-dmin))
        
        return cs[data]
    
    def _generate_trajectory_line_collections(
        self,
        traj_xy: np.ndarray,
    ):
        """_summary_

        Args:
            traj_xy (np.ndarray): Shape (T, 2)

        Returns:
            _type_: _description_
        """
        traj_xy = np.stack((traj_xy[:-1], traj_xy[1:]), axis=1) # (T-1, 2, 2)
        
        traj_vecs = None
        for i in range(traj_xy.shape[0]):
            traj_vec_i = traj_xy[i]
            x_linspace = np.linspace(traj_vec_i[0, 0], traj_vec_i[1, 0], 51)
            y_linspace = np.linspace(traj_vec_i[0, 1], traj_vec_i[1, 1], 51)
            xy = np.stack((x_linspace, y_linspace), axis=1)
            xy = np.stack((xy[:-1], xy[1:]), axis=1)
            if traj_vecs is None:
                traj_vecs = xy
            else:
                traj_vecs = np.concatenate((traj_vecs, xy), axis=0)  
        
        return traj_vecs
    
    def draw_trajectory(
        self, 
        traj: Union[np.ndarray, Trajectory],
        mask: Optional[np.ndarray] = None,
        cmap: Optional[str] = 'autumn_r',
        scale: int = 1,
        linewidths: int = 1, 
        on: Optional[str] = 'image'):

        # check dimensions
        if traj is not None:
            assert isinstance(traj, np.ndarray) and traj.ndim == 2, 'traj should be a 2D numpy array'
        
        T, _ = traj.shape
        
        # filter out invalid trajectory
        traj = traj[mask][..., :2]
        # traj may be empty after masking
        if traj.shape[0] == 0:
            return
        
        # at least 1 valid step
        if traj.shape[0] <= 1:
            return
        
        # setup colors: each line segment has a color
        # every two steps are connected by a line
        segments_per_line = 50
        y = np.sin(np.linspace(1/2*np.pi, 3/2*np.pi, T*segments_per_line))
        colors = self.color_map(y, cmap)
        if self.image_mode.lower() == 'bgr':
            colors[:, [0, 1, 2]] = colors[:, [2, 1, 0]] # rgb to bgr
        
        # generate trajectory line collections
        vecs = self._generate_trajectory_line_collections(traj)      
        # scale meters to pixels
        vecs = vecs * scale
        
        # move center to the middle of the image if in bev mode
        if on == 'bev':
            vecs[..., 0] += self.width / 2
            vecs[..., 1] += self.height / 2
        
        # line collection
        line_collect = LineCollection(
            vecs.tolist(),
            colors=colors,
            linestyles='solid',
            linewidths=linewidths,
            cmap=cmap)
        self.ax_save.add_collection(line_collect)

    @master_only                                 
    def draw_trajectory_on_bev(
        self,
        traj: np.ndarray,
        mask: Optional[np.ndarray] = None,
        cmap: Optional[str] = 'autumn_r',
        scale=10,
        linewidths=1,
        input_meta: Optional[dict] = None
    ):
        """Draw trajectory on BEV image.
            
        
        Args:
            trajs (np.ndarray): Trajectory to draw.
                TrajectoryData: single trajectory for one agent
                list[TrajectoryData]: one trajectory for each agent                
            scale (int): The scale of the BEV image.
        """
       # assertions
        # traj: (N, M, T, d)
        assert isinstance(traj, np.ndarray), 'traj should be a numpy array'
        assert isinstance(mask, np.ndarray), 'mask should be a numpy array'
        if traj.ndim == 2:
            traj = traj[None, None, ...]
        if traj.ndim == 3:
            traj = traj[None, ...]
        
        # mask: (N, T)
        if mask is not None and mask.ndim == 1:
            mask = mask[None, ...]

        # (N, M, T, d)
        N, M, T, _ = traj.shape
        
        # mask out invalid trajectory
        if mask is None:
            mask = np.ones((N, T)).astype(np.bool_)
            
        # mmdet3d lidar to bev image (depth mode)
        if 'lidar2img' in input_meta and input_meta['lidar2img'] is not None:
            traj = np.concatenate([
                traj[..., :2], 
                1.0*np.ones((N, M, T, 1)), # close to ground
                np.ones((N, M, T, 1))], 
                axis=-1)
            
            traj_img = traj @ np.array(input_meta['lidar2img']).T
            traj = traj_img[..., :2] # (N, M, T, 2) 
                
        # future trajectory
        future_steps = input_meta['future_steps']

        # agents
        for i in range(N):
            # modes
            for j in range(M):
                # future trajectory by default            
                traj_ij = traj[i][j]
                mask_i = mask[i]
                self.draw_trajectory(
                    traj = traj_ij, 
                    mask = mask_i, 
                    cmap = cmap, 
                    scale = scale, 
                    linewidths = linewidths,
                    on = 'bev')
               
    @master_only                                 
    def draw_trajectory_on_image(
        self,
        traj: np.ndarray,
        mask: Optional[np.ndarray] = None,
        cmap: Optional[str] = 'winter_r',
        linewidths=1,
        input_meta: Optional[dict] = None
    ):
        """Draw trajectory on BEV image.
            
        
        Args:
            trajs (np.ndarray): Trajectory to draw. Trajectory should be transformed to image coord.
                TrajectoryData: single trajectory for one agent
                list[TrajectoryData]: one trajectory for each agent                
            scale (int): The scale of the BEV image.
        """
       # assertions
        # traj: (N, M, T, d)
        assert isinstance(traj, np.ndarray), 'traj should be a numpy array'
        if mask is not None:
            assert isinstance(mask, np.ndarray), 'mask should be a numpy array'
        assert isinstance(input_meta, dict) and 'future_steps' in input_meta, \
            'input_meta should be a dictionary, and should contain lidar2img and future_steps'
        
        # traj: (N, M, T, d)
        if traj.ndim == 2:
            traj = traj[None, None, ...]
        if traj.ndim == 3:
            traj = traj[None, ...]
        # mask: (N, T)
        if mask is not None and mask.ndim == 1:
            mask = mask[None, ...]
            
        N, M, T, _ = traj.shape
        # mask out invalid trajectory
        if mask is None:
            mask = np.ones((N, T))
            
        # lidar to image: (x, y, z, 1)
        traj = np.concatenate((traj[..., :2], 
                                   -1.5*np.ones((N, M, T, 1)), # close to ground
                                   np.ones((N, M, T, 1))), 
                                  axis=-1)
        
        traj_img = traj @ np.array(input_meta['lidar2img']).T
        traj_img[..., 0] = traj_img[..., 0] / np.maximum(traj_img[..., 2], 1e-5)
        traj_img[..., 1] = traj_img[..., 1] / np.maximum(traj_img[..., 2], 1e-5) 
        traj = traj_img[..., :2] # (N, T, 2)
        
        
        # future trajectory
        future_steps = input_meta['future_steps']

        for i in range(N):
            for j in range(M):
                # future trajectory by default            
                traj_ij = traj[i][j]
                mask_i = mask[i]
                self.draw_trajectory(
                    traj = traj_ij, 
                    mask = mask_i, 
                    cmap = cmap, 
                    linewidths = linewidths)
    
        return self.get_image()
    
    # multi-view image
    @master_only
    def draw_multiviews(
        self, 
        imgs, 
        view_names: Optional[List[str]] = None, 
        target_size: Optional[Tuple[int]]=(2133, 800), 
        arrangement: Optional[Tuple[int]]=(2, 3),
        text_colors: Optional[Union[Tuple[int], str]] = (255, 255, 255),
        text_size: Optional[int] = 20
    ):
        """Set multiview images to draw.
        """
        assert isinstance(imgs, list), 'imgs should be a list'
        if view_names is not None:
            assert len(view_names) == len(imgs), 'view_names should have the same length with imgs'
        
        num_views = len(imgs)
        row, col = arrangement
        assert row * col >= num_views, 'The product of row and col in ' \
                                    'the `arrangement` is less than ' \
                                    'num of views, please set the ' \
                                    '`arrangement` correctly'

        # add multi-view names to image
        views = []
        # default view names of not specified
        if view_names is None:
            view_names = [f'View {i+1}' for i in range(num_views)]
        # draw multi-view images
        for name, img in zip(view_names, imgs):
            self.set_image(img)
            self.draw_texts(name, np.array([10, 10]), font_sizes=text_size, colors=text_colors)
            views.append(self.get_image())

        # TODO: support multi-view image with different shapes
        rows = []
        for i in range(num_views):
            if i % col == 0:
                rows.append([])
            rows[-1].append(views[i])
            
        # stack multi-view images
        multiview = cv2.vconcat([cv2.hconcat(row) for row in rows])
        multiview = cv2.resize(multiview, target_size)
        
        return multiview
        
    @master_only
    def show(self,
             save_path: Optional[str] = None,
             drawn_img: Optional[np.ndarray] = None,
             win_name: str = 'image',
             wait_time: int = -1,
             continue_key: str = 'right',
             backend: str = 'matplotlib') -> None:
        """Show the drawn point cloud/image.

        Args:
            save_path (str, optional): Path to save open3d visualized results.
                Defaults to None.
            drawn_img_3d (np.ndarray, optional): The image to show. If
                drawn_img_3d is not None, it will show the image got by
                Visualizer. Defaults to None.
            drawn_img (np.ndarray, optional): The image to show. If drawn_img
                is not None, it will show the image got by Visualizer.
                Defaults to None.
            win_name (str): The image title. Defaults to 'image'.
            wait_time (int): Delay in milliseconds. 0 is the special value that
                means "forever". Defaults to 0.
            continue_key (str): The key for users to continue. Defaults to ' '.
            backend (str): The backend to show the image. Defaults to
                'matplotlib'. Other option is 'cv2'.
        """

        # In order to show multi-modal results at the same time, we show image
        # firstly and then show point cloud since the running of
        # Open3D will block the process
        if hasattr(self, '_image'):
            super().show(drawn_img=drawn_img, 
                         win_name=win_name,
                         wait_time=wait_time, 
                         continue_key=continue_key,
                         backend=backend)

        if hasattr(self, 'o3d_vis'):
            if hasattr(self, 'view_port'):
                self.view_control.convert_from_pinhole_camera_parameters(
                    self.view_port)
            self.flag_exit = not self.o3d_vis.poll_events()
            self.o3d_vis.update_renderer()
            # if not hasattr(self, 'view_control'):
            #     self.o3d_vis.create_window()
            #     self.view_control = self.o3d_vis.get_view_control()
            self.view_port = \
                self.view_control.convert_to_pinhole_camera_parameters()  # noqa: E501
            if wait_time != -1:
                self.last_time = time.time()
                while time.time(
                ) - self.last_time < wait_time and self.o3d_vis.poll_events():
                    self.o3d_vis.update_renderer()
                    self.view_port = \
                        self.view_control.convert_to_pinhole_camera_parameters()  # noqa: E501
                while self.flag_pause and self.o3d_vis.poll_events():
                    self.o3d_vis.update_renderer()
                    self.view_port = \
                        self.view_control.convert_to_pinhole_camera_parameters()  # noqa: E501

            else:
                while not self.flag_next and self.o3d_vis.poll_events():
                    self.o3d_vis.update_renderer()
                    self.view_port = \
                        self.view_control.convert_to_pinhole_camera_parameters()  # noqa: E501
                self.flag_next = False
            self.o3d_vis.clear_geometries()
            try:
                del self.pcd
            except (KeyError, AttributeError):
                pass
            if save_path is not None:
                if not (save_path.endswith('.png')
                        or save_path.endswith('.jpg')):
                    save_path += '.png'
                self.o3d_vis.capture_screen_image(save_path)
            if self.flag_exit:
                self.o3d_vis.destroy_window()
                self.o3d_vis.close()
                self._clear_o3d_vis()
                sys.exit(0)

    def escape_callback(self, vis):
        self.o3d_vis.clear_geometries()
        self.o3d_vis.destroy_window()
        self.o3d_vis.close()
        self._clear_o3d_vis()
        sys.exit(0)

    def space_action_callback(self, vis, action, mods):
        if action == 1:
            if self.flag_pause:
                print_log(
                    'Playback continued, press [SPACE] to pause.',
                    logger='current')
            else:
                print_log(
                    'Playback paused, press [SPACE] to continue.',
                    logger='current')
            self.flag_pause = not self.flag_pause
        return True

    def right_callback(self, vis):
        self.flag_next = True
        return False


    def _draw_instances_3d(self,
                           data_input: dict,
                           instances: InstanceData,
                           input_meta: dict,
                           vis_task: str,
                           show_pcd_rgb: bool = False,
                           palette: Optional[List[tuple]] = None,
                           view_names: Optional[str] = None) -> dict:
        """Draw 3D instances of GT or prediction on the image or multi-view images.
        
        If the instances is empty, draw the original image.

        Args:
            data_input (dict): The input dict to draw. with image in rgb mode as default
            instances (:obj:`InstanceData`): Data structure for instance-level
                annotations or predictions.
            input_meta (dict): Meta information.
            vis_task (str): Visualization task, it includes: 'lidar_det',
                'multi-modality_det', 'mono_det'.
            show_pcd_rgb (bool): Whether to show RGB point cloud.
            palette (List[tuple], optional): Palette information corresponding
                to the category. Defaults to None.

        Returns:
            dict: The drawn point cloud and image whose channel is RGB.
        """

        # TODO: if no instances, return the original image
        num_instances = len(instances)

        bboxes_3d = instances.bbox  # BaseInstance3DBoxes
        labels_3d = instances.label

        data_3d = dict()

        if vis_task in ['lidar_det', 'multi-modality_det', 'multi-modality_planning']:
            assert 'points' in data_input
            points = data_input['points']
            check_type('points', points, (np.ndarray, Tensor))
            points = tensor2ndarray(points)

            if num_instances > 0:
                if not isinstance(bboxes_3d, DepthInstance3DBoxes):
                    _, bboxes_3d_depth = to_depth_mode(None, bboxes_3d)
                else:
                    bboxes_3d_depth = bboxes_3d.clone()

                max_label = int(max(labels_3d) if len(labels_3d) > 0 else 0)
                bbox_color = palette if self.bbox_color is None \
                    else self.bbox_color
                bbox_palette = get_palette(bbox_color, max_label + 1)
                colors = [bbox_palette[label] for label in labels_3d]
                
            if 'axis_align_matrix' in input_meta:
                points = DepthPoints(points, points_dim=points.shape[1])
                rot_mat = input_meta['axis_align_matrix'][:3, :3]
                trans_vec = input_meta['axis_align_matrix'][:3, -1]
                points.rotate(rot_mat.T)
                points.translate(trans_vec)
                points = tensor2ndarray(points.tensor)

            self.set_points(
                points, pcd_mode=0, mode='xyzrgb' if show_pcd_rgb else 'xyz')
            
            if num_instances > 0:
                self.draw_bboxes_3d(bboxes_3d, bbox_color=colors)
                data_3d['bboxes_3d'] = tensor2ndarray(bboxes_3d_depth.tensor)
                
            data_3d['points'] = points

        if vis_task in ['mono_det', 'multi-modality_det', 'multi-modality_planning']:
            assert 'img' in data_input
            img = data_input['img']
            if isinstance(img, list) or (isinstance(img, (np.ndarray, Tensor))
                                         and len(img.shape) == 4):
                # show multi-view images
                img_size = img[0].shape[-2:]
                img_col = self.multi_imgs_col
                img_row = math.ceil(len(img) / img_col)
                              
                # initialize a combined image
                composed_img = [np.zeros((*img_size, 3)) for _ in range(img_col * img_row)]
                
                for i, single_img in enumerate(img):
                    # Note that we should keep the same order of elements both
                    # in `img` and `input_meta`
                    if isinstance(single_img, Tensor):
                        single_img = single_img.permute(1, 2, 0).numpy()
                    self.set_image(single_img)
                    single_img_meta = dict()
                    for key, meta in input_meta.items():
                        if isinstance(meta,
                                      (Sequence, np.ndarray,
                                       Tensor)) and len(meta) == len(img):
                            single_img_meta[key] = meta[i]
                        else:
                            single_img_meta[key] = meta
                    
                    if num_instances > 0:
                        max_label = int(
                            max(labels_3d) if len(labels_3d) > 0 else 0)
                        bbox_color = palette if self.bbox_color is None \
                            else self.bbox_color
                        bbox_palette = get_palette(bbox_color, max_label + 1)
                        colors = [bbox_palette[label] for label in labels_3d]
                        self.draw_bboxes_3d_on_image(
                            bboxes_3d,
                            single_img_meta,
                            img_size=single_img.shape[:2][::-1],
                            edge_colors=colors)
                    if vis_task == 'mono_det' and hasattr(
                            instances, 'centers_2d'):
                        centers_2d = instances.centers_2d
                        self.draw_points(centers_2d)
                    #composed_img[(i // img_col) *
                    #             img_size[0]:(i // img_col + 1) * img_size[0],
                    #             (i % img_col) *
                    #             img_size[1]:(i % img_col + 1) *
                    #             img_size[1]] = self.get_image()
                    composed_img[i] = self.get_image()
                
                # arrange images given names
                img_names = input_meta['img_names'] # camera for each view
                if view_names is not None:
                    composed_img = [composed_img[img_names.index(name)] for name in view_names]

                composed_img = self.draw_multiviews(imgs = composed_img, 
                                        view_names = view_names if view_names is not None else img_names,
                                        target_size = self.multi_view_size,
                                        arrangement = (img_row, img_col),
                                        text_colors = (255, 255, 255)
                )
                    
                data_3d['img'] = composed_img
            else:
                # show single-view image
                # TODO: Solve the problem: some line segments of 3d bboxes are
                # out of image by a large margin
                if isinstance(data_input['img'], Tensor):
                    img = img.permute(1, 2, 0).numpy()
                self.set_image(img)

                if num_instances > 0:
                    max_label = int(max(labels_3d) if len(labels_3d) > 0 else 0)
                    bbox_color = palette if self.bbox_color is None \
                        else self.bbox_color
                    bbox_palette = get_palette(bbox_color, max_label + 1)
                    colors = [bbox_palette[label] for label in labels_3d]

                    self.draw_proj_bboxes_3d(
                        bboxes_3d, input_meta, edge_colors=colors)
                if vis_task == 'mono_det' and hasattr(instances, 'centers_2d'):
                    centers_2d = instances.centers_2d
                    self.draw_points(centers_2d)
                drawn_img = self.get_image()
                data_3d['img'] = drawn_img

        return data_3d
    
    # draw map
    def draw_vector_map(
        self,
        vectors: np.ndarray,
        map_labels: List[int],
        pcd_range: List[float] = [-50, -50, -1.5, 50, 50, 1.5],
        pixels_per_meter: float = 10,
        map_classes: List[str] = ['divider', 'ped_crossing', 'boundary'],
        map_colors: List[Tuple[int]] = ['cornflowerblue', 'royalblue', 'slategrey'],
        map_format: str = 'fixed_num_pts'):
        """Draw vector map on the image.
        """
        assert isinstance(vectors, np.ndarray), 'vectors should be a numpy array'
        
        # check dimensions
        assert len(map_classes) == len(map_colors), 'map_classes and map_colors should have the same length'.format(
            len(map_classes), len(map_colors))
        
        if map_format not in ['fixed_num_pts', 'polyline', 'bbox']:
            raise ValueError('map_format should be one of fixed_num_pts, polyline, bbox')
        
        # convert to bgr color, the default color code is in rgb
        if self.image_mode == 'bgr':
            map_colors = self._rgb_to_bgr(map_colors)
        
        # generate a bev 
        bev = self.draw_bev(
            pcd_range = pcd_range,
            pixels_per_meter = pixels_per_meter,
        )
        
        width, height = bev.shape[1], bev.shape[0]

        # sample points for each vector in a map box
        # (num_box, num_points, 2)
        if map_format == 'fixed_num_pts':
            assert vectors.ndim == 3, 'vectors should be a 3D numpy array'
            assert vectors.shape[-1] == 2, 'vectors should be a 3D numpy array with last dimension of 2'
            assert len(vectors) == len(map_labels), 'vectors and map_labels should have the same length'
            
            for pts, label in zip(vectors, map_labels):
                # draw points
                pts = pts.reshape(-1, 2)
                pts_x, pts_y = pts[:, 0], pts[:, 1]
                
                # local map is in lidar coord, plot them in depth coord
                pts_x, pts_y = -pts_y, pts_x 
                
                # scale the points to pixels
                pts_x *= pixels_per_meter
                pts_y *= pixels_per_meter
                pts_x += width // 2
                pts_y += height // 2
                
                self.draw_points(np.stack([pts_x, pts_y], axis=1),
                                colors=[map_colors[label]],
                                sizes=4)
                
                self.draw_lines(np.stack([pts_x[:-1], pts_x[1:]], axis=1),
                                np.stack([pts_y[:-1], pts_y[1:]], axis=1),
                                colors=[map_colors[label]],
                                line_widths=1)
                
        return self.get_image()
    
    def _draw_map_bev(
        self,
        data_sample,
        pcd_range: List[float] = [-50, -50, -1.5, 50, 50, 1.5],
        pixels_per_meter: float = 10,
        map_format: str = 'fixed_num_pts',
        map_classes: List[str] = ['divider', 'ped_crossing', 'boundary'],
        map_palette: List[Tuple[int]] = ['cornflowerblue', 'royalblue', 'slategrey'],
        bboxes_palette: List[Tuple[int]] = None,
        to_mmdet3d_lidar = None,
        ) -> np.ndarray:
        
        # draw vector map
        if map_format == 'fixed_num_pts':
            vectors = data_sample.gt_map_vectors.pt.fixed_num_sampled_points
            vectors = vectors.numpy()
            labels = data_sample.gt_map_vectors.label.numpy()
            
            self.draw_vector_map(
                vectors = vectors,
                map_labels = labels,
                pcd_range = pcd_range,
                pixels_per_meter = pixels_per_meter,
                map_classes = map_classes,
                map_colors = map_palette,
                map_format = map_format
            )
            
        elif map_format == 'polyline':
            pass 
        else:
            self.draw_bev(
                pcd_range = pcd_range,
                pixels_per_meter = pixels_per_meter,
                )

        # draw boxes on map bev
        ego_box = self._get_ego_box(
            ego_size = data_sample.metainfo['ego_size'],
            to_mmdet3d_lidar = to_mmdet3d_lidar,
        )
        num_instances = len(data_sample.gt_instances_3d)
        bboxes_label = data_sample.gt_instances_3d.label
        if num_instances > 0:
            max_label = int(
                max(bboxes_label) if len(bboxes_label) > 0 else 0)
            bbox_color = bboxes_palette if self.bbox_color is None \
                else self.bbox_color
            bbox_palette = get_palette(bbox_color, max_label + 1)
            colors = [bbox_palette[label] for label in bboxes_label]
            bboxes_3d = data_sample.gt_instances_3d.bbox
        

        self.draw_bboxes_on_bev(
            bbox_3d_ego = ego_box,
            bboxes_3d_instances = bboxes_3d,
            scale = pixels_per_meter,
            edge_colors_instances = colors,
        )

        bev = self.get_image()
        return bev

    @master_only
    def add_datasample(self,
                       name: str,
                       data_input: dict,
                       data_sample: Optional[Det3DDataSample] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: float = 0,
                       out_file: Optional[str] = None,
                       o3d_save_path: Optional[str] = None,
                       vis_task: str = 'mono_det',
                       pred_score_thr: float = 0.3,
                       step: int = 0,
                       show_pcd_rgb: bool = False,
                       multi_view_names: Optional[List[str]] = None,
                       pcd_range: Optional[List[float]] = None,
                       map_format: str = 'fixed_num_pts',
                       pixels_per_meter: float = 10,
                       to_mmdet3d_lidar = None,
        ) -> None:
        """Draw datasample and save to all backends.
            - draw ego trajectory planning on given camera, e.g., front camera
            - draw 3D bboxes on multi-view images

        - If GT and prediction are plotted at the same time, they are displayed
          in a stitched image where the left image is the ground truth and the
          right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and the images
          will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be saved to
          ``out_file``. It is usually used when the display is not available.

        Args:
            name (str): The image identifier.
            data_input (dict): It should include the point clouds or image
                to draw.
            data_sample (:obj:`Det3DDataSample`, optional): Prediction
                Det3DDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT Det3DDataSample.
                Defaults to True.
            draw_pred (bool): Whether to draw Prediction Det3DDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn point clouds and image.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str, optional): Path to output file. Defaults to None.
            o3d_save_path (str, optional): Path to save open3d visualized
                results. Defaults to None.
            vis_task (str): Visualization task. Defaults to 'mono_det'.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
            show_pcd_rgb (bool): Whether to show RGB point cloud. Defaults to
                False.
            multi_view_names (list[str], optional): The names of the multi-view
                images. Defaults to None.
        """
        assert vis_task in (
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det', 'multi-modality_planning'), f'got unexpected vis_task {vis_task}.'
        assert map_format in ('fixed_num_pts', 'polyline', 'bbox'), f'got unexpected map_format {map_format}.'
        
        classes = self.dataset_meta.get('classes', None)
        map_classes = self.dataset_meta.get('map_classes', None)
        # For object detection datasets, no palette is saved
        palette = self.dataset_meta.get('palette', None)
        map_palette = self.dataset_meta.get('map_palette', None)
        
        ignore_index = self.dataset_meta.get('ignore_index', None)
        if vis_task == 'lidar_seg' and ignore_index is not None and 'seg_mask' in data_sample.gt_pts:  # noqa: E501
            keep_index = data_sample.gt_pts.seg_mask != ignore_index  # noqa: E501
        else:
            keep_index = None

        gt_data_3d = None
        pred_data_3d = None

        if not hasattr(self, 'o3d_vis') and vis_task in [
                'multi-view_det', 'lidar_det', 'lidar_seg',
                'multi-modality_det', 'multi-modality_planning'
        ]:
            self.o3d_vis = self._initialize_o3d_vis(show=show)
        
        # copy data_input to avoid overwriting the original data
        data_input_cpy = copy.deepcopy(data_input)
        if draw_gt and data_sample is not None:
            # draw gt ego trajectory on front camera
            front_cam_idx = data_sample.metainfo['img_names'].index('CAM_FRONT')
            
            if data_sample.gt_ego is not None and vis_task == 'multi-modality_planning':
                img = data_input['img'][front_cam_idx].permute(1, 2, 0).numpy()
                
                # we use original lidar2img because in VisualizationHook, the image is reloaded from file
                # without using the images after the pipeline
                if 'ori_lidar2img' in data_sample.metainfo:
                    lidar2img = data_sample.metainfo['ori_lidar2img'][front_cam_idx]
                else:
                    lidar2img = data_sample.metainfo['lidar2img'][front_cam_idx]
                lidar2img = np.array(lidar2img)
                ego_traj = data_sample.gt_ego.traj.cumsum(axis=0)[..., :2]
                ego_traj = ego_traj.numpy()
                ego_traj_mask = data_sample.gt_ego.traj_mask.numpy()
                input_meta = {'lidar2img': lidar2img,
                              'future_steps': ego_traj.shape[-2]}

                self.set_image(img)
                self.draw_trajectory_on_image(
                    ego_traj, 
                    ego_traj_mask, 
                    input_meta=input_meta,
                    linewidths=4)
                img_traj = self.get_image()
                
                # save back to data_input
                data_input_cpy['img'][front_cam_idx] = torch.from_numpy(img_traj).permute(2, 0, 1)

            # draw 3d bboxes on images
            if data_sample.gt_instances_3d is not None:
                gt_data_3d = self._draw_instances_3d(
                    data_input_cpy, 
                    data_sample.gt_instances_3d,
                    data_sample.metainfo, 
                    vis_task, 
                    show_pcd_rgb, 
                    palette,
                    view_names = multi_view_names
                )
            # draw lidar segmentation
            if data_sample.gt_pts is not None and vis_task == 'lidar_seg':
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing semantic ' \
                                            'segmentation results.'
                assert 'pts' in data_input
                self._draw_pts_sem_seg(data_input_cpy['pts'],
                                       data_sample.gt_pts.seg, palette,
                                       keep_index)

            # draw vector map and bbox
            if data_sample.gt_map_vectors is not None:
                bev = self._draw_map_bev(
                    data_sample,
                    pcd_range = pcd_range,
                    pixels_per_meter = pixels_per_meter,
                    map_format = map_format,
                    map_classes = map_classes,
                    map_palette = map_palette,
                    bboxes_palette = palette,
                    to_mmdet3d_lidar = to_mmdet3d_lidar
                )
                
                if gt_data_3d is not None:
                    gt_data_3d['bev'] = bev
                else:
                    gt_data_3d = dict()
                    gt_data_3d['bev'] = bev
                
        if draw_pred and data_sample is not None:
            # draw gt ego trajectory on front camera
            if data_sample.pred_ego is not None and vis_task == 'multi-modality_planning':
                img = data_input['img'][front_cam_idx].permute(1, 2, 0).numpy()
                lidar2img = data_sample.metainfo['lidar2img'][front_cam_idx]
                ego_traj = data_sample.pred_ego.traj.numpy().cumsum(axis=1)[..., :2]
                ego_traj_mask = data_sample.pred_ego.get('traj_mask', None)
                if ego_traj_mask is not None:
                    ego_traj_mask = ego_traj_mask.numpy()
                input_meta = {'lidar2img': lidar2img,
                              'future_steps': ego_traj.shape[1]}

                self.draw_trajectory_image(img, ego_traj, ego_traj_mask, input_meta=input_meta)
                img_traj = self.get_image()
                
                # save back to data_input
                data_input_cpy['img'][front_cam_idx] = torch.from_numpy(img_traj).permute(2, 0, 1)
                
            # draw 3d bboxes on images
            if data_sample.pred_instances_3d is not None:
                pred_instances_3d = data_sample.pred_instances_3d
                # .cpu can not be used for BaseInstance3DBoxes
                # so we need to use .to('cpu')
                if hasattr(pred_instances_3d, 'scores') and pred_instances_3d.score is not None:                                       
                    pred_instances_3d = pred_instances_3d[
                        pred_instances_3d.score > pred_score_thr].to('cpu')
                    
                pred_data_3d = self._draw_instances_3d(data_input_cpy,
                                                       pred_instances_3d,
                                                       data_sample.metainfo,
                                                       vis_task, 
                                                       show_pcd_rgb,
                                                       palette)
            # draw lidar segmentation
            if data_sample.pred_pts is not None and vis_task == 'lidar_seg':
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing semantic ' \
                                            'segmentation results.'
                assert 'pts' in data_input
                self._draw_pts_sem_seg(data_input_cpy['pts'],
                                       data_sample.pred_pts.seg, palette,
                                       keep_index)

        # monocular 3d object detection image
        if vis_task in ['mono_det', 'multi-modality_det', 'multi-modality_planning']:
            if gt_data_3d is not None and pred_data_3d is not None:
                drawn_img_3d = np.concatenate(
                    (gt_data_3d['img'], pred_data_3d['img']), axis=1)
            elif gt_data_3d is not None:
                if 'img' in gt_data_3d and 'bev' in gt_data_3d:
                    img = gt_data_3d['img']
                    bev = gt_data_3d['bev']
                    # resize bev to img 
                    bev = cv2.resize(bev, (img.shape[1]//self.multi_imgs_col, img.shape[0]))
                    drawn_img_3d = np.concatenate((img, bev), axis=1)
                elif 'img' in gt_data_3d: 
                    drawn_img_3d = gt_data_3d['img']
                elif 'bev' in gt_data_3d:
                    drawn_img_3d = gt_data_3d['bev']
            elif pred_data_3d is not None:
                drawn_img_3d = pred_data_3d['img']
            else:  # both instances of gt and pred are empty
                drawn_img_3d = None
        else:
            drawn_img_3d = None


        if show:
            backend = 'matplotlib'#'matplotlib' # cv2
            drawn_img = drawn_img_3d
            if backend == 'matplotlib' and self.image_mode.lower() == 'bgr':
                drawn_img = cv2.cvtColor(drawn_img_3d, cv2.COLOR_BGR2RGB)
            elif backend == 'cv2' and self.image_mode.lower() == 'rgb':
                drawn_img = cv2.cvtColor(drawn_img_3d, cv2.COLOR_RGB2BGR)    
                

            self.show(
                o3d_save_path,
                drawn_img,
                win_name=name,
                wait_time=wait_time,
                backend=backend)
            
        if out_file is not None:
            # check the suffix of the name of image file
            if not (out_file.endswith('.png') or out_file.endswith('.jpg')):
                out_file = f'{out_file}.png'
            if drawn_img is not None:
                mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)

    def _get_ego_box(
        self, 
        ego_size,
        to_mmdet3d_lidar=None
        ):
        """Get ego box in depth coordinate.
        """
        # ego_size: (l, w, h)
        l, w, h = ego_size
        
        # ego box in nuscenes lidar coord 
        box = np.array([[0, 0, 0, l, w, h, np.pi/2, 0, 0]])
        
        # bev box need to be drawn in mmdet3d depth coordinate
        # if dataset has been converted to mmdet3d lidar coord, 
        # ego_box should be in mmdet3d depth coord for plotting on bev
        if to_mmdet3d_lidar is not None:
            box = DepthInstance3DBoxes(box, box_dim=9)
        # else the dataset is in original coord, assuming nuscenes lidar coord as default
        # then to be consistent with all other boxes, use mmdet3d lidar coord
        else:
            box = LiDARInstance3DBoxes(box, box_dim=9)

        return box