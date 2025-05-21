import os 
import cv2
import numpy as np
import torch 

from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmdet3d.structures import LiDARInstance3DBoxes, DepthInstance3DBoxes, limit_period, Box3DMode
from fsd.runner import Runner
from fsd.registry import DATASETS, VISUALIZERS

import matplotlib.pyplot as plt

def get_cam_names(img_paths):
    """Get camera names from image paths.
        
    Args:
        img_paths (list): List of image paths. e.g., xx/xx/xx/CAM_FRONT_LEFT/xx.jpg

    Returns:
        list: List of camera names.
    """
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    cam_names = []
    for path in img_paths:
        for subpath in path.split('/'):
            if subpath in cams:
                cam_names.append(subpath)   
                break
    return cam_names

def _convert_kitti_to_mmdet3d(data_sample):
    """Convert kitti coordinate to mmdet3d coordinate.
    
    Args:
        bboxes_3d (LiDARInstance3DBoxes): 3D boxes in kitti coordinate.

    Returns:
        LiDARInstance3DBoxes: 3D boxes in mmdet3d coordinate.
    """
    bboxes_3d = data_sample.gt_instances_3d.bbox
    bboxes_data = bboxes_3d.tensor.clone()
    bboxes_data[:, 3:6] = bboxes_data[:, [4, 3, 5]]
    bboxes_data[:, 6] = -bboxes_data[:, 6] - np.pi/2
    bboxes_3d = bboxes_3d.new_box(data=bboxes_data)
    
    data_sample.gt_instances_3d.bbox = bboxes_3d
    return data_sample

def draw_boxes_3d_on_image(data_inputs, data_samples, vis):
    """Draw 3D boxes on image.
    
    Args:
        data_inputs (dict): Data inputs.
        data_samples (list): Data samples.
        vis (Visualizer): Visualizer.
    """
    instances = data_samples[0].gt_instances_3d
    bboxes_3d = instances.bbox 

    # display the data
    imgs = data_inputs['img'][0]
    imgs = [img.numpy().transpose(1, 2, 0) for img in imgs]
    #pts = data_inputs['pts'].tensor.numpy()
    
    # draw front camera image
    cam_names = get_cam_names(data_samples[0].metainfo['img_path'])
    front_cam_idx = cam_names.index('CAM_FRONT')
    front_img = imgs[front_cam_idx]
    lidar2img = np.array(data_samples[0].metainfo['lidar2img'][front_cam_idx]) # 4x4
        
    vis.set_image(front_img)
    vis.draw_bboxes_3d_on_image(bboxes_3d, 
                        input_meta = {'lidar2img': lidar2img},
                        edge_colors= 'orange'
                        )
    return vis.get_image()

def draw_trajectory_on_image(data_inputs, data_samples, vis):
    # display the data
    imgs = data_inputs['img'][0]
    imgs = [img.numpy().transpose(1, 2, 0) for img in imgs]
    
    # draw front camera image
    cam_names = get_cam_names(data_samples[0].metainfo['img_path'])
    front_cam_idx = cam_names.index('CAM_FRONT')
    front_img = imgs[front_cam_idx]
    lidar2img = np.array(data_samples[0].metainfo['lidar2img'][front_cam_idx]) # 4x4
    
    # ego traj
    ego_traj = data_samples[0].gt_ego.traj[:, [0, 1, 3]].cumsum(axis=0)
    ego_traj_xyr = ego_traj.numpy()
    ego_traj_mask = data_samples[0].gt_ego.traj_mask.numpy()
    
    # draw 
    vis.set_image(front_img)
    vis.draw_trajectory_on_image(
        ego_traj_xyr,
        ego_traj_mask,
        input_meta = {'lidar2img': lidar2img,
                      'future_steps': ego_traj.shape[0]},
        linewidths=4)

    return vis.get_image()

def draw_bev_bboxes(data_inputs, data_samples, vis, ds):
    """Draw BEV boxes on image.
    
    Args:
        data_inputs (dict): Data inputs.
        data_samples (list): Data samples.
        vis (Visualizer): Visualizer.
    """
    # instances
    instances = data_samples[0].gt_instances_3d
    bboxes_3d = instances.bbox
    
    # ego
    ego_size = data_samples[0].metainfo['ego_size']
    # nuscene lidar coordinate
    ego_box = torch.tensor([[0, 0, 0, ego_size[0], ego_size[1], ego_size[2], np.pi/2, 0, 0]])
    
    # bev box need to be draw on the mmdet3d lidar coordinate
    # if no transformation is provided, we assume the original coordinate is
    # nuscenes lidar coordinate
    if ds.dataset.to_mmdet3d_lidar is None:
        ego_box = LiDARInstance3DBoxes(ego_box, box_dim=9)
    
    else:
        # nuscene lidar coord is the same as mmdet3d depth coord
        ego_box = DepthInstance3DBoxes(ego_box, box_dim=9)
    
    # display the data
    bev = 255*np.ones((900, 1200, 3), dtype=np.uint8)
    vis.set_image(bev, origin='lower')
    vis.draw_bboxes_on_bev(bbox_3d_ego = ego_box,
                        bboxes_3d_instances = bboxes_3d)

    return vis.get_image()
    
def draw_multiviews(data_inputs, data_samples, vis):
    # display the data
    imgs = data_inputs['img'][0]
    imgs = [img.numpy().transpose(1, 2, 0) for img in imgs]
    
    # draw front camera image
    cam_names = get_cam_names(data_samples[0].metainfo['img_path'])
    front_cam_idx = cam_names.index('CAM_FRONT')
    front_img = imgs[front_cam_idx]
    lidar2img = np.array(data_samples[0].metainfo['lidar2img'][front_cam_idx]) # 4x4
    
    # draw trajectory on front image
    draw_trajectory_on_image(data_inputs, data_samples, vis)
    
    # get front image with trajectory
    front_image = vis.get_image()
    
    # multi-view:
    imgs[front_cam_idx] = front_image
    
    # show multiview images following the order [CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT,
    # CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT]
    view_order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                  'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    # get the index of each camera in the original order
    cam_idx = [cam_names.index(cam) for cam in view_order]
    # reorder the images
    imgs = [imgs[i] for i in cam_idx]
    # draw the multiview images
    multiview_imgs = vis.draw_multiviews(imgs, 
                        view_order,
                        target_size=(2100, 900), 
                        arrangement=(2,3),
                        text_colors=(255, 255, 255))
    #cv2.imwrite('./multiview_imgs.jpg', multiview_imgs) # bgr   
    
    return multiview_imgs
    
def draw_boxes_on_point_cloud(data_inputs, data_samples, vis):
    """Draw point cloud on image.
    
    Args:
        data_inputs (dict): Data inputs.
        data_samples (list): Data samples.
        vis (Visualizer): Visualizer.
    """
    instances = data_samples[0].gt_instances_3d
    bboxes_3d = instances.bbox 

    
    # display the data
    imgs = data_inputs['img'][0]
    imgs = [img.numpy().transpose(1, 2, 0) for img in imgs]
    pts = data_inputs['points'][0].numpy()
    
    # ego box
    ego_size = data_samples[0].metainfo['ego_size']
    # nuscene lidar coordinate
    ego_box = torch.tensor([[0, 0, 0, ego_size[0], ego_size[1], ego_size[2], np.pi/2, 0, 0]])
    
    # bev box need to be draw on the mmdet3d lidar coordinate
    # if no transformation is provided, we assume the original coordinate is
    # nuscenes lidar coordinate
    if ds.dataset.to_mmdet3d_lidar is None:
        ego_box = LiDARInstance3DBoxes(ego_box, box_dim=9)
    
    else:
        # nuscene lidar coord is the same as mmdet3d depth coord
        ego_box = DepthInstance3DBoxes(ego_box, box_dim=9)
    
    vis.set_points(pts, pcd_mode=0) # 0: lidar, 1: cam mode 2: depth
    bboxes_colors = [(0, 255, 0) for _ in range(len(bboxes_3d))] 
    vis.draw_bboxes_3d(bboxes_3d, bboxes_colors)
    vis.draw_bboxes_3d(ego_box, [(255, 0, 0)])
    vis.show()

def draw_trajectory_on_bev(data_inputs, data_samples, vis, ds):
    # display the data
    imgs = data_inputs['img'][0]
    imgs = [img.numpy().transpose(1, 2, 0) for img in imgs]
    
    # bev image
    bev = 255 * np.ones((900, 1200, 3), dtype=np.uint8)
    vis.set_image(bev, origin='lower')
    bev = draw_bev_bboxes(data_inputs, data_samples, vis, ds)
    
    # ego traj
    ego_traj = data_samples[0].gt_ego.traj[:, [0, 1, 3]].cumsum(axis=0)
    ego_traj_xyr = ego_traj.numpy()
    ego_traj_mask = data_samples[0].gt_ego.traj_mask.numpy()
    
    # lidar2bev for trajectory transformation
    depth2lidar = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    lidar2bev = None
    if ds.dataset.to_mmdet3d_lidar is not None:
        lidar2bev = np.linalg.inv(ds.dataset.to_mmdet3d_lidar)
    # if the dataset in nuscene lidar coordinate (mmdet3d depth coordinate),
    # the trajectory is still in nuscene lidar coordinate
    # to plot such trajectory in mmdet3d lidar coordinate, we need depth2lidar
    # but bev plot is in mmdet3d depth coord, thus we need to inverse the depth2lidar
    else:
        lidar2bev = np.linalg.inv(depth2lidar)
        
    # draw
    vis.draw_trajectory_on_bev(
        ego_traj_xyr,
        ego_traj_mask,
        cmap='autumn',
        input_meta = {
            'lidar2img': lidar2bev,
            'future_steps': ego_traj.shape[0]},
        linewidths=2)    
    
    return vis.get_image()


def draw_mutimodal_trajectory_on_bev(data_inputs, data_samples, vis, ds):
    # display the data
    imgs = data_inputs['img'][0]
    imgs = [img.numpy().transpose(1, 2, 0) for img in imgs]
    
    # bev image
    bev = 255 * np.ones((900, 1200, 3), dtype=np.uint8)
    vis.set_image(bev, origin='lower')
    bev = draw_bev_bboxes(data_inputs, data_samples, vis, ds)
    
    # ego traj
    ego_traj = data_samples[0].gt_ego.traj[:, [0, 1, 3]].cumsum(axis=0)
    ego_traj_xyr = ego_traj.numpy()
    ego_traj_mask = data_samples[0].gt_ego.traj_mask.numpy()
    # add noise to the trajectory to make multimodal
    M = 3
    ego_multimodal_traj = [ego_traj + np.random.rand(*ego_traj_xyr.shape) for _ in range(M)] 
    ego_multimodal_traj = np.stack(ego_multimodal_traj, axis=0)

    # lidar2bev for trajectory transformation
    depth2lidar = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # lidar2bev 
    lidar2bev = None
    if ds.dataset.to_mmdet3d_lidar is not None:
        lidar2bev = np.linalg.inv(ds.dataset.to_mmdet3d_lidar)
    # if the dataset in nuscene lidar coordinate (mmdet3d depth coordinate),
    # the trajectory is still in nuscene lidar coordinate
    # to plot such trajectory in mmdet3d lidar coordinate, we need depth2lidar
    # but bev plot is in mmdet3d depth coord, thus we need to inverse the depth2lidar
    else:
        lidar2bev = np.linalg.inv(depth2lidar)
            
    # draw
    vis.draw_trajectory_on_bev(
        ego_multimodal_traj,
        ego_traj_mask,
        cmap='autumn',
        input_meta = {
            'lidar2img': lidar2bev,
            'future_steps': ego_multimodal_traj.shape[-2]},
        linewidths=2)    
    
    return vis.get_image()


def draw_mutimodal_trajectory_on_image(data_inputs, data_samples, vis):
    # display the data
    imgs = data_inputs['img'][0]
    imgs = [img.numpy().transpose(1, 2, 0) for img in imgs]
    
    # draw front camera image
    cam_names = get_cam_names(data_samples[0].metainfo['img_path'])
    front_cam_idx = cam_names.index('CAM_FRONT')
    front_img = imgs[front_cam_idx]
    lidar2img = np.array(data_samples[0].metainfo['lidar2img'][front_cam_idx]) # 4x4
    
    # ego traj
    ego_traj = data_samples[0].gt_ego.traj[:, [0, 1, 3]].cumsum(axis=0)
    ego_traj_xyr = ego_traj.numpy()
    ego_traj_mask = data_samples[0].gt_ego.traj_mask.numpy()
    M = 3
    ego_multimodal_traj = [ego_traj_xyr + np.random.rand(*ego_traj_xyr.shape) for _ in range(M)]
    ego_multimodal_traj = np.stack(ego_multimodal_traj, axis=0)
    
    # draw 
    vis.set_image(front_img)
    vis.draw_trajectory_on_image(
        ego_multimodal_traj,
        ego_traj_mask,
        input_meta = {'lidar2img': lidar2img,
                      'future_steps': ego_multimodal_traj.shape[-2]},
        linewidths=4)

    return vis.get_image()


def draw_map(data_inputs, data_samples, vis):
    
    # map data
    data_sample = data_samples[0]
    gt_map = data_sample.gt_map_vectors
    
    formats = ['se_pts', 'fixed_num_pts', 'bbox', 'polyline']
    format = formats[1]
    

    
    
    classes = ['divider', 'ped_crossing', 'boundary']
    colors = ['cornflowerblue', 'royalblue', 'slategrey']
    vecs = gt_map.pt 
    labels = gt_map.label.cpu().numpy()
    
    # (num_boxes, num_vectors, num_points, 2)
    all_pts = vecs.fixed_num_sampled_points.to('cpu').numpy()
    #all_pts = all_pts.reshape(len(labels), -1, 2)
    
    # (num_boxes, 4)
    all_se_pts = vecs.start_end_points.to('cpu').numpy()
    all_boxes = vecs.bbox.to('cpu').numpy()
    all_polys = vecs.instance_list

    fig, axes = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
    plt.xlim(xmin=-50, xmax=50)
    plt.ylim(ymin=-50, ymax=50)
    
    pixels_per_meter = 10
    
    if format == 'se_pts':
        for pts, label in zip(all_se_pts, labels):
            vec = pts.reshape(-1, 2)
            
            pts_x = vec[:, 0]
            pts_y = vec[:, 1]

            # lidar2depth
            # pts are in nuscene lidar coord, final bev is in mmdet3d depth coord
            # if ds.dataset.to_mmdet3d_lidar is None, then the data is facked in mmdet3d lidar coord
            # to render these mmdet3d points in lidar coord to depth coord, we need lidar2depth
            pts_x, pts_y = -pts_y, pts_x       
            
            plt.quiver(pts_x[:-1], 
                        pts_y[:-1], 
                        pts_x[1:] - pts_x[:-1], 
                        pts_y[1:] - pts_y[:-1], 
                        scale_units='xy', 
                        angles='xy', 
                        scale=1, 
                        color=colors[label])
    elif format == 'fixed_num_pts':
        for pts, label in zip(all_pts, labels):
            pts = pts.reshape(-1, 2)
            pts_x, pts_y = pts[:, 0], pts[:, 1]
    
            # lidar2depth
            # pts are in nuscene lidar coord, final bev is in mmdet3d depth coord
            # if ds.dataset.to_mmdet3d_lidar is None, then the data is facked in mmdet3d lidar coord
            # to render these mmdet3d points in lidar coord to depth coord, we need lidar2depth
            pts_x, pts_y = -pts_y, pts_x        
            
            axes.plot(pts_x, pts_y, color=colors[label], linewidth=1, alpha=0.8, zorder=-1)
            axes.scatter(pts_x, pts_y, color=colors[label], s=4, alpha=0.8, zorder=-1)    
    elif format == 'bbox':
        for box, label in zip(all_boxes, labels):
            pts_x, pts_y = box[0], box[1]
                        
            width = box[2] - box[0]
            height = box[3] - box[1]
            axes.add_patch(
                plt.Rectangle((pts_x, pts_y), width, height,
                    linewidth=0.4, edgecolor=colors[label], facecolor='none', alpha=0.8))

    elif format == 'polyline':
        for poly, label in zip(all_polys, labels):
            pts = np.array(list(poly.coords))
            pts_x, pts_y = pts[:, 0], pts[:, 1]
            
            # lidar2depth
            # pts are in nuscene lidar coord, final bev is in mmdet3d depth coord
            # if ds.dataset.to_mmdet3d_lidar is None, then the data is facked in mmdet3d lidar coord
            # to render these mmdet3d points in lidar coord to depth coord, we need lidar2depth
            pts_x, pts_y = -pts_y, pts_x
            
            
            plt.plot(pts_x, pts_y, color=colors[label], linewidth=1, alpha=0.8, zorder=-1)
            plt.scatter(pts_x, pts_y, color=colors[label], s=4, alpha=0.8, zorder=-1)
    
    # this setting can work, but fig size is not controlled
    plt.axis('off')
    plt.savefig('map.png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    
    # agents: kitti to lidar
    instances = data_samples[0].gt_instances_3d
    bboxes_3d = instances.bbox
    
    # ego
    ego_size = data_samples[0].metainfo['ego_size']
    # nuscene lidar coordinate
    ego_box = torch.tensor([[0, 0, 0, ego_size[0], ego_size[1], ego_size[2], np.pi/2, 0, 0]])
    
    # bev box need to be draw on the mmdet3d lidar coordinate
    # if no transformation is provided, we assume the original coordinate is
    # nuscenes lidar coordinate
    if ds.dataset.to_mmdet3d_lidar is None:
        ego_box = LiDARInstance3DBoxes(ego_box, box_dim=9)
    
    else:
        # nuscene lidar coord is the same as mmdet3d depth coord
        ego_box = DepthInstance3DBoxes(ego_box, box_dim=9)
    
    # display the data
    #bev = cv2.imread('map.png')
    bev = img
    #bev = 255 * np.ones((1000, 1000, 3), dtype=np.uint8)
    bev = cv2.flip(bev, 0) # flip the image saved by matplotlib
    print(bev.shape)
    
    vis.set_image(bev, origin='lower')
    vis.draw_bboxes_on_bev(bbox_3d_ego = ego_box,
                        bboxes_3d_instances = bboxes_3d,
                        scale=pixels_per_meter)
    
    return vis.get_image()


def draw_map1(data_inputs, data_samples, vis):
    # map data
    data_sample = data_samples[0]
    gt_map = data_sample.gt_map_vectors
    
    formats = ['se_pts', 'fixed_num_pts', 'bbox', 'polyline']
    format = formats[1]
    

    
    
    classes = ['divider', 'ped_crossing', 'boundary']
    colors = ['cornflowerblue', 'royalblue', 'slategrey']
    vecs = gt_map.pt 
    labels = gt_map.label.cpu().numpy()
    
    # (num_boxes, num_points, 2)
    all_pts = vecs.fixed_num_sampled_points.to('cpu').numpy()

    #all_pts = all_pts.reshape(len(labels), -1, 2)
    
    # (num_boxes, 4)
    all_se_pts = vecs.start_end_points.to('cpu').numpy()
    all_boxes = vecs.bbox.to('cpu').numpy()
    all_polys = vecs.instance_list

    pixels_per_meter = 10
    
    
    """
    if format == 'se_pts':
        for pts, label in zip(all_se_pts, labels):
            vec = pts.reshape(-1, 2)
            
            pts_x = vec[:, 0]
            pts_y = vec[:, 1]

            # lidar2depth
            # pts are in nuscene lidar coord, final bev is in mmdet3d depth coord
            # if ds.dataset.to_mmdet3d_lidar is None, then the data is facked in mmdet3d lidar coord
            # to render these mmdet3d points in lidar coord to depth coord, we need lidar2depth
            pts_x, pts_y = -pts_y, pts_x       
            
            plt.quiver(pts_x[:-1], 
                        pts_y[:-1], 
                        pts_x[1:] - pts_x[:-1], 
                        pts_y[1:] - pts_y[:-1], 
                        scale_units='xy', 
                        angles='xy', 
                        scale=1, 
                        color=colors[label])
    elif format == 'fixed_num_pts':
        for pts, label in zip(all_pts, labels):
            pts = pts.reshape(-1, 2)
            pts_x, pts_y = pts[:, 0], pts[:, 1]
    
            # lidar2depth
            # pts are in nuscene lidar coord, final bev is in mmdet3d depth coord
            # if ds.dataset.to_mmdet3d_lidar is None, then the data is facked in mmdet3d lidar coord
            # to render these mmdet3d points in lidar coord to depth coord, we need lidar2depth
            pts_x, pts_y = -pts_y, pts_x        
            
            # scale the points to pixels
            pts_x *= pixels_per_meter
            pts_y *= pixels_per_meter
            pts_x += width // 2
            pts_y += height // 2
            
            vis.draw_points(np.stack([pts_x, pts_y], axis=1),
                            colors=colors[label],
                            sizes=4)
            
            vis.draw_lines(np.stack([pts_x[:-1], pts_x[1:]], axis=1),
                            np.stack([pts_y[:-1], pts_y[1:]], axis=1),
                            colors=colors[label],
                            line_widths=1)
            
    elif format == 'bbox':
        pass
            
    elif format == 'polyline':
        for poly, label in zip(all_polys, labels):
            pts = np.array(list(poly.coords))
            pts_x, pts_y = pts[:, 0], pts[:, 1]
            
            # lidar2depth
            # pts are in nuscene lidar coord, final bev is in mmdet3d depth coord
            # if ds.dataset.to_mmdet3d_lidar is None, then the data is facked in mmdet3d lidar coord
            # to render these mmdet3d points in lidar coord to depth coord, we need lidar2depth
            pts_x, pts_y = -pts_y, pts_x
            
            
            plt.plot(pts_x, pts_y, color=colors[label], linewidth=1, alpha=0.8, zorder=-1)
            plt.scatter(pts_x, pts_y, color=colors[label], s=4, alpha=0.8, zorder=-1)
    """
    
    vis.draw_vector_map(
        all_pts,
        labels,
        ds_cfg.point_cloud_range,
        pixels_per_meter=pixels_per_meter,
        map_format=format
    )
    
    # this setting can work, but fig size is not controlled
    #plt.axis('off')
    #plt.savefig('map.png', dpi=100, bbox_inches='tight', pad_inches=0)
    #plt.close()
    img = vis.get_image()
    
    # agents: kitti to lidar
    instances = data_samples[0].gt_instances_3d
    bboxes_3d = instances.bbox
    
    # ego
    ego_size = data_samples[0].metainfo['ego_size']
    # nuscene lidar coordinate
    ego_box = torch.tensor([[0, 0, 0, ego_size[0], ego_size[1], ego_size[2], np.pi/2, 0, 0]])
    
    # bev box need to be draw on the mmdet3d lidar coordinate
    # if no transformation is provided, we assume the original coordinate is
    # nuscenes lidar coordinate
    if ds.dataset.to_mmdet3d_lidar is None:
        ego_box = LiDARInstance3DBoxes(ego_box, box_dim=9)
    
    else:
        # nuscene lidar coord is the same as mmdet3d depth coord
        ego_box = DepthInstance3DBoxes(ego_box, box_dim=9)
    
    # display the data
    #bev = cv2.imread('map.png')
    #bev = img
    #bev = 255 * np.ones((1000, 1000, 3), dtype=np.uint8)
    # flip the image for lower origin
    #img = cv2.flip(img, 0) # flip the image 
    #vis.set_image(img, origin='lower')
    vis.draw_bboxes_on_bev(bbox_3d_ego = ego_box,
                        bboxes_3d_instances = bboxes_3d,
                        scale=pixels_per_meter)
    
    return vis.get_image()
    
    
init_default_scope('fsd')
ds_cfg = Config.fromfile('fsd/configs/datasets/nuscenes_vad.py')
ds = Runner.build_dataloader(ds_cfg.test_dataloader)



vis_cfg = Config(dict(
    type='PlanningVisualizer',
    _scope_ = 'fsd',
    save_dir='./temp_dir',
    image_mode='rgb' if ds_cfg.to_rgb else 'bgr',
    vis_backends=[dict(type='LocalVisBackend')],
    name='vis')
)
vis = VISUALIZERS.build(vis_cfg) 
vis.dataset_meta = ds.dataset.metainfo

for i, item in enumerate(ds):
    data_inputs = item['inputs']
    data_samples = item['data_samples']
    data_samples = [_convert_kitti_to_mmdet3d(data_sample) for data_sample in data_samples]
    
    ## draw images
    #img = draw_boxes_3d_on_image(data_inputs, data_samples, vis)
    #img = draw_bev_bboxes(data_inputs, data_samples, vis, ds)
    #img = draw_trajectory_on_image(data_inputs, data_samples, vis)
    #img = draw_multiviews(data_inputs, data_samples, vis)
    #img = draw_trajectory_on_bev(data_inputs, data_samples, vis, ds)
    #img = draw_mutimodal_trajectory_on_bev(data_inputs, data_samples, vis, ds)
    #img = draw_mutimodal_trajectory_on_image(data_inputs, data_samples, vis)
    #img = draw_map(data_inputs, data_samples, vis)
    
    #img = draw_map1(data_inputs, data_samples, vis)
    
    
    # resize the image
    #img = cv2.resize(img, (1000, 1000))
    #backend = 'cv2'#'matplotlib' # cv2
    #if backend == 'matplotlib' and vis.image_mode == 'bgr':
    #    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #elif backend == 'cv2' and vis.image_mode == 'rgb':
    #    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    
    
    #vis.show(drawn_img=img, wait_time=-1, backend=backend) # cv2 uses bgr

    
    ## draw point cloud
    #draw_boxes_on_point_cloud(data_inputs, data_samples, vis)
    #vis.show(wait_time=-1) 
    cam_names = data_samples[0].metainfo['img_names']
    front_cam_idx = cam_names.index('CAM_FRONT')

    view_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
    

    for b, data_sample in enumerate(data_samples):
        data_input = {}
        data_input['img'] = data_inputs['img'][b]
        data_input['points'] = data_inputs['points'][b]
        
        data_sample = data_samples[b]
        vis.add_datasample(
            name='test',
            data_input = data_input,
            data_sample = data_sample,
            draw_gt=True,
            draw_pred=False,
            show=True,
            wait_time=0.05,
            step=b,
            vis_task='multi-modality_planning',
            show_pcd_rgb=False,
            multi_view_names=view_names,
            pcd_range=ds_cfg.point_cloud_range,
            map_format='fixed_num_pts',
            map_colors=['cornflowerblue', 'royalblue', 'slategrey'],
            pixels_per_meter=10,
            to_mmdet3d_lidar=ds.dataset.to_mmdet3d_lidar,
        )

    print('Press any key to continue...')

    
"""
from fsd.visualization import PlanningVisualizer
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.datasets import LoadPointsFromFile

points = np.fromfile('./data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin', dtype=np.float32)
print(points.shape)
points1 = points.reshape(-1, 5)[:, :3]

# original lidar points are in nuscenes lidar coordinate
lf = LoadPointsFromFile(
    coord_type='DEPTH',
    load_dim=5,
    use_dim=[0, 1, 2])


info = {'lidar_points': 
            {'lidar_path': './data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin'}}

info = lf.transform(info)
points2 = info['points'].tensor.numpy()

#visualizer = Det3DLocalVisualizer()
visualizer = PlanningVisualizer()

# set point cloud in visualizer: 
visualizer.set_points(points2, pcd_mode=2) # 0: lidar, 1: cam mode 2: depth
#bboxes_3d = LiDARInstance3DBoxes(
#    torch.tensor([[0, 0, 0, 4.2000, 1.4800, 1.8900,
#                   -1.5808]]))
# Draw 3D bboxes
#visualizer.draw_bboxes_3d(bboxes_3d, bbox_color=[(0, 255, 0)])
visualizer.show()
"""