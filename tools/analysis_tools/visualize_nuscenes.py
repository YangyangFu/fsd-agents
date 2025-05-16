import os 
import cv2
import numpy as np
import torch 

from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmdet3d.structures import LiDARInstance3DBoxes, DepthInstance3DBoxes, limit_period, Box3DMode
from fsd.runner import Runner
from fsd.registry import DATASETS, VISUALIZERS


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
    
    # lidar2bev 
    lidar2bev = None
    if ds.dataset.to_mmdet3d_lidar is not None:
        lidar2bev = np.linalg.inv(ds.dataset.to_mmdet3d_lidar)
    
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
    
    # lidar2bev 
    lidar2bev = None
    if ds.dataset.to_mmdet3d_lidar is not None:
        lidar2bev = np.linalg.inv(ds.dataset.to_mmdet3d_lidar)
    
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



init_default_scope('fsd')
ds_cfg = Config.fromfile('fsd/configs/_base_/datasets/nuscenes.py')
#ds_cfg = Config.fromfile('fsd/configs/InterFuser/interfuser_r50_carla.py')
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

for i, item in enumerate(ds):
    data_inputs = item['inputs']
    data_samples = item['data_samples']
    
    ## draw images
    #img = draw_boxes_3d_on_image(data_inputs, data_samples, vis)
    #img = draw_bev_bboxes(data_inputs, data_samples, vis, ds)
    #img = draw_trajectory_on_image(data_inputs, data_samples, vis)
    #img = draw_multiviews(data_inputs, data_samples, vis)
    #img = draw_trajectory_on_bev(data_inputs, data_samples, vis, ds)
    #img = draw_mutimodal_trajectory_on_bev(data_inputs, data_samples, vis, ds)
    #img = draw_mutimodal_trajectory_on_image(data_inputs, data_samples, vis)
    
    #backend = 'matplotlib'#'matplotlib' # cv2
    #if backend == 'matplotlib' and vis.image_mode == 'bgr':
    #    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #elif backend == 'cv2' and vis.image_mode == 'rgb':
    #    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    
        
    #vis.show(drawn_img=img, wait_time=0.05, backend=backend) # cv2 uses bgr

    
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
        )

    print('Press any key to continue...')
    
"""
from mmdet3d.visualization import Det3DLocalVisualizer

points = np.fromfile('./data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin', dtype=np.float32)
print(points.shape)
points = points.reshape(-1, 5)[:, :3]
print(points.shape)
visualizer = Det3DLocalVisualizer()
# set point cloud in visualizer
visualizer.set_points(points, pcd_mode=2)
#bboxes_3d = LiDARInstance3DBoxes(
#    torch.tensor([[0, 0, 0, 4.2000, 3.4800, 1.8900,
#                   -1.5808]]))
# Draw 3D bboxes
#visualizer.draw_bboxes_3d(bboxes_3d)
visualizer.show()
"""