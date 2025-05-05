import os
import math
import copy
import argparse
from os import path as osp
from collections import OrderedDict
from typing import List, Tuple, Union
import numpy as np
from shapely.geometry import MultiPoint, box

import mmcv
import mmengine
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmdet3d.structures.ops.box_np_ops import points_cam2img
from mmdet3d.datasets.convert_utils import NuScenesNameMapping

# future and past traj
from nuscenes.prediction import PredictHelper
from tools.dataset_converters.update_infos_to_v2 import update_nuscenes_infos
from tools.dataset_converters.create_gt_database import create_groundtruth_database

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

# https://en.wikipedia.org/wiki/Renault_Zoe
ego_width, ego_length, ego_height = 1.730, 4.084, 1.562

FPS = 2 # frames per second

def create_nuscenes_infos(root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    print(version, root_path)
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    nusc_can_bus = NuScenesCanBus(dataroot=root_path)
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_can_bus, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_infos_test.pkl'.format(info_prefix))
        mmengine.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_infos_train.pkl'.format(info_prefix))
        mmengine.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(root_path,
                                 '{}_infos_val.pkl'.format(info_prefix))
        mmengine.dump(data, info_val_path)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmengine.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _get_can_bus_info(nusc, nusc_can_bus, sample):
    """
    Ref: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/can_bus/README.md
    
    (x, y, z, qx, qy, qz, qw, ax, ay, az, rx, ry, rz, vx, vy, vz, steering, throttle, brake) 
    - x, y, z: position in world frame, in m
    - qx, qy, qz, qw: ego frame orientation
    - ax, ay, az: acceleration in ego vehicle frame, in m/s^2
    - rx, ry, rz: angular velocity in ego vehicle frame, in rad/s
    - vx, vy, vz: velocity in ego vehicle frame, in m/s
    - steering: steering angle in radian, positive means turn left
    - throttle: throttle in [0, 1]
    - brake: brake in [0, 1]
    
    Note the values may be inaccurate because of IMU sensor noise.
    Usually one can overwrite with calibrated sensor data.
    
    """
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    map_location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']
    
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
        steer_list = nusc_can_bus.get_messages(scene_name, 'steeranglefeedback')
        vehicle_monitor_list = nusc_can_bus.get_messages(scene_name, 'vehicle_monitor')
    except:
        return np.zeros(19)  # server scenes do not have can bus information.
    
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    # get the can bus information
    # (x, y, z, qx, qy, qz, qw, ax, ay, az, rx, ry, rz, vx, vy, vz)
    for key in ['pos', 'orientation', 'accel', 'rotation_rate', 'vel']:
        can_bus.extend(last_pose[key])  
    
    ## get control signals
    # (steering, throttle, brake)
    # get steering in radians: positive means turn left
    # [-7.7, 6.3]
    last_steer = steer_list[0]
    for i, steer in enumerate(steer_list):
        if steer['utime'] > sample_timestamp:
            break
        last_steer = steer
    steer = last_steer['value']
    # flip x axis if in left hand traffic, e.g., singapore
    left_hand_traffic = True if 'singapore' in map_location else False
    if left_hand_traffic:
        steer = -steer
    can_bus.extend([steer])
    
    # NOTE: this may cause issue because the sampling freq of vehicle monitor is only 2Hz
    # throttle and brake
    last_veh = vehicle_monitor_list[0]
    for i, veh in enumerate(vehicle_monitor_list):
        if veh['utime'] > sample_timestamp:
            break
        last_veh = veh
    throttle = last_veh['throttle'] / 1000. # nomalize to [0, 1]
    brake = last_veh['brake'] / 126. # normalize to [0, 1] 
    can_bus.extend([throttle, brake])
    
    return np.array(can_bus)


def _fill_trainval_infos(nusc,
                         nusc_can_bus,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10,
                         fut_ts=6,
                         his_ts=4):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    # helper for future prediction
    nusc_helper = PredictHelper(nusc)
    
    train_nusc_infos = []
    val_nusc_infos = []

    cat2idx = {}
    for idx, dic in enumerate(nusc.category):
        cat2idx[dic['name']] = idx

    for sample in mmengine.track_iter_progress(nusc.sample):
        map_location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        if sample['prev'] != '':
            sample_prev = nusc.get('sample', sample['prev'])
            sd_rec_prev = nusc.get('sample_data', sample_prev['data']['LIDAR_TOP'])
            pose_record_prev = nusc.get('ego_pose', sd_rec_prev['ego_pose_token'])
        else:
            pose_record_prev = None
        if sample['next'] != '':
            sample_next = nusc.get('sample', sample['next'])
            sd_rec_next = nusc.get('sample_data', sample_next['data']['LIDAR_TOP'])
            pose_record_next = nusc.get('ego_pose', sd_rec_next['ego_pose_token'])
        else:
            pose_record_next = None

        ## nuscene lidar coord
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmengine.check_file_exist(lidar_path)
        fut_valid_flag = True
        test_sample = copy.deepcopy(sample)
        for i in range(fut_ts):
            if test_sample['next'] != '':
                test_sample = nusc.get('sample', test_sample['next'])
            else:
                fut_valid_flag = False
        ##
        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'prev': sample['prev'],
            'next': sample['next'],
            'sweeps': [],
            'cams': dict(),
            'scene_token': sample['scene_token'],  # temporal related info
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'fut_valid_flag': fut_valid_flag,
            'map_location': map_location
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        # camera to lidar
        for cam in camera_types:
            cam_token = sample['data'][cam]
            # this could save time compared with the original code
            cam_intrinsic = np.array(nusc.get('calibrated_sensor', 
                                              nusc.get('sample_data', cam_token)['calibrated_sensor_token'])['camera_intrinsic']
                                    )
            #cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        # previous lidar to current lidar
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        # obtain annotation
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesNameMapping:
                    names[i] = NuScenesNameMapping[names[i]]
            names = np.array(names)
            
            # box sample annotation token: list[str]
            box_annotation_tokens = [anno['token'] for anno in annotations]
            # box instance token: list[str]: unique id for tracking
            box_tokens = [anno['instance_token'] for anno in annotations]
            
            # we need to convert rot to SECOND lidar format.
            # SECOND lidar yaw is left-handed definition. This is still in 
            # Nuscenes lidar coord system but the box definition is following SECOND format.
            ##!! change the rot format will break all checkpoint, so...
            #gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            
            # to mmdet3d box format (x, y, z, dx, dy, dz, yaw), dx is heading direction
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            
            # get future coords for each box
            # [num_box, fut_ts*2]
            num_box = len(boxes)
            gt_fut_trajs = np.zeros((num_box, fut_ts, 3))
            gt_fut_yaw = np.zeros((num_box, fut_ts))
            gt_fut_masks = np.zeros((num_box, fut_ts))
            gt_fut_goal = np.zeros((num_box, 3))
            # agents history and future
            for i, anno in enumerate(annotations):
                # get future annos for instance
                future_annos = nusc_helper.get_future_for_agent(
                    instance_token=anno['instance_token'],
                    sample_token=anno['sample_token'],
                    seconds=fut_ts / FPS,
                    in_agent_frame=False,
                    just_xy=False,
                )
                prev_box = boxes[i]
                for j in range(len(future_annos)):
                    # get future box
                    fut_box = Box(future_annos[j]['translation'],
                                  future_annos[j]['size'],
                                  Quaternion(future_annos[j]['rotation']))
                    # Move box to ego vehicle coord system.
                    fut_box.translate(-np.array(pose_record['translation']))
                    fut_box.rotate(Quaternion(pose_record['rotation']).inverse)
                    #  Move box to sensor coord system: top lidar coord in current frame.
                    fut_box.translate(-np.array(cs_record['translation']))
                    fut_box.rotate(Quaternion(cs_record['rotation']).inverse)
                    
                    # get future traj in difference 
                    gt_fut_trajs[i, j] = fut_box.center - prev_box.center
                    gt_fut_yaw[i, j] = fut_box.orientation.yaw_pitch_roll[0] - prev_box.orientation.yaw_pitch_roll[0]
                    gt_fut_masks[i, j] = 1

                    # advance
                    prev_box = fut_box
                    
                # end goal for each agent
                if fut_box:
                    gt_fut_goal[i, :] = fut_box.center - boxes[i].center
                
                #gt_fut_yaw_diff[i, j] = nusc_helper.get_heading_change_rate_for_agent(future_annos[j]['instance_token'], future_annos[j]['sample_token'])

            #######################################################
            ##          ego 
            #######################################################
            # ego size (l, w, h)
            ego_size = np.array([ego_length, ego_width, ego_height])
            
            # get ego history traj (offset format)
            ego_his_trajs_xyzr = _get_ego_trajectory_history(sample, nusc, his_ts)
            ego_his_trajs = ego_his_trajs_xyzr[1:] - ego_his_trajs_xyzr[:-1]
            # each row is valid if no nan for the row
            ego_his_mask = np.all(np.isfinite(ego_his_trajs), axis=1)
            # fill nan with 0
            ego_his_trajs[np.isnan(ego_his_trajs)] = 0

            # get ego futute traj 
            ego_fut_trajs_xyzr = _get_ego_trajectory_future(sample, nusc, fut_ts)
            # the last valid point is the goal
            for goal_idx in range(fut_ts, -1, -1):
                if np.all(ego_fut_trajs_xyzr[goal_idx, :] != np.nan):
                    break 
            ego_fut_goal = ego_fut_trajs_xyzr[goal_idx, :]
            ego_fut_trajs = ego_fut_trajs_xyzr[1:] - ego_fut_trajs_xyzr[:-1]
            ego_fut_mask = np.all(np.isfinite(ego_fut_trajs), axis=1)
            # fill nan with 0
            ego_fut_trajs[np.isnan(ego_fut_trajs)] = 0

            # drive command according to goal points
            if ego_fut_trajs_xyzr[goal_idx][0] >= 2:
                ego_command = np.array([1, 0, 0])  # Turn Right
            elif ego_fut_trajs_xyzr[goal_idx][0] <= -2:
                ego_command = np.array([0, 1, 0])  # Turn Left
            else:
                ego_command = np.array([0, 0, 1])  # Go Straight

            ### ego can bus
            # -----------------------------------------------
            can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)
            # update from calibrated data
            can_bus[0:3] = pose_record['translation']
            can_bus[3:7] = pose_record['rotation']
            
            
            # the calibrated sensor provides more accurate estimation of pose
            ego_yaw = quaternion_yaw(Quaternion(pose_record['rotation']))
            ego_pos = np.array(pose_record['translation'])
            if pose_record_prev is not None:
                ego_yaw_prev = quaternion_yaw(Quaternion(pose_record_prev['rotation']))
                ego_pos_prev = np.array(pose_record_prev['translation'])
            if pose_record_next is not None:
                ego_yaw_next = quaternion_yaw(Quaternion(pose_record_next['rotation']))
                ego_pos_next = np.array(pose_record_next['translation'])
            assert (pose_record_prev is not None) or (pose_record_next is not None), 'prev token and next token all empty'
            if pose_record_prev is not None:
                ego_w = (ego_yaw - ego_yaw_prev) / (1./FPS)
                ego_v = np.linalg.norm(ego_pos[:2] - ego_pos_prev[:2]) / (1./FPS)
                ego_vx, ego_vy = ego_v * math.cos(ego_yaw + np.pi/2), ego_v * math.sin(ego_yaw + np.pi/2)
            else:
                ego_w = (ego_yaw_next - ego_yaw) / (1./FPS)
                ego_v = np.linalg.norm(ego_pos_next[:2] - ego_pos[:2]) / (1./FPS)
                ego_vx, ego_vy = ego_v * math.cos(ego_yaw + np.pi/2), ego_v * math.sin(ego_yaw + np.pi/2)

            # velocity in world frame
            ego_velocity = np.array([ego_vx, ego_vy])
            ego_yaw_velocity = ego_w
            
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_ids'] = box_tokens
            info['gt_annotation_tokens'] = box_annotation_tokens
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag
            
            info['gt_boxes_fut_traj'] = gt_fut_trajs.astype(np.float32)
            info['gt_boxes_fut_mask'] = gt_fut_masks.astype(np.bool_)
            info['gt_boxes_fut_yaw'] = gt_fut_yaw.astype(np.float32)
            info['gt_boxes_fut_goal'] = gt_fut_goal.astype(np.float32)
            
            info['ego_size'] = ego_size.astype(np.float32)
            info['ego_fut_goal'] = ego_fut_goal.astype(np.float32)
            info['ego_velocity'] = ego_velocity.astype(np.float32)
            info['ego_yaw_velocity'] = ego_yaw_velocity.astype(np.float32)
            info['can_bus'] = can_bus.astype(np.float32)         
            info['gt_ego_his_traj'] = ego_his_trajs[:, :3].astype(np.float32)
            info['gt_ego_his_yaw'] = ego_his_trajs[:, 3].astype(np.float32)
            info['gt_ego_his_mask'] = ego_his_mask
            info['gt_ego_fut_traj'] = ego_fut_trajs[:, :3].astype(np.float32)
            info['gt_ego_fut_yaw'] = ego_fut_trajs[:, 3].astype(np.float32)
            info['gt_ego_fut_mask'] = ego_fut_mask
            info['gt_ego_fut_cmd'] = ego_command.astype(np.float32)

            
        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos

def _get_ego_trajectory_history(sample, nusc, steps):
    """Get ego trajectory history in current sample lidar frame
    """
    # initialzie with nan
    xyz = np.full((steps+1, 3), np.nan)
    r = np.full((steps+1, 1), np.nan)
    # current lidar pose
    lidar_curr2global = get_global_sensor_pose(sample, nusc, inverse=False)
    global2lidar_curr = np.linalg.inv(lidar_curr2global)
    
    # historical lidar pose
    lidar_adj2global = lidar_curr2global
    adj2curr = global2lidar_curr @ lidar_adj2global
    sample_prev = sample
    xyz[steps, :] = adj2curr[:3, 3]
    r[steps] = np.arctan2(adj2curr[1, 0], adj2curr[0, 0]) # yaw in [-pi, pi]
    
    for i in range(steps-1, -1, -1):                    
        if sample_prev['prev'] == '':
            break
        else:
            sample_prev = nusc.get('sample', sample_prev['prev'])
            lidar_adj2global = get_global_sensor_pose(sample_prev, nusc, inverse=False)
            adj2curr = global2lidar_curr @ lidar_adj2global
            xyz[i, :] = adj2curr[:3, 3]
            r[i] = np.arctan2(adj2curr[1, 0], adj2curr[0, 0]) # yaw in [-pi, pi]
    xyzr = np.concatenate((xyz, r), axis=1)
    
    return xyzr

def _get_ego_trajectory_future(sample, nusc, steps):
    """Get ego future trajectory in current sample lidar frame
    """
    # initialzie with nan
    xyz = np.full((steps+1, 3), np.nan)
    r = np.full((steps+1, 1), np.nan)
    # current lidar pose
    lidar_curr2global = get_global_sensor_pose(sample, nusc, inverse=False)
    global2lidar_curr = np.linalg.inv(lidar_curr2global)
    
    # future lidar pose
    lidar_adj2global = lidar_curr2global
    adj2curr = global2lidar_curr @ lidar_adj2global
    sample_next = sample
    xyz[0, :] = adj2curr[:3, 3]
    r[0] = np.arctan2(adj2curr[1, 0], adj2curr[0, 0]) # yaw in [-pi, pi]
    
    for i in range(1, steps+1):                    
        if sample_next['next'] == '':
            break
        else:
            sample_next = nusc.get('sample', sample_next['next'])
            lidar_adj2global = get_global_sensor_pose(sample_next, nusc, inverse=False)
            adj2curr = global2lidar_curr @ lidar_adj2global
            xyz[i, :] = adj2curr[:3, 3]
            r[i] = np.arctan2(adj2curr[1, 0], adj2curr[0, 0]) # yaw in [-pi, pi]
    xyzr = np.concatenate((xyz, r), axis=1)
    
    return xyzr
            
def get_global_sensor_pose(rec, nusc, inverse=False):
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])

    sd_ep = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
    if inverse is False:
        global_from_ego = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False)
        ego_from_sensor = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
        # translation equivalent writing
        # pose_translation = np.array(sd_cs["translation"])
        # rot_mat = Quaternion(sd_ep['rotation']).rotation_matrix
        # pose_translation = np.dot(rot_mat, pose_translation)
        # # pose_translation = pose[:3, 3]
        # pose_translation = pose_translation + np.array(sd_ep["translation"])
    else:
        sensor_from_ego = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True)
        ego_from_global = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True)
        pose = sensor_from_ego.dot(ego_from_global)
    return pose

def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    # sensor2ego
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    # ego2global
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    # Tlidar2sensor = Tego2sensor@Tglobal2ego@Tego2global@Tlidar2ego
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=False):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool): Whether to export mono3d annotation. Default: False.
    """
    # get bbox annotations for camera
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    nusc_infos = mmengine.load(info_path)['infos']
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    # info_2d_list = []
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmengine.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(
                nusc,
                cam_info['sample_data_token'],
                visibilities=['', '1', '2', '3', '4'],
                mono3d=mono3d)
            (height, width, _) = mmengine.imread(cam_info['data_path']).shape
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['data_path'].split('data/nuscenes/')
                    [-1],
                    id=cam_info['sample_data_token'],
                    token=info['token'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=info['ego2global_rotation'],
                    ego2global_translation=info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmengine.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(nusc,
                 sample_data_token: str,
                 visibilities: List[str],
                 mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera \
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name']
            attr_id = nus_attributes.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various informations on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): flie name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in NuScenesNameMapping:
        return None
    cat_name = NuScenesNameMapping[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec
