# Dataset

## Carla

### Annotation file

The following keys are stored in the annotation file as `*.pkl`
```
    - frame_idx -> current frame idx
    - folder -> data folder for current sample, e.g., `v1/DynamicObjectCrossing_TownXX_RouteXX_WeatherXX`
    - town_name -> town name, e.g., TownXX
    - command_far_xy -> farther waypoint in world coordinates
    - command_far -> command to farther waypoint
    - command_near_xy -> nearby waypoint in world coordinates
    - command_near -> command to nearby command
    - ego_yaw -> in radians. carla world coordinate or left-handed system world??
    - ego_translation -> 3D
    - ego_vel -> 3D
    - ego_accel -> 3D
    - ego_rotation_rate ->3D
    - ego_size -> 3D
    - world2ego -> (4, 4)
    - brake -> [0, 1]
    - throttle -> [0, 1]
    - steer -> [0, 1]
    - gt_ids -> id of instances in the scence (multi view camera, or lidar or combined?) (N,)
    - gt_boxes -> boxes in XXX coordinates for each npc instance including traffic sign, traffic light, vehicle, etc, (N, 9) with (3d center in lidar coord, 3d size (extent x 2), yaw in lidar coord, speed x, speed y)
    - gt_names -> class name in Carla for each instance, (N, )
    - num_points -> number of lidar points hitting on the bounding box. used to filter invisible instance.
    - npc2world -> transformation matrix for npc (N, 4, 4)
    - sensors
        - CAM_XXX
            - cam2ego
            - instrinsic
            - world2cam
            - data_path

        - LIDAR_TOP
            - lidar2ego
            - world2lidar
```

In the dataset, we need process the annotation file to get the desired annotations for the algorithms.


Before going into transformation pipeline, the data are preprocessed to have the following fields:

**NOTE:** there is no camera class in `cam_instrinsic` to relate the matrix to correspoinding camera.

```
- folder
- scene_token 
- frame_idx
- ego_yaw 
- ego_translation
- sensors
- world2lidar
- gt_ids
- gt_boxes 
- gt_names
- ego_vel
- ego_accel
- ego_rotation_rate
- npc2world
- gt_lane_labels
- gt_lane_bboxes
- gt_lane_masks
- timestamp
- img_filename
- lidar2img
- cam_intrinsic
- lidar2cam
- l2g_r_mat
- l2g_t
- ann_info
- can_bus
- occ_has_invalid_frame
- occ_img_is_valid
- occ_future_ann_infos
- occ_l2e_r_mats
- occ_l2e_t_vecs
- occ_e2g_r_mats
- occ_e2g_t_vecs
- sdc_planning
- sdc_planning_mask
- command
- img_fields: store key name related to img. empty in dataset
- bbox3d_fields
- pts_mask_fields
- pts_seg_fields
- bbox_fields
- mask_fields
- seg_fields
- box_type_3d
- box_mode_3d
```



## QUESTIONS

- Q1: For nuscene dataset in MMDET3D, the following data processing procedures are performed when formating a MMDET3D dataset:
    1. downlown original nuscene dataset
    2. use `create_data.py` provided by mmdet3d to process the annotation file and raw data to mmdet3d format, which generates a a few *.pkl files
        - the ground truth bboxes are in global frame as descrubed [here](https://github.com/nutonomy/nuscenes-devkit/tree/master/python-sdk/nuscenes/eval/detection), are they processed into lidara frame at this step?
    
            - **YES**. The ground truth bboxes in global frame are processed into lidar coordinate as shown [here](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/tools/dataset_converters/nuscenes_converter.py#L174) 

    3. the mmdet3d dataset then is built on these *.pkl files and raw data


- Q2: How is BEVFormer deal with BEV targets/labels? The original 3d bboxes annotation if for the whole view. If some object is outside of BEV grid as defined by the `bev_size`, does BEVformer filter out these targets?
    - I posted a question [here](https://github.com/fundamentalvision/BEVFormer/issues/275), and I think they just use all the ground truth labels as their BEV size is similar to the lidar range.