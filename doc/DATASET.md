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
    - gt_boxes -> boxes for each npc instance including traffic sign, traffic light, vehicle, etc, (N, 9)
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

