# Data Structure
The underlying data structure designed for automoumous motion planning and control tasks.

## TrajectoryData

```mermaid
classDiagram
    direction RL
    class TrajectoryData{
        +float: time
        +float: time_step
        +int: num_steps
        +int: past_steps
        +int: future_steps
        +Array: xyzr
        +Array: mask
    }
    class MultiModalTrajectoryData{
        +int: num_modalities
        +Array: scores
    }
    MultiModalTrajectoryData <|-- TrajectoryData
```

## PlanningDataSample
All annotations are in lidar coord

```mermaid
classDiagram
    direction RL

    class PlanningDataSample{
        +dict: metainfo
        +Ego: ego
        +Instances: instances
        +Grids: grids
        +VectorMap: map
    }
    class Ego{
        +dict: metainfo
        +Array: command
        +Array: goal_point
        +TrajectoryData: gt_traj
        +TrajectoryData: pred_traj
    }
    class Instances{
        +dict: metainfo
        +ArrayLike: ids
        +LiDAR3DBoxes: gt_bboxes_3d
        +LiDAR3DBoxes: pred_bboxes_3d
        +ArrayLike: gt_labels
        +ArrayLike: pred_labels
        +List[TrajectoryData]: gt_traj
        +List[TrajectoryData]: pred_traj
    }
    class VectorMap{

    }
    class Grids{

    }
    class BaseDataElement{

    }
    class InstanceData{

    }
    Ego <|.. BaseDataElement
    Instances <|.. InstanceData
    VectorMap <|.. BaseDataElement
    Grids <|.. BaseDataElement
    PlanningDataSample --o Ego
    PlanningDataSample --o Instances
    PlanningDataSample --o VectorMap
    PlanningDataSample --o Grids
```


```python
class PlanningDataSample(BaseDataElement):
    """
    Planning is usually performed with multi-modal sensors. The resulting annotation usually contains 3D bounding boxes from lidar annotation, which can then be transformed to various camera sensors if needed. 
    Here assume 3D annotation is available in Lidar coordinate (as in NuScene)

    The following attributes are reserved:
    
    metainfo:
        - timestamp
        - lidar_to_world
        - class_names


    DATA FIELD:
    gt_ego
        - goal_point
        - ego2world
        - future_trajectories
        - past_trajectories

    gt_instances_3d: InstanceData 
        - metainfo
            - xxx

        - bboxes: (LiDAR3DBoxes), 3d bounding boxes in lidar coordinate, with the option of including velocities
        - labels: (list or tensor), classification labels
        - ids: instance id used for tracking/instance seg
        - bboxes2world??? lidar2world?? -> TODO: do we need this??
        - past_trajectories: (Trajectories) in ego coordinate at current frame
        - future_trajectories: (Trajectories) in ego coordinate at current frame

    
    gt_pts_seg: PointData 


    MAP DATA
    gt_map
        - lane_labels
        - lane_bboxes
        - lane_mask

    """


```
