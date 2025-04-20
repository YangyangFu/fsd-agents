# Dataset
Agent's pose in world is the transformation from agent to world.

For the definition of coordinate system for different datasets, please refer to [here](./COORDINATE_SYSTEM.md).




## QUESTIONS

- Q1: For nuscene dataset in MMDET3D, the following data processing procedures are performed when formating a MMDET3D dataset:
    1. downlown original nuscene dataset
    2. use `create_data.py` provided by mmdet3d to process the annotation file and raw data to mmdet3d format, which generates a a few *.pkl files
        - the ground truth bboxes are in global frame as descrubed [here](https://github.com/nutonomy/nuscenes-devkit/tree/master/python-sdk/nuscenes/eval/detection), are they processed into lidara frame at this step?
    
            - **YES**. The ground truth bboxes in global frame are processed into lidar coordinate as shown [here](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/tools/dataset_converters/nuscenes_converter.py#L174). Box.wlh is for x, y ,z axis, thus in this box convention, x is right, y is front, and z is up -> Nuscenes Lidar coordinate.

    3. the mmdet3d dataset then is built on these *.pkl files and raw data


- Q2: How is BEVFormer deal with BEV targets/labels? The original 3d bboxes annotation if for the whole view. If some object is outside of BEV grid as defined by the `bev_size`, does BEVformer filter out these targets?
    - I posted a question [here](https://github.com/fundamentalvision/BEVFormer/issues/275), and I think they just use all the ground truth labels as their BEV size is similar to the lidar range.