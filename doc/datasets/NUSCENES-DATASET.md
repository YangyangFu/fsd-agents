# Nuscenes Dataset

## Raw Nuscences Dataset


## Convert to Planning Coordinate

**TODOS**:
- The NuScenes boxes in `mmdetection3d/tools/dataset_converters/nuscenes_converter.py` as [here](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/tools/dataset_converters/nuscenes_converter.py#L258) seem never changed from NuScenes Lidar coordinate to MMDet3D Lidar coordinate as claimed in [here](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/docs/en/advanced_guides/datasets/nuscenes.md?plain=1#L109).


**NOTE**
- usually different algorithms on the same dataset might process the data into different coordinate systems. To make that algorithm work without retraining, we need to convert the data into the coordinate system that the algorithm is trained on.