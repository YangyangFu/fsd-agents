# VAD: Vectorized Scene Representation for Efficient Autonomous Driving


## Prepare Dataset

```bash
python tools/data_converters/nuscenes_converter_vad.py nuscenes --root-path ./data/nuscenes-v1.0/mini --out-dir ./data/nuscenes-v1.0/mini --extra-tag vad_nuscenes --version v1.0-mini --canbus ./data/nuscenes-v1.0
```

This will prepare the data into `*pkl` files, and then update the `*pkl` files to comply with mmengine Dataset v2.


## Dataset Process


### Bugs in Original VAD Dataset

1. Original data processing code for `agent_lcf_feat` (i.e., agent local contextual feature) is not correct. For each agent, the feature is a vector of size 9.
   - x, y: position in Nuscenes lidar coordinate
   - yaw: yaw in radians in Nuscenes lidar coordinate following Nuscenes yaw convention
   - vx, vy: velocity in Nuscenes lidar coordinate
   - w, l, h: agent box width, length, height in meters 
   - `label`: agent label 
  
    The `label` uses the orignal [Nuscenes label](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_nuscenes.md), which contains 23 classes. However, the modeling training code uses 10 classes by using a mapping table. Thus, this inconsistency should be fixed for agent local contextual feature. 

2. When generating the `*.pkl` files, the yaw of box calculation is wrong in their code, see [here](https://github.com/hustvl/VAD/issues/105). They used the following code to calculate the yaw of the box:

```python
def quart_to_rpy(qua):
    # NOTE: shouldn't be w,x,y,z in quaternion
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw
```

   - The above code is wrong. The correct code should be:

```python
def quart_to_rpy(qua):
    w, x, y, z = qua
    roll = math.atan2(2 * (w * x - y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y + x * z))
    yaw = math.atan2(2 * (w * z - x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw
```
3. `can_bus` in original VAD dataset is not correct. See [here](https://github.com/hustvl/VAD/pull/89).