# VAD: Vectorized Scene Representation for Efficient Autonomous Driving


## Prepare Dataset

```bash
python tools/data_converters/nuscenes_converter_vad.py nuscenes --root-path ./data/nuscenes-v1.0/mini --out-dir ./data/nuscenes-v1.0/mini --extra-tag vad_nuscenes --version v1.0-mini --canbus ./data/nuscenes-v1.0
```

This will prepare the data into `*pkl` files, and then update the `*pkl` files to comply with mmengine Dataset v2.