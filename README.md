# FSDagents
Full self-driving agents benchmark on closed-loop simulation


## TODOS
- [x] architecture design with heads
    - maybe consider waypoint heads, heat map head, classification head etc
- [x] add transformer-based sensor fusion benchmarks
- [] add inverse reinforcement learning benchmarks
- [] add LLM benchmark

**2024-08**:
- [x] training pipeline works
- [x] InterFuser uses focus view from front camera image. the resulting image size is smaller than other images, how to bundle this for stacking images before going into the model
    - focus view is padded to other shapes during pipeline
    - need remove paddings before extracting features
- [x] address warnings during training, e.g., init_weights()
- [x] L1 loss with mask for waypoint loss
- [x] move pts to histogram into data preprocessor
- [x] add TASK_UTILS registry
- [x] Interfuser head/nect should rename to generic head/neck or TASKS_UTILS
- [x] batch first for all data and model inputs. each module/head/neck can have their own batch_first definition to be compatible with called torch modules or control output shape
    - currently batch first is used to when needed by a built-in torch module such as GRU, do we assume batch_first everywhere?
- [x] check model parameters: trainable/nontrainable. The original has 52935567 in total
    - [x] [here](https://github.com/facebookresearch/detectron2/blob/543fd075e146261c2e2b0770c9b537314bdae572/detectron2/utils/analysis.py#L63-L65) shows the use of dicts in the inputs for complexity analysis. `get_complexity_info` can be used but need support list of tensors as inputs instead of dictionaries.
    - [x] small mismatch of model parameters compared with official models.
- [] how to control data type globally?
- [] data time is too long, 80-90% of total time
- [] add base planning module
- [] closed-loop evaluation code on carla sim environment based on carla leaderboard 2.0
- [] visualize multi-view/lidar data in data set
- [] visualization of prediction details during closed-loop simulation
  
