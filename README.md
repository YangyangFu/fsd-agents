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
- [] InterFuser uses focus view from front camera image. the resulting image size is smaller than other images, how to bundle this for stacking images before going into the model
    - focus view is padded to other shapes during pipeline
    - need remove paddings before extracting features
- [] address warnings during training, e.g., init_weights()
- [x] L1 loss with mask for waypoint loss
- [] move pts to histogram into data preprocessor
- [] add TASK_UTILS registry
- [] Interfuser head/nect should rename to generic head/neck or TASKS_UTILS
- [] data time is too long, 80-90% of total time
- [x] batch first for all data and model inputs. each module/head/neck can have their own batch_first definition to be compatible with called torch modules or control output shape
    - currently batch first is used to when needed by a built-in torch module such as GRU, do we assume batch_first everywhere?
- [] add base planning module