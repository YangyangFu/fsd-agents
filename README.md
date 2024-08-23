# FSDagents
Full self-driving agents benchmark on closed-loop simulation


## TODOS
- [x] architecture design with heads
    - maybe consider waypoint heads, heat map head, classification head etc
- [x] add transformer-based sensor fusion benchmarks
- [] add inverse reinforcement learning benchmarks
- [] add LLM benchmark

**2024-08**:
- InterFuser uses focus view from front camera image. the resulting image size is smaller than other images, how to bundle this for stacking images before going into the model