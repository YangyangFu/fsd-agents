from mmengine.runner import Runner
from fsd.registry import RUNNERS 

RUNNERS.register_module(name='Runner', module=Runner)
