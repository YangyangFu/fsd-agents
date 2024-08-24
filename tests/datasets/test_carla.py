from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.visualization import Visualizer
from fsd.runner import Runner

cfg = Config.fromfile('tests/datasets/_carla_config.py')
init_default_scope('fsd')
dl = Runner.build_dataloader(cfg.train_dataloader)

vis = Visualizer()
for sample in dl:
    print(sample.keys())
    print(sample['inputs']['img'][0][0].shape)
    vis.set_image(sample['inputs']['img'][0][0].numpy().transpose(1, 2, 0))
    vis.show(backend='cv2')
    print(sample['data_samples'])
    