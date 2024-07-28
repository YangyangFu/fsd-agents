from mmdet.models.backbones.resnet import ResNet, ResNetV1d

from fsd.registry import BACKBONES

BACKBONES.register_module(module=ResNet, name='ResNet')
BACKBONES.register_module(module=ResNetV1d, name='ResNetV1d')

