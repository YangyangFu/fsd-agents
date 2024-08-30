import torch

from timm.models import create_model
from timm.models.resnet import ResNet, resnet50d

#from fsd.registry import BACKBONES

cfg =dict(
    pretrained=True,
    out_indices=[4],
    features_only=True
)


#model = create_model('resnet50d', **cfg)

#model = BACKBONES.build(cfg)
#m = resnet50d(pretrained=True)
class ResNet50d_(torch.nn.Module):
    def __init__(self, pretrained=True, **kwargs):
        super(ResNet50d_, self).__init__()
        model = create_model('resnet50d', **cfg)
        print(dir(model))
        self.model = model

ResNet50d = ResNet50d_().model 
model = ResNet50d
print(model)
