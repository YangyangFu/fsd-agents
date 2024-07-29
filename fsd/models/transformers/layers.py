from typing import List, Union
from fsd.utils import OptConfigType, ConfigType 

from mmengine.model import BaseModule
from mmcv.cnn.bricks.transformer import BaseTransformerLayer, TransformerLayerSequence
from fsd.registry import TRANSFORMERS

LayerConfigType = Union[List[ConfigType], ConfigType]

#register basic vision transformer to scope fsd
DETRLayer = BaseTransformerLayer
DETRLayerSequence = TransformerLayerSequence
TRANSFORMERS.register_module(module=DETRLayer, name='DETRLayer')
TRANSFORMERS.register_module(module=DETRLayerSequence, name='DETRLayerSequence')

# classes for deformable transformer
@TRANSFORMERS.register_module()
class DeformableDETRLayer(BaseModule):
    pass

@TRANSFORMERS.register_module()
class DeformableDETRLayerSequence(BaseModule):
    pass