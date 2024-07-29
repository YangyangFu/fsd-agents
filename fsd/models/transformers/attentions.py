import torch
import torch.nn as nn 
import torch.nn.functional as F

from mmcv.cnn.bricks.transformer import MultiheadAttention
from fsd.registry import TRANSFORMERS

TRANSFORMERS.register_module(module=MultiheadAttention, name='MultiheadAttention')

