# Wrapper for naive transformer model
from typing import List, Dict, Tuple, Union
import torch
import torch.nn as nn

from mmcv.cnn import build_transformer_layer_sequence

from mmdet.utils import ConfigType, OptConfigType

class NaiveTransformer(nn.Module):
    def __init__(self, 
                 encoder: List[ConfigType],
                 decoder: List[ConfigType],
                 init_cfg: OptConfigType = None
                 ) -> None:
        super(NaiveTransformer, self).__init__(init_cfg = init_cfg)
        
        self._init_layers()
        
    
    def _init_layers(self) -> None:
        """Initialize encoder and decoder
        """
        self.encoder = build_transformer_layer_sequence(self.encoder_cfg)
        self.decoder = build_transformer_layer_sequence(self.decoder_cfg)
        self.embed_dims = self.encoder.embed_dims
        self.num_layers_encoder = self.encoder.num_layers
        self.num_layers_decoder = self.decoder.num_layers
    
    def forward(self, 
                src_seq: torch.Tensor,
                tgt_seq: torch.Tensor,):
        
        # encoder
        key, = value = self.encoder(src_seq)
        
