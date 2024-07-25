# Wrapper for naive transformer model
from typing import List, Dict, Tuple, Union
import torch

from mmengine.model import BaseModule
from mmengine.model.weight_init import xavier_init
from mmdet.utils import ConfigType, OptConfigType
from mmdet.models import (DetrTransformerDecoder, DetrTransformerEncoder)

# DETR transformer without head
# The existing DETR in mmdet is a full model with detection head
class NaiveDETR(BaseModule):
    """Implements the DETR transformer.

    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:

        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, 
                 encoder: ConfigType, 
                 decoder: ConfigType, 
                 init_cfg: OptConfigType = None) -> None:
        super(NaiveDETR, self).__init__(init_cfg=init_cfg)
        self.encoder = DetrTransformerEncoder(**encoder)
        self.decoder = DetrTransformerDecoder(**decoder)
        self.embed_dims = self.encoder.embed_dims

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, 
                src: torch.Tensor, 
                query_pos_embed: torch.Tensor, 
                key_pos_embed: torch.Tensor,
                key_padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function for `Transformer`.

        Args:
            src (Tensor): The input data of shape (bs, dim, h, w).
            query_pos_embed (Tensor): The positional encoding for query. 
                Shape [num_query, dim].
            key_pos_embed (Tensor): The positional encoding for key.
                Shape [bs, dim, h, w].
            key_padding_mask (Tensor): The padding mask for key.
                Shape [bs, h, w].

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        
        ## encoder forward 
        # [bs, num_query, dim]
        bs, hw, dim = src.shape
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        # src = src.view(bs, dim, -1).permute(0, 2, 1)  # [bs, dim, h, w] -> [bs, h*w, dim]
        #key_pos_embed = key_pos_embed.view(bs, dim, -1).permute(0, 2, 1)
        query_pos_embed = query_pos_embed.unsqueeze(0).repeat(
            bs, 1, 1)  # [num_query, dim] -> [bs, num_query, dim]
        #key_padding_mask = key_padding_mask.view(bs, -1)  # [bs, h, w] -> [bs, h*w]
        # [bs, num_keys, dim]
        memory = self.encoder(
            query=src,
            query_pos=key_pos_embed,
            key_padding_mask=key_padding_mask)
        
        ## decoder forward
        # target: [bs, num_query, dim]
        target = torch.zeros_like(query_pos_embed)
        # out_dec: [num_dec_layers, bs, num_query, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            query_pos=query_pos_embed,
            key_pos=key_pos_embed,
            key_padding_mask=key_padding_mask)

        return out_dec, memory
        

# add a main 
if __name__ == "__main__":
    from torch.nn.modules.activation import ReLU
    
    encoder_cfg = dict( # DetrTransformerEncoder
        num_layers=6,
        layer_cfg=dict(  # DetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type=ReLU, inplace=True)
            )
        )
    )        
    decoder_cfg = dict(  # DetrTransformerDecoder
        num_layers=8,
        layer_cfg=dict(  # DetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            ),
            cross_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            ),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type=ReLU, inplace=True)
            )
        ),
        return_intermediate=True
    )
    print(encoder_cfg)
    detr = NaiveDETR(encoder_cfg, decoder_cfg)

    # test forward
    src = torch.randn(1, 256, 200)
    query_pos_embed = torch.randn(100, 256)
    key_pos_embed = torch.randn(200, 256)
    key_padding_mask = torch.randn(1, 200)
    out_dec, memory = detr(src, query_pos_embed, key_pos_embed, key_padding_mask)
    # expect: out_dec: [8, 1, 100, 256], memory: [1, 200, 256]
    print(out_dec.shape, memory.shape) 
    
    