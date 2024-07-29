import warnings
import copy 

from typing import List, Union
from mmengine.config import ConfigDict
from fsd.utils import OptConfigType, ConfigType 

import torch 
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList, Sequential
from mmcv.cnn import Linear, build_norm_layer
from fsd.registry import TRANSFORMERS, MODELS
from mmengine.registry import MODELS as MMENGINE_MODELS

LayerConfigType = Union[List[ConfigType], ConfigType]


class LayerScale(nn.Module):
    """LayerScale layer.

    Args:
        dim (int): Dimension of input features.
        inplace (bool): Whether performs operation in-place.
            Default: `False`.
        data_format (str): The input data format, could be 'channels_last'
            or 'channels_first', representing (B, C, H, W) and
            (B, N, C) format data respectively. Default: 'channels_last'.
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self,
                 dim: int,
                 inplace: bool = False,
                 data_format: str = 'channels_last',
                 scale: float = 1e-5):
        super().__init__()
        assert data_format in ('channels_last', 'channels_first'), \
            "'data_format' could only be channels_last or channels_first."
        self.inplace = inplace
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(dim) * scale)

    def forward(self, x) -> torch.Tensor:
        if self.data_format == 'channels_first':
            shape = tuple((1, -1, *(1 for _ in range(x.dim() - 2))))
        else:
            shape = tuple((*(1 for _ in range(x.dim() - 1)), -1))
        if self.inplace:
            return x.mul_(self.weight.view(*shape))
        else:
            return x * self.weight.view(*shape)

@MODELS.register_module()
class FFN(BaseModule):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        layer_scale_init_value (float): Initial value of scale factor in
            LayerScale. Default: 1.0
    """

    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 layer_scale_init_value=0.):
        super().__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    MMENGINE_MODELS.build(act_cfg), nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = MMENGINE_MODELS.build(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(embed_dims, scale=layer_scale_init_value)
        else:
            self.gamma2 = nn.Identity()

    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        out = self.gamma2(out)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)

@TRANSFORMERS.register_module('DETRLayer')
class BaseTransformerLayer(BaseModule):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(self,
                 attn_cfgs: ConfigType,
                 ffn_cfgs: OptConfigType=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order: List[str]=None,
                 norm_cfg: OptConfigType=dict(type='LN'),
                 init_cfg: OptConfigType=None,
                 batch_first: bool=True,
                 **kwargs):

        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    f'and other FFN related arguments '
                    f'to a dict named `ffn_cfgs`. ', DeprecationWarning)
                ffn_cfgs[new_name] = kwargs[ori_name]

        super().__init__(init_cfg)

        self.batch_first = batch_first

        assert set(operation_order) & {
            'self_attn', 'norm', 'ffn', 'cross_attn'} == \
            set(operation_order), f'The operation_order of' \
            f' {self.__class__.__name__} should ' \
            f'contains all four operation type ' \
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"

        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = TRANSFORMERS.build(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(
                MODELS.build(ffn_cfgs[ffn_index]))

        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `key`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


@TRANSFORMERS.register_module('DETRLayerSequence')
class TransformerLayerSequence(BaseModule):
    """Base class for TransformerEncoder and TransformerDecoder in vision
    transformer.

    As base-class of Encoder and Decoder in vision transformer.
    Support customization such as specifying different kind
    of `transformer_layer` in `transformer_coder`.

    Args:
        transformerlayer (list[obj:`mmcv.ConfigDict`] |
            obj:`mmcv.ConfigDict`): Config of transformerlayer
            in TransformerCoder. If it is obj:`mmcv.ConfigDict`,
             it would be repeated `num_layer` times to a
             list[`mmcv.ConfigDict`]. Default: None.
        num_layers (int): The number of `TransformerLayer`. Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self, transformerlayers: LayerConfigType, 
                 num_layers: int=6, 
                 init_cfg: OptConfigType=None):
        super().__init__(init_cfg)
        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        else:
            assert isinstance(transformerlayers, list) and \
                   len(transformerlayers) == num_layers
        self.num_layers = num_layers
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(TRANSFORMERS.build(transformerlayers[i]))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm

    def forward(self,
                query,
                key,
                value,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerCoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_queries, bs, embed_dims)`.
            key (Tensor): The key tensor with shape
                `(num_keys, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_keys, bs, embed_dims)`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor], optional): Each element is 2D Tensor
                which is used in calculation of corresponding attention in
                operation_order. Default: None.
            query_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in self-attention
                Default: None.
            key_padding_mask (Tensor): ByteTensor for `key`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor:  results with shape [num_queries, bs, embed_dims].
        """
        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_padding_mask=query_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
        return query


# classes for deformable transformer
@TRANSFORMERS.register_module()
class DeformableDETRLayer(BaseModule):
    pass

@TRANSFORMERS.register_module()
class DeformableDETRLayerSequence(BaseModule):
    pass