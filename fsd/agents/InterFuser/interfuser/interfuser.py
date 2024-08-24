# Wrapper for naive transformer model
from typing import List, Dict, Tuple, Union, Sequence, AnyStr, TypedDict
import torch
import torch.nn as nn

from mmdet3d.models import Base3DDetector
from mmdet.models import (SinePositionalEncoding, \
                            LearnedPositionalEncoding)
from fsd.utils import ConfigType, OptConfigType, DataSampleType, OptDataSampleType
from fsd.registry import NECKS as FSD_NECKS
from fsd.registry import AGENTS as FSD_AGENTS
from fsd.registry import BACKBONES as FSD_BACKBONES
from fsd.registry import TRANSFORMERS as FSD_TRANSFORMERS
from fsd.registry import HEADS as FSD_HEADS

# define tying
INPUT_DATA_TYPE = Dict[AnyStr, torch.Tensor]

@FSD_AGENTS.register_module()
class InterFuser(Base3DDetector):
    def __init__(self,
                 num_queries: int,
                 embed_dims: int,
                 img_backbone: OptConfigType = None,
                 pts_backbone: OptConfigType = None,
                 img_neck: OptConfigType = None, # simple projection to a given dimension
                 pts_neck: OptConfigType = None, # simple projection to a given dimension
                 encoder: ConfigType = None,
                 decoder: ConfigType = None,
                 heads: ConfigType = None,
                 positional_encoding: ConfigType = None,
                 multi_view_encoding: ConfigType = None,
                 train_cfg: ConfigType = None,
                 test_cfg: ConfigType = None,
                 init_cfg: OptConfigType = None,
                 data_preprocessor: ConfigType = None,
                 **kwargs):
        
        """InterFuser model for multi-modality fusion.
        
        Args:
            num_queries (int): The number of queries.
            embed_dims (int): The dimension of embeddings.
            img_backbone (OptConfigType): The config of image backbone.
            pts_backbone (OptConfigType): The config of point cloud backbone.
            img_neck (OptCOnfigType): The config of image neck.
            pts_neck (OptConfigType): The config of point cloud neck.
            encoder (ConfigType): The config of encoder.
            decoder (ConfigType): The config of decoder.
            planner_head (ConfigType): The config of planner head.
            positional_encoding (ConfigType): The config of positional encoding.
            multi_view_encoding (ConfigType): The config of multi-view encoding.
            train_cfg (ConfigType): The config of training.
            test_cfg (ConfigType): The config of testing.
            init_cfg (OptConfigType): The config of initialization.
            data_preprocessor (ConfigType): The config of data preprocessor.
        
        """
        
        
        super(InterFuser, self).__init__(init_cfg=init_cfg,
                                         data_preprocessor=data_preprocessor,
                                         **kwargs)
        self.num_queries = num_queries
        self.embed_dims = embed_dims
        
        ## img backbone
        if img_backbone:
            self.img_backbone = FSD_BACKBONES.build(img_backbone)
        if pts_backbone:
            self.pts_backbone = FSD_BACKBONES.build(pts_backbone)
        
        ## neck applied to features extracted from img backbone abd pts backbone
        if img_neck:
            self.img_neck = FSD_NECKS.build(img_neck)
        if pts_neck:
            self.pts_neck = FSD_NECKS.build(pts_neck)
        
        ## embeddings
        # fixed positional encoding for encoder
        if positional_encoding:
            self.positional_encoding = SinePositionalEncoding(**positional_encoding)
            assert self.positional_encoding.num_feats * 2 == self.embed_dims, \
                'embed_dims should be exactly 2 times of num_feats. ' \
                f'Found {self.embed_dims} and {self.positional_encoding.num_feats}.'
                
        # learnable positional encoding for multi-modulity and multi-view sensors
        if multi_view_encoding:
            # sensor encoding
            self.multi_view_encoding = nn.Embedding(**multi_view_encoding)
            # another sensor encoding
            self.multi_view_mean_encoding = nn.Embedding(**multi_view_encoding)
            
        # learnable query embedding
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
                    
        # query position embedding
        # NOTE: the original paper applies learnable positional encoding for only parts of the query.
        # the query position seems only being applied to waypoint queries and traffic info queries
        self.num_queries_waypoints = 10
        self.num_queries_traffic_info = 1
        self.query_positional_encoding = nn.Embedding(
            self.num_queries_traffic_info + self.num_queries_waypoints, \
            self.embed_dims)
        
        ## detr transformer
        if encoder:
            self.encoder = FSD_TRANSFORMERS.build(encoder)
        if decoder:
            self.decoder = FSD_TRANSFORMERS.build(decoder)
        
        # train/test config
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # TODO: need better definitions: waypoint prediction, traffic info head, object density head etc
        # planner head
        #self.planner_head = MODELS.build(planner_head)
        if heads:
            self.heads = FSD_HEADS.build(heads)
            
        # init weights
        self.init_weights()
        
    @property
    def with_pts_backbone(self):
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_img_backbone(self):
        return hasattr(self, 'img_backbone') and self.img_backbone is not None
    
    @property
    def with_encoder(self):
        return hasattr(self, 'encoder') and self.encoder is not None
    
    @property
    def with_decoder(self):
        return hasattr(self, 'decoder') and self.decoder is not None
    
    @property
    def with_img_neck(self):
        return hasattr(self, 'img_neck') and self.img_neck is not None
    
    @property
    def with_pts_neck(self):
        return hasattr(self, 'pts_neck') and self.pts_neck is not None
    
    @property
    def with_multi_view_encoding(self):
        return hasattr(self, 'multi_view_encoding') and self.multi_view_encoding is not None
    
    def init_weights(self):
        super().init_weights()
        # init weights for embeddings using uniform
        for m in self.query_embedding, self.query_positional_encoding:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.uniform_(p)
        
    # extrack image features for each camera view
    def extract_img_feat(self, 
                         img: torch.Tensor, 
                         input_metas: List[dict]) -> dict:
        """Extract features of images.
        
        Args:
            img (torch.Tensor): Image of one sample, multi-view or single-view.
                Shape: [B, N, C, H, W] or [B, C, H, W].
            input_metas (List[dict]): The meta information of multiple samples
            
        """
        if img is None:
            return None
        
        if self.with_img_backbone and img is not None:
            dim = img.dim()            
            # update real input shape of each single img
            #for img_meta in input_metas:
            #    img_meta.update(input_shape=input_shape)
            if dim == 5:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            
            # choose the last from the tuple after backbone as feats
            img_feats = self.img_backbone(img)[-1]

            # reshape back to the original shape
            if dim == 5:
                _, Cf, Hf, Wf = img_feats.size()
                img_feats = img_feats.view(B, N, Cf, Hf, Wf)
            
        return img_feats

    # extrack point cloud features -> BEV features in this model
    def extract_pts_feat(self, 
                         pts_bev: torch.Tensor, 
                         input_metas: List[dict]) -> dict:
        """Extract features of points.
        
        Args:
            pts_bev (torch.Tensor): Point cloud BEV of one sample.
                Shape: [B, C, H, W].
            input_metas (List[dict]): The meta information of multiple samples.
        """
        if pts_bev is None:
            return None
        
        if self.with_pts_backbone and pts_bev is not None:
            dim = pts_bev.dim()
            # 
            if dim == 5:
                B, N, C, H, W = pts_bev.size()
                pts_bev = pts_bev.view(B * N, C, H, W)
            # extrack features from backbone
            pts_feats = self.pts_backbone(pts_bev)[-1]
            # point features
            if dim == 5:
                _, Cf, Hf, Wf = pts_feats.size()
                pts_feats = pts_feats.view(B, N, Cf, Hf, Wf)
            
        return pts_feats
    
    def extract_feat(self, 
                     batch_inputs_dict: INPUT_DATA_TYPE) -> INPUT_DATA_TYPE:
        """Extract features of images and points.
        
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'img' and 'pts' keys.
                
                - img (torch.Tensor): Image of each sample.
                - pts (torch.Tensor): Point cloud BEV of each sample.
                
        Returns:
            dict: The extracted features.
        """
        imgs = batch_inputs_dict['img'] if 'img' in batch_inputs_dict else None
        pts = batch_inputs_dict['pts'] if 'pts' in batch_inputs_dict else None
        # multiview-image
        img_feats = []
        for img in imgs:
            img_feats.append(self.extract_img_feat(img, None))
        
        pts_feats = self.extract_pts_feat(pts, None)
        
        return dict(img=img_feats, pts=pts_feats)
    
    def apply_neck(self, feats):
        """Apply neck for features.
        """
        # apply on image features
        img_feats = []
        for img in feats['img']:
            img_feats.append(self._apply_img_neck(img))
        feats['img'] = img_feats
        
        # apply on point cloud features -> BEV features      
        feats['pts'] = self._apply_pts_neck(feats['pts'])
        
        return feats
    
    def _apply_img_neck(self, img_feats):
        """Apply neck for image features.
        """
        if self.with_img_neck:
            dim_img = img_feats.dim()
            if dim_img == 5:
                B, N, C, H, W = img_feats.size()
                img_feats = img_feats.view(B * N, C, H, W)
            img_feats = self.img_neck(img_feats)
            if dim_img == 5:
                _, Cf, Hf, Wf = img_feats.size()
                img_feats = img_feats.view(B, N, Cf, Hf, Wf)
        
        return img_feats
    
    def _apply_pts_neck(self, pts_feats):
        """Apply neck for point cloud features.
        """
        if self.with_pts_neck:
            dim_pts = pts_feats.dim()
            if dim_pts == 5:
                B, N, C, H, W = pts_feats.size()
                pts_feats = pts_feats.view(B * N, C, H, W)
            pts_feats = self.pts_neck(pts_feats)
            if dim_pts == 5:
                _, Cf, Hf, Wf = pts_feats.size()
                pts_feats = pts_feats.view(B, N, Cf, Hf, Wf)
        
        return pts_feats

    def _forward_transformer(self, batch_inputs_dict, batch_targets_dict):
        """Forward function in tensor mode
        
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'img' and 'pts' keys.
                
                - img (torch.Tensor): Image of each sample.
                - pts (torch.Tensor): Point cloud BEV of each sample.
                
            batch_targets_dict (dict): The data samples dict.
        
        Returns:
            torch.Tensor: The output tensor that represents the model output without any post-processing.
        """

        # feature extraction
        feats = self.extract_feat(batch_inputs_dict)
        feats = self.apply_neck(feats)
        img_feats = feats['img']
        pts_feats = feats['pts']
        del feats
        
        
        assert isinstance(img_feats, list), 'multi-view img feats should be in a list.'
        assert pts_feats.dim() == 4, 'pts_feats should be 4-dim, (B, C, H, W).'
                
        N_img = len(img_feats)
        N_pts = 1
        B, _, _, _ = pts_feats.size()
        device = pts_feats.device
        
        #B, N_img, e, H, W = img_feats.size()
        #_, N_pts, _, _, _ = pts_feats.size()
        
        # encoder is a standard transformer encoder
        # decoder is a standard DETR decoder
        # for encoder inputs, get query/key embeddings + positional encodings + sensor encodings
        sensor_pos_encodings = self.multi_view_encoding(torch.arange(N_img + N_pts, device=device))
        sensor_mean_pos_encodings = self.multi_view_mean_encoding(torch.arange(N_img + N_pts, device=device))
        
        query_encoder = []
        # img: (B, C, H, w)
        for idx, feat in enumerate(img_feats + [pts_feats]):
            B, C, H, W = feat.size()
            # (B, C, 1)
            feat_mean = feat.mean(dim=[2, 3]).unsqueeze(-1)
            # (B, C, H, W)
            feat_embed = feat + self.positional_encoding(mask=None, input=feat) + \
                        sensor_pos_encodings[idx][None, :, None, None].repeat(B, 1, H, W)
            # (B, C, 1)
            feat_mean_embed = feat_mean + \
                        sensor_mean_pos_encodings[idx][None, :, None].repeat(B, 1, 1)
            query_encoder.extend([feat_embed.view(B, C, -1), feat_mean_embed.view(B, C, -1)])
        
        # (B, C, L) -> (B, L, C)
        query_encoder = torch.cat(query_encoder, dim=-1).permute(0, 2, 1)
        

        # query embedding
        # (N, dim) [num_objects_map, num_traffic_info, num_waypoints]
        query_decoder = self.query_embedding(torch.arange(self.num_queries, device=device))
        # (N_pos, dim,) -> (N, dim) -> (B, N, dim)
        query_pos_decoder = self.query_positional_encoding(torch.arange(self.query_positional_encoding.num_embeddings, device=device))
        query_pos_decoder = torch.cat([torch.zeros(self.num_queries - self.num_queries_waypoints - self.num_queries_traffic_info, self.embed_dims, device=device),
                                     query_pos_decoder[:self.num_queries_traffic_info, :], 
                                     query_pos_decoder[-self.num_queries_waypoints:, :]], 
                                    dim=0)
        query_pos_decoder = query_pos_decoder.unsqueeze(0).repeat(B, 1, 1)
        # (B, N, dim) 
        query_decoder = query_decoder.unsqueeze(0).repeat(B, 1, 1)
          
        ## encoder
        # [bs, NHW, dim]
        memory = self.encoder(
            query = query_encoder,
            key = query_encoder,
            value = query_encoder,
            query_pos = None, # self attention: query_pos = key_pos
            key_pos = None,
            attn_masks = None,
            query_key_padding_mask = None,
            key_padding_mask = None,
        )
        
        ## decoder (bs, num_queries, dim)
        output_dec = self.decoder(
            query = query_decoder,
            key = memory,
            value = memory,
            query_pos = query_pos_decoder,
            key_pos = None,
            attn_masks = None,
            query_key_padding_mask = None,
            key_padding_mask = None,
        )
                
        return output_dec
    
    def _forward_heads(self, output_decoder, goal_points):
        """Forward function for heads in tensor mode
        
        Args:
            output_decoder (torch.Tensor): The output tensor of decoder.
            batch_targets_dict (dict): The data samples dict.
        
        Returns:
            torch.Tensor: The output tensor that represents the model output without any post-processing.
        """

        output = self.heads(output_decoder, goal_points)
        
        return output
    
    def _forward(self, batch_inputs_dict, batch_targets_dict) -> Dict[AnyStr, torch.Tensor]:
        
        goal_points = batch_inputs_dict.get('goal_points', None)
        output_dec = self._forward_transformer(batch_inputs_dict, batch_targets_dict)
        output = self._forward_heads(output_dec, goal_points)
        
        return output
    
    def loss(self, batch_inputs_dict, data_samples, **kwargs):
        goal_points = batch_inputs_dict.get('goal_points', None)
        output_dec = self._forward_transformer(batch_inputs_dict, data_samples)
        losses = self.heads.loss(output_dec, goal_points, data_samples)
        
        return losses 
    
    def predict(self, batch_inputs_dict, batch_targets_dict, **kwargs):
        goal_points = batch_inputs_dict.get('goal_points', None)
        output_dec = self._forward_transformer(batch_inputs_dict, batch_targets_dict)
        preds = self.heads.predict(output_dec, goal_points)
        
        return preds
    
    def forward(self,
                inputs: Union[dict, List[dict]],
                data_samples: List[OptDataSampleType] = None,
                mode: str = 'tensor',
                **kwargs):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`Det3DDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs  (dict | list[dict]): When it is a list[dict], the
                outer list indicate the test time augmentation. Each
                dict contains batch inputs
                which include 'points' and 'img' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - img (torch.Tensor): Image tensor has shape (B, C, H, W) or 
                    (B, N, C, H, W).
            batch_targets_dict (dict): The
                annotation data of every samples. When it is a list[list], the
                outer list indicate the test time augmentation, and the
                inter list indicate the batch. Otherwise, the list simply
                indicate the batch. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a dict of predictions.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            
            return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
