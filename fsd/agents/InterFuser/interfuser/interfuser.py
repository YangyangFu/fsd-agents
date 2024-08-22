# Wrapper for naive transformer model
from typing import List, Dict, Tuple, Union, Sequence, AnyStr, TypedDict
import torch
import torch.nn as nn

from mmdet3d.models import Base3DDetector
from mmdet.models import (SinePositionalEncoding, \
                            LearnedPositionalEncoding)
from fsd.utils import ConfigType, OptConfigType
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
            self.multi_view_encoding = nn.Embedding(**multi_view_encoding)

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
        imgs = batch_inputs_dict['img'].data if 'img' in batch_inputs_dict else None
        pts = batch_inputs_dict['pts'].data if 'pts' in batch_inputs_dict else None
        img_feats = self.extract_img_feat(imgs, None)
        pts_feats = self.extract_pts_feat(pts, None)
        
        return dict(imgs=img_feats, pts=pts_feats)
    
    def _apply_neck(self, feats):
        """Apply neck for features.
        """
        # apply on image features
        if self.with_img_neck:
            dim_img = feats['imgs'].dim()
            if dim_img == 5:
                B, N, C, H, W = feats['imgs'].size()
                feats['imgs'] = feats['imgs'].view(B * N, C, H, W)
            feats['imgs'] = self.img_neck(feats['imgs'])
            if dim_img == 5:
                _, Cf, Hf, Wf = feats['imgs'].size()
                feats['imgs'] = feats['imgs'].view(B, N, Cf, Hf, Wf)
        
        # apply on point cloud features -> BEV features      
        if self.with_pts_neck:
            dim_pts = feats['pts'].dim()
            if dim_pts == 5:
                B, N, C, H, W = feats['pts'].size()
                feats['pts'] = feats['pts'].view(B * N, C, H, W)
                
            feats['pts'] = self.pts_neck(feats['pts'])
            if dim_pts == 5:
                _, Cf, Hf, Wf = feats['pts'].size()
                feats['pts'] = feats['pts'].view(B, N, Cf, Hf, Wf)
        
        return feats
    
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
        feats = self.extract_feat(batch_inputs_dict)
        feats = self._apply_neck(feats)
        img_feats = feats['imgs']
        pts_feats = feats['pts']
        goal_points = batch_inputs_dict.get('goal_points', None)
        del feats
        
        
        assert img_feats.dim() in [4, 5], 'img_feats should be 4-dim or 5-dim tensor.'
        assert pts_feats.dim() in [4, 5], 'img_feats should be 4-dim or 5-dim tensor.'
        if img_feats.dim() == 4:
            img_feats = img_feats.unsqueeze(1)
        if pts_feats.dim() == 4:
            pts_feats = pts_feats.unsqueeze(1)
        
        B, N_img, e, H, W = img_feats.size()
        _, N_pts, _, _, _ = pts_feats.size()
        
        ## concatenate and apply positional encoding and multi-view encoding
        # (B, N, dim, H, W) -> (B, NHW, dim)
        key_embed = torch.cat([img_feats, pts_feats], dim=1).permute(0, 1, 3, 4, 2).reshape(B, -1, e)
        

        # (B, H, W) -> (B, N, dim, H, W) -> (B, NHW, dim)
        mask = torch.zeros(B, H, W)
        # (B, dim, H, W)
        key_pos_encodings = self.positional_encoding(mask=mask)
        # (B, N, dim, H, W)
        key_pos_encodings = key_pos_encodings.unsqueeze(1).repeat(1, N_img + N_pts, 1, 1, 1)
            
        if self.with_multi_view_encoding:
            # (N, dim)
            sensor_pos_encodings = self.multi_view_encoding(torch.arange(N_img + N_pts))
            # (B, N, dim, H, W)
            sensor_pos_encodings = sensor_pos_encodings[None, :, :, None, None].repeat(B, 1, 1, H, W)
            key_pos_encodings += sensor_pos_encodings
        # (B, N, dim, H, W) -> (B, N, H, W, dim) -> (B, NHW, dim)    
        key_pos_encodings = key_pos_encodings.permute(0, 1, 3, 4, 2).reshape(B, -1, e)
        
        # query embedding
        # (N, dim) [num_objects_map, num_traffic_info, num_waypoints]
        query_embed = self.query_embedding(torch.arange(self.num_queries))
        # (N_pos, dim,) -> (N, dim) -> (B, N, dim)
        query_pos_embed = self.query_positional_encoding(torch.arange(self.query_positional_encoding.num_embeddings))
        query_pos_embed = torch.cat([torch.zeros(self.num_queries - self.num_queries_waypoints - self.num_queries_traffic_info, self.embed_dims),
                                     query_pos_embed[:self.num_queries_traffic_info, :], 
                                     query_pos_embed[-self.num_queries_waypoints:, :]], 
                                    dim=0)
        query_pos_embed = query_pos_embed.unsqueeze(0).repeat(B, 1, 1)
        # (B, N, dim) 
        query_embed = query_embed.unsqueeze(0).repeat(B, 1, 1)
          
        ## encoder
        # [bs, NHW, dim]
        memory = self.encoder(
            query = key_embed,
            key = key_embed,
            value = key_embed,
            query_pos = key_pos_encodings,
            key_pos = None,
            attn_masks = None,
            query_key_padding_mask = None,
            key_padding_mask = None,
        )
        
        ## decoder (bs, num_queries, dim)
        output_dec = self.decoder(
            query = query_embed,
            key = memory,
            value = memory,
            query_pos = query_pos_embed,
            key_pos = key_pos_encodings,
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
    
    def loss(self, batch_inputs_dict, batch_targets_dict, **kwargs):
        goal_points = batch_inputs_dict.get('goal_points', None)
        output_dec = self._forward_transformer(batch_inputs_dict, batch_targets_dict)
        losses = self.heads.loss(output_dec, goal_points, batch_targets_dict)
        
        return losses 
    
    def predict(self, batch_inputs_dict, batch_targets_dict, **kwargs):
        goal_points = batch_inputs_dict.get('goal_points', None)
        output_dec = self._forward_transformer(batch_inputs_dict, batch_targets_dict)
        preds = self.heads.predict(output_dec, goal_points)
        
        return preds
    
    def forward(self,
                inputs: Union[dict, List[dict]],
                batch_targets_dict = None,
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
                which include 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Image tensor has shape (B, C, H, W).
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
            return self.loss(inputs, batch_targets_dict, **kwargs)
        elif mode == 'predict':
            
            return self.predict(inputs, batch_targets_dict, **kwargs)
        elif mode == 'tensor':
            return self._forward(inputs, batch_targets_dict, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
