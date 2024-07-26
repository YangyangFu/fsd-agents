# Wrapper for naive transformer model
from typing import List, Dict, Tuple, Union, Sequence
import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmdet.registry import MODELS as MMDET_MODELS
from mmdet3d.utils import ConfigType, OptConfigType
from mmdet3d.registry import MODELS as MMDET3D_MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.models import Base3DDetector

from mmdet.models import SinePositionalEncoding, LearnedPositionalEncoding

@MMDET_MODELS.register_module()
class InterfuserNect(BaseModule):
    """Interfuser feature neck.

        A simple 1x1 convolutional layer to project the input features to a given dimension.
    """
    def __init__(self, in_channels: int, out_channels: int, init_cfg: OptConfigType = None):
        super(InterfuserNect, self).__init__(init_cfg=init_cfg)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class InterFuser(Base3DDetector):
    def __init__(self,
                 num_queries: int,
                 embed_dims: int,
                 img_backbone: OptConfigType = None,
                 pts_backbone: OptConfigType = None,
                 img_neck: OptConfigType = None, # simple projection to a given dimension
                 pts_nect: OptConfigType = None, # simple projection to a given dimension
                 encoder: ConfigType = None,
                 decoder: ConfigType = None,
                 planner_head: ConfigType = None,
                 positional_encoding: ConfigType = None,
                 multi_view_encoding: ConfigType = None,
                 query_embedding: ConfigType = None,
                 train_cfg: ConfigType = None,
                 test_cfg: ConfigType = None,
                 init_cfg: OptConfigType = None,
                 data_preprocessor: ConfigType = None,
                 **kwargs):
        
        super(InterFuser, self).__init__(init_cfg=init_cfg,
                                         data_preprocessor=data_preprocessor,
                                         **kwargs)
        self.num_queries = num_queries
        self.embed_dims = embed_dims

        ## img backbone
        if img_backbone:
            self.img_backbone = MMDET_MODELS.build(img_backbone)
        if pts_backbone:
            self.pts_backbone = MMDET_MODELS.build(pts_backbone)
        
        ## neck applied to features extracted from img backbone abd pts backbone
        if img_neck:
            self.img_neck = MMDET_MODELS.build(img_neck)
        if pts_nect:
            self.pts_neck = MMDET_MODELS.build(pts_nect)
        
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
        # if the query position seems only being applied to waypoint queries and traffic info queries
        self.num_queries_waypoints = 10
        self.num_queries_traffic_info = 1
        self.query_positional_encoding = nn.Embedding(
            self.num_queries_waypoints + self.num_queries_traffic_info, \
            self.embed_dims)
        
        ## detr transformer
        if encoder:
            self.encoder = MMDET3D_MODELS.build(encoder)
        if decoder:
            self.decoder = MMDET3D_MODELS.build(decoder)
        
        # train/test config
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # TODO: need more definitions: waypoint prediction, traffic info head, object density head etc
        # planner head
        #self.planner_head = MODELS.build(planner_head)
        
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
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            
            # update real input shape of each single img
            #for img_meta in input_metas:
            #    img_meta.update(input_shape=input_shape)

            if img.dim() == 5:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
                
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        
        if img.dim() == 5:
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
        if self.with_pts_backbone and pts_bev is not None:
            pts_feats = self.pts_backbone(pts_bev)
        else:
            return None
        return pts_feats
    
    def extract_backbone_feats(self, 
                     batch_inputs_dict: dict) -> Dict[str: torch.Tensor]:
        """Extract features of images and points.
        
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'img' and 'pts' keys.
                
                - img (torch.Tensor): Image of each sample.
                - pts (torch.Tensor): Point cloud BEV of each sample.
                
        Returns:
            dict: The extracted features.
        """
        imgs = batch_inputs_dict.get('imgs', None)
        pts = batch_inputs_dict.get('pts', None)
        img_feats = self.extract_img_feat(imgs, None)
        pts_feats = self.extract_pts_feat(pts, None)
        
        return dict(imgs=img_feats, pts=pts_feats)
    
    def _apply_embed(self):
        """Apply embedding for features.
        """
        
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
            feats['pts'] = self.pts_neck(feats['pts'])
            
        return feats
    
    def _forward(self, batch_inputs_dict, data_samples):
        """Forward function in tensor mode
        """
        feats = self.extract_backbone_feats(batch_inputs_dict)
        feats = self._apply_neck(feats)
        img_feats = feats['imgs']
        pts_feats = feats['pts']
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
        # (N, dim)
        sensor_pos_encodings = self.multi_view_encoding(torch.arange(N_img + N_pts))
        # (B, dim, H, W) -> (B, N, dim, H, W) -> (B, NHW, dim)
        mask = torch.zeros(B, H, W)
        key_pos_encodings = self.positional_encoding(mask=mask)
        key_pos_encodings = key_pos_encodings.unsqueeze(0).repeat(1, N_img + N_pts, 1, 1, 1) + \
                sensor_pos_encodings[None, :, :, None, None].repeat(B, 1, 1, H, W)
        key_pos_encodings = key_pos_encodings.permute(0, 1, 3, 4, 2).reshape(B, -1, e)
        
        # query embedding
        # (num_queries, dim) [num_waypoints, num_objects_map, num_traffic_info]
        query_embed = self.query_embedding(torch.arange(self.num_queries))
        # (num_query_pos, dim,) -> (B, num_queries, dim)
        query_pos_embed = self.query_positional_encoding(torch.arange(self.query_positional_encoding.num_embeddings))
        query_pos_embed = torch.cat([query_pos_embed[:self.num_queries_waypoints, :], 
                                     torch.zeros(self.num_queries - self.num_queries_waypoints - self.num_queries_traffic_info, self.embed_dims),
                                     query_pos_embed[-self.num_queries_traffic_info:, :]], 
                                    dim=0)
        query_pos_embed = query_pos_embed.unsqueeze(0).repeat(B, 1, 1)
        # (B, num_queries, dim) 
        query_embed = query_embed.unsqueeze(0).repeat(B, 1, 1)
          
        ## encoder
        # [bs, NHW, dim]
        memory = self.encoder(
            query = key_embed,
            query_pos = key_pos_encodings,
            key_padding_mask = None
        )
        
        ## decoder
        output_dec = self.decoder(
            query = query_embed,
            key = memory,
            value = memory,
            query_pos = query_pos_embed,
            key_pos = key_pos_encodings,
            key_padding_mask = None
        )
        
        # planner head
        
        results = output_dec
        return results