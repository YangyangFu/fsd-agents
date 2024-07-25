# Wrapper for naive transformer model
from typing import List, Dict, Tuple, Union, Sequence
import torch
import torch.nn as nn

from mmdet3d.utils import ConfigType, OptConfigType
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.models import Base3DDetector

class InterFuser(Base3DDetector):
    def __init__(self,
                 img_backbone: OptConfigType = None,
                 pts_backbone: OptConfigType = None,
                 encoder: ConfigType = None,
                 decoder: ConfigType = None,
                 planner_head: ConfigType = None,
                 train_cfg: ConfigType = None,
                 test_cfg: ConfigType = None,
                 init_cfg: OptConfigType = None,
                 data_preprocessor: ConfigType = None,
                 **kwargs):
        
        super(InterFuser, self).__init__(init_cfg=init_cfg,
                                         data_preprocessor=data_preprocessor,
                                         **kwargs)
        # img backbone
        if img_backbone:
            self.img_backbone = MODELS.build(img_backbone)
        if pts_backbone:
            self.pts_backbone = MODELS.build(pts_backbone)
        
        # detr transformer
        if encoder:
            self.encoder = MODELS.build(encoder)
        if decoder:
            self.decoder = MODELS.build(decoder)
        
        # train/test config
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # TODO: need more definitions: waypoint prediction, traffic info head, object density head etc
        # planner head
        self.planner_head = MODELS.build(planner_head)
        
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
            for img_meta in input_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
                
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
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
    
    def extract_feat(self, 
                     batch_inputs_dict: dict) -> Sequence[torch.Tensor]:
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
        
        return (img_feats, pts_feats)
    
    
    def _forward(self, batch_inputs_dict, data_samples):
        """Forward function in tensor mode
        """
        img_feats, pts_feats = self.extract_feat(batch_inputs_dict)
        
        # concatenate and apply positional encoding and multi-view encoding
        
        # encoder
        
        # decoder
        
        # planner head
        
        results = None
        return results