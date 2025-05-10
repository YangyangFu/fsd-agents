# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Yangyang Fu
# ---------------------------------------------
from typing import Dict, List, Tuple, Union, Optional
import torch
import copy
from mmengine.structures import InstanceData
from mmdet3d.structures.ops import bbox3d2result

from .utils.grid_mask import GridMask
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from fsd.registry import MODELS
from fsd.structures import Instances

@MODELS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer."""

    def __init__(self,
                 use_grid_mask: bool = False,
                 pts_voxel_encoder: Optional[Dict] = None,
                 pts_middle_encoder: Optional[Dict] = None,
                 pts_fusion_layer: Optional[Dict] = None,
                 img_backbone: Optional[Dict] = None,
                 pts_backbone: Optional[Dict] = None,
                 img_neck: Optional[Dict] = None,
                 pts_neck: Optional[Dict] = None,
                 pts_bbox_head: Optional[Dict] = None,
                 img_roi_head: Optional[Dict] = None,
                 img_rpn_head: Optional[Dict] = None,
                 train_cfg: Optional[Dict] = None,
                 test_cfg: Optional[Dict] = None,
                 video_test_mode: bool = False
                 ) -> None:
        """Initialize BEVFormer.

        Args:
            use_grid_mask (bool): Whether to use grid mask augmentation.
            pts_voxel_encoder (dict, optional): Config for point cloud voxel encoder.
            pts_middle_encoder (dict, optional): Config for point cloud middle encoder.
            pts_fusion_layer (dict, optional): Config for point cloud fusion layer.
            img_backbone (dict, optional): Config for image backbone.
            pts_backbone (dict, optional): Config for point cloud backbone.
            img_neck (dict, optional): Config for image neck.
            pts_neck (dict, optional): Config for point cloud neck.
            pts_bbox_head (dict, optional): Config for point cloud bounding box head.
            img_roi_head (dict, optional): Config for image ROI head.
            img_rpn_head (dict, optional): Config for image RPN head.
            train_cfg (dict, optional): Training configuration.
            test_cfg (dict, optional): Testing configuration.
            video_test_mode (bool): Whether to use temporal information during inference.
        """
        super(BEVFormer,
              self).__init__(pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }


    def extract_img_feat(self, 
                         img: torch.Tensor, 
                         len_queue: Optional[int] = None) -> List[torch.Tensor]:
        """Extract features of images.

        Args:
            img (torch.Tensor): Image tensor with shape (B, N, C, H, W).
            len_queue (int, optional): The length of the queue. Defaults to None.

        Returns:
            List[torch.Tensor]: Extracted features of images, reshaped based on `len_queue`.
        """
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self,
                     img: torch.Tensor, 
                     len_queue: Optional[int] = None) -> List[torch.Tensor]:
        """Extract features of images.

        Args:
            img (torch.Tensor): Image tensor with shape (B, N, C, H, W).
            len_queue (int, optional): The length of the queue. Defaults to None.

        Returns:
            List[torch.Tensor]: Extracted features of images.
        """

        img_feats = self.extract_img_feat(img, len_queue=len_queue)
        
        return img_feats


    def forward_pts_train(self,
                          pts_feats: List[torch.Tensor],
                          data_samples: List[Dict],
                          img_metas: List[Dict],
                          prev_bev: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward function for training the point cloud branch.

        Args:
            pts_feats (List[torch.Tensor]): Features of the point cloud branch.
            data_samples (List[Dict]): Ground truth data samples.
            img_metas (List[Dict]): Meta information of samples.
            prev_bev (torch.Tensor, optional): BEV features of the previous frame. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)
        losses = self.pts_bbox_head.loss(preds_dicts = outs, 
                                         batch_data_samples = data_samples)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, 
                inputs: Dict[str, Union[torch.Tensor, List[torch.Tensor]]], 
                data_samples: List[Dict], 
                mode: str = 'loss', 
                **kwargs) -> Union[torch.Tensor, Dict, List[Dict]]:
        """Unified entry for forward process in both training and testing.

        Args:
            inputs (Dict): Input data containing 'points' and 'img'.
                - points (List[torch.Tensor]): Point cloud of each sample.
                - img (torch.Tensor): Image tensor with shape (B, C, H, W) or (B, N, C, H, W).
            data_samples (List[Dict]): Annotation data of each sample.
            mode (str): Mode of operation ('loss', 'predict', or 'tensor'). Defaults to 'loss'.

        Returns:
            Union[torch.Tensor, Dict, List[Dict]]: Output depends on the mode.
        """
        if mode == "loss":
            return self.forward_train(inputs, data_samples, **kwargs)
        else:
            return self.forward_test(inputs, data_samples, **kwargs)
    
    def obtain_history_bev(self, 
                           imgs_queue: torch.Tensor, 
                           img_metas: List[Dict]) -> torch.Tensor:
        """Obtain history BEV features iteratively.

        Args:
            imgs_queue (torch.Tensor): Image queue with shape (B, L, N, C, H, W).
            img_metas (List[Dict]): Meta information of each sample.

        Returns:
            torch.Tensor: History BEV features.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas_i = [each[i] for each in img_metas]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas_i, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    def forward_train(self,
                      inputs: Dict[str, Union[torch.Tensor, List[torch.Tensor]]], 
                      data_samples: List[Dict],
                      **kwargs) -> Dict[str, torch.Tensor]:
        """Forward function for training.

        Args:
            inputs (Dict): Input data containing 'points' and 'img'.
                - points (List[torch.Tensor]): Point cloud of each sample.
                - img (torch.Tensor): Image tensor with shape (B, C, H, W) or (B, N, C, H, W).
            data_samples (List[Dict]): Ground truth data samples.

        Returns:
            Dict[str, torch.Tensor]: Losses of different branches.
        """
        # get inputs
        device = inputs['img'][0].device
        img = inputs['img']
        img_metas = [sample.bev_metas for sample in data_samples]
        len_queue = img[0].size(0)
        
        # separate inputs
        prev_img = torch.stack([im[:-1, ...] for im in img], dim=0).to(device) # (B, L-1, N, C, H, W)
        curr_img = torch.stack([im[-1, ...] for im in img], dim=0).to(device) # (B, N, C, H, W)

        # previous images for bev
        prev_img_metas = [each[:-1] for each in img_metas]
        prev_bev = None
        if len_queue > 1:
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        # current image
        curr_img_metas = [each[-1] for each in img_metas]
        curr_img_feats = self.extract_feat(img=curr_img)
        
        # loss
        losses = dict()
        losses_pts = self.forward_pts_train(curr_img_feats, 
                                            data_samples,
                                            img_metas=curr_img_metas, 
                                            prev_bev=prev_bev)

        losses.update(losses_pts)
        return losses

    def forward_test(self, 
                     inputs: Dict[str, Union[torch.Tensor, List[torch.Tensor]]], 
                     data_samples: List[Dict], 
                     **kwargs) -> List[Dict]:
        """Forward function for testing.

        Args:
            inputs (Dict): Input data containing 'points' and 'img'.
                - points (List[torch.Tensor]): Point cloud of each sample.
                - img (torch.Tensor): Image tensor with shape (B, C, H, W) or (B, N, C, H, W).
            data_samples (List[Dict]): Annotation data of each sample.

        Returns:
            List[Dict]: Predictions for each sample.
        """
        img = inputs['img']
        device = img[0].device
        img = torch.stack(img, dim=0).to(device)
        img_metas = [sample.bev_metas for sample in data_samples]

        #TODO: this seems to only work with batch=1
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['bev_attr'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['bev_attr'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['bev_attr'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['bev_attr'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['bev_attr'][-1] = 0
            img_metas[0][0]['bev_attr'][:3] = 0

        # NOTE: only support batch size = 1
        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img, prev_bev=self.prev_frame_info['prev_bev'])
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        
        # format for nuscenes evaluation
        # bbox_results: List[Dict]
        pred_instances_3d = []
        # batched
        for bbox_result in bbox_results:
            instance = Instances(
                score = bbox_result['scores_3d'],
                label = bbox_result['labels_3d'],
                bbox = bbox_result['bboxes_3d']
            ) 
            pred_instances_3d.append(instance)
               
        data_samples = self.add_pred_to_datasample(
            data_samples = data_samples,
            data_instances_3d=pred_instances_3d)
        
        return data_samples

    def simple_test_pts(self, 
                        x: List[torch.Tensor], 
                        img_metas: List[Dict], 
                        prev_bev: Optional[torch.Tensor] = None, 
                        rescale: bool = False) -> Tuple[torch.Tensor, List[Dict]]:
        """Test function for point cloud branch.

        Args:
            x (List[torch.Tensor]): Extracted features.
            img_metas (List[Dict]): Meta information of samples.
            prev_bev (torch.Tensor, optional): BEV features of the previous frame. Defaults to None.
            rescale (bool): Whether to rescale the results. Defaults to False.

        Returns:
            Tuple[torch.Tensor, List[Dict]]: BEV embeddings and bounding box results.
        """
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, 
                    img_metas: List[Dict], 
                    img: Optional[torch.Tensor] = None, 
                    prev_bev: Optional[torch.Tensor] = None, 
                    rescale: bool = False) -> Tuple[torch.Tensor, List[Dict]]:
        """Test function without augmentation.

        Args:
            img_metas (List[Dict]): Meta information of samples.
            img (torch.Tensor, optional): Image tensor with shape (B, C, H, W) or (B, N, C, H, W). Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of the previous frame. Defaults to None.
            rescale (bool): Whether to rescale the results. Defaults to False.

        Returns:
            Tuple[torch.Tensor, List[Dict]]: BEV embeddings and bounding box results.
        """
        img_feats = self.extract_feat(img=img)
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)

        return new_prev_bev, bbox_pts
