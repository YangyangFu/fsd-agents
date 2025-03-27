# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Yangyang Fu
# ---------------------------------------------
from typing import Dict, List, Tuple
import torch
import copy
from mmengine.structures import InstanceData
from mmdet3d.structures.ops import bbox3d2result

from .utils.grid_mask import GridMask
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from fsd.registry import MODELS


@MODELS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 video_test_mode=False
                 ):

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
                         len_queue: int=None) -> List[torch.Tensor]:
        """Extract features of images.
        
        Args:
            img (torch.Tensor): Image tensor with shape (B, N, C, H, W).
            img_metas (dict): Meta information of each sample.
            len_queue (int): The length of the queue. Defaults to None.
        
        Returns:
            list[torch.Tensor]: Extracted features of images
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
                    len_queue: int=None) -> List[torch.Tensor]:
        """Extract features of images.
        
        Args:
            img (torch.Tensor): Image tensor with shape (B, N, C, H, W).
            len_queue (int): The length of the queue. Defaults to None.
        
        Returns:
            list[torch.Tensor]: Extracted features of images
        """

        img_feats = self.extract_img_feat(img, len_queue=len_queue)
        
        return img_feats


    def forward_pts_train(self,
                          pts_feats,
                          data_samples,
                          img_metas,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)
        losses = self.pts_bbox_head.loss(preds_dicts = outs, 
                                         batch_data_samples = data_samples)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, inputs, 
                data_samples, 
                mode:str = 'loss', 
                **kwargs,):
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
            data_samples (dict): The
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
        if mode == "loss":
            return self.forward_train(inputs, data_samples, **kwargs)
        else:
            return self.forward_test(inputs, data_samples, **kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                #img_metas = [each for each in img_metas[i]]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas[i], prev_bev, only_bev=True)
            self.train()
            return prev_bev

    def forward_train(self,
                      inputs, 
                      data_samples,
                      **kwargs
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        # get inputs
        device = inputs['img'][0].device
        img = inputs['img']
        img_metas = kwargs['img_metas']
        
        # separate inputs
        len_queue = img[0].size(0)
        prev_img = torch.stack([im[:-1, ...] for im in img], dim=0).to(device) # (B, L-1, N, C, H, W)
        img = torch.stack([im[-1, ...] for im in img], dim=0).to(device) # (B, N, C, H, W)

        # previous images for bev
        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        # current image
        curr_img_metas = img_metas[len_queue-1]
        img_feats = self.extract_feat(img=img)
        
        # loss
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, data_samples,
                                            img_metas=curr_img_metas, 
                                            prev_bev=prev_bev)

        losses.update(losses_pts)
        return losses

    def forward_test(self, inputs, data_samples, **kwargs):

        img = inputs['img']
        device = img[0].device
        img = torch.stack(img, dim=0).to(device)
        img_metas = kwargs['img_metas']

        #TODO: this seems to only work with batch=1
        if img_metas['scene_token'][0] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas['scene_token'][0]

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas['can_bus'][0][:3])
        tmp_angle = copy.deepcopy(img_metas['can_bus'][0][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas['can_bus'][0][:3] -= self.prev_frame_info['prev_pos']
            img_metas['can_bus'][0][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas['can_bus'][0][-1] = 0
            img_metas['can_bus'][0][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas, img, prev_bev=self.prev_frame_info['prev_bev'])
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        
        # format for nuscenes evaluation
        # bbox_results: List[Dict]
        pred_instances_3d = []
        # batched
        for bbox_result in bbox_results:
            instance = InstanceData(
                scores_3d = bbox_result['scores_3d'],
                labels_3d = bbox_result['labels_3d'],
                bboxes_3d = bbox_result['bboxes_3d']
            ) 
            pred_instances_3d.append(instance)
               
        data_samples = self.add_pred_to_datasample(
            data_samples = data_samples,
            data_instances_3d=pred_instances_3d)
        
        return data_samples

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img)
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)

        return new_prev_bev, bbox_pts
