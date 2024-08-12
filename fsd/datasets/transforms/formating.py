from collections.abc import Sequence

import numpy as np
import torch
#from fsd.structures.data_container import BaseDataElement as DC

#from mmcv.core.bbox.structures.base_box3d import BaseInstance3DBoxes
#from mmcv.core.points import BasePoints
from mmengine.structures import BaseDataElement, InstanceData, PixelData
from mmdet3d.structures import BaseInstance3DBoxes, BasePoints, PointData
from mmengine.utils import is_str
from fsd.registry import TRANSFORMS

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@TRANSFORMS.register_module()
class ToTensor:
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert data in results to :obj:`torch.Tensor`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted
                to :obj:`torch.Tensor`.
        """
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class ImageToTensor:
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img.transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class Transpose:
    """Transpose some results by given keys.

    Args:
        keys (Sequence[str]): Keys of results to be transposed.
        order (Sequence[int]): Order of transpose.
    """

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        """Call function to transpose the channel order of data in results.

        Args:
            results (dict): Result dict contains the data to transpose.

        Returns:
            dict: The result dict contains the data transposed to \
                ``self.order``.
        """
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, order={self.order})'


@TRANSFORMS.register_module()
class ToBaseDataElement:
    """Convert results to :obj:`mmengine.BaseDataElement` by given fields.

    Args:
        fields (Sequence[dict]): Each field is a dict like
            ``dict(key='xxx', **kwargs)``. The ``key`` in result will
            be converted to :obj:`mmengine.BaseDataElement` with ``**kwargs``.
            Default: ``(dict(key='img', stack=True), dict(key='gt_bboxes'),
            dict(key='gt_labels'))``.
    """

    def __init__(self,
                 fields=(dict(key='img', stack=True), dict(key='gt_bboxes'),
                         dict(key='gt_labels'))):
        self.fields = fields

    def __call__(self, results):
        """Call function to convert data in results to
        :obj:`mmengine.BaseDataElement`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted to \
                :obj:`mmengine.BaseDataElement`.
        """

        for field in self.fields:
            field = field.copy()
            key = field.pop('key')
            results[key] = BaseDataElement(results[key], **field)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(fields={self.fields})'

@TRANSFORMS.register_module()
class WrapFieldsToLists:
    """Wrap fields of the data dictionary into lists for evaluation.

    This class can be used as a last step of a test or validation
    pipeline for single image evaluation or inference.

    Example:
        >>> test_pipeline = [
        >>>    dict(type='LoadImageFromFile'),
        >>>    dict(type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
        >>>    dict(type='Pad', size_divisor=32),
        >>>    dict(type='ImageToTensor', keys=['img']),
        >>>    dict(type='Collect', keys=['img']),
        >>>    dict(type='WrapFieldsToLists')
        >>> ]
    """

    def __call__(self, results):
        """Call function to wrap fields into lists.

        Args:
            results (dict): Result dict contains the data to wrap.

        Returns:
            dict: The result dict where value of ``self.keys`` are wrapped \
                into list.
        """

        # Wrap dict fields into lists
        for key, val in results.items():
            results[key] = [val]
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'
    

#TRANSFORMS._module_dict.pop('DefaultFormatBundle')
@TRANSFORMS.register_module()
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to BaseDataElement (stack=True)
    - proposals: (1)to tensor, (2)to BaseDataElement
    - gt_bboxes: (1)to tensor, (2)to BaseDataElement
    - gt_bboxes_ignore: (1)to tensor, (2)to BaseDataElement
    - gt_labels: (1)to tensor, (2)to BaseDataElement
    - gt_masks: (1)to tensor, (2)to BaseDataElement (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to BaseDataElement (stack=True)
    """

    def __init__(self, ):
        return

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['img'] = BaseDataElement(data=to_tensor(imgs))
            else:
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                results['img'] = BaseDataElement(data=to_tensor(img))
        for key in [
                'proposals', 'gt_bboxes_ignore', 'gt_labels_3d', 
                'pts_instance_mask', 'pts_semantic_mask', 'centers2d', 'depths'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = BaseDataElement(data=[to_tensor(res) for res in results[key]])
            else:
                results[key] = BaseDataElement(data=to_tensor(results[key]))
        if 'gt_bboxes_3d' in results:
            if isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = BaseDataElement(
                    data=results['gt_bboxes_3d'])
            else:
                results['gt_bboxes_3d'] = BaseDataElement(
                    data=to_tensor(results['gt_bboxes_3d']))

        if 'gt_bboxes_mask' in results:
            results['gt_bboxes_mask'] = BaseDataElement(data=results['gt_bboxes_mask'])

        return results

    def __repr__(self):
        return self.__class__.__name__

@TRANSFORMS.register_module()
class DefaultFormatBundle3D(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to BaseDataElement (stack=True)
    - proposals: (1)to tensor, (2)to BaseDataElement
    - gt_bboxes: (1)to tensor, (2)to BaseDataElement
    - gt_bboxes_ignore: (1)to tensor, (2)to BaseDataElement
    - gt_labels: (1)to tensor, (2)to BaseDataElement
    """

    def __init__(self, class_names, with_gt=True, with_label=True):
        super(DefaultFormatBundle3D, self).__init__()
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if 'pts' in results:
            assert isinstance(results['pts'], BasePoints)
            results['pts'] = BaseDataElement(data=results['pts'].tensor)

        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            results[key] = BaseDataElement(data=to_tensor(results[key]))

        if self.with_gt:
            # Clean GT bboxes in the final
            if 'gt_bboxes_mask' in results:
                gt_bboxes_mask = results['gt_bboxes_mask']
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][
                    gt_bboxes_mask]
                if 'gt_classes' in results:
                    results['gt_names_3d'] = results['gt_names_3d'][
                        gt_bboxes_mask]
                if 'centers2d' in results:
                    results['centers2d'] = results['centers2d'][
                        gt_bboxes_mask]
                if 'depths' in results:
                    results['depths'] = results['depths'][gt_bboxes_mask]
            if 'gt_bboxes_mask' in results:
                gt_bboxes_mask = results['gt_bboxes_mask']
                if 'gt_bboxes' in results:
                    results['gt_bboxes'] = results['gt_bboxes'][gt_bboxes_mask]
                results['gt_names'] = results['gt_names'][gt_bboxes_mask]
            if self.with_label:
                if 'gt_names' in results and len(results['gt_names']) == 0:
                    results['gt_labels'] = np.array([], dtype=np.int64)
                    results['attr_labels'] = np.array([], dtype=np.int64)
                elif 'gt_names' in results and isinstance(
                        results['gt_names'][0], list):
                    # gt_labels might be a list of list in multi-view setting
                    results['gt_labels'] = [
                        np.array([self.class_names.index(n) for n in res],
                                 dtype=np.int64) for res in results['gt_names']
                    ]
                elif 'gt_names' in results:
                    results['gt_labels'] = np.array([
                        self.class_names.index(n) for n in results['gt_names']
                    ],
                                                    dtype=np.int64)
                # we still assume one pipeline for one frame LiDAR
                # thus, the 3D name is list[string]
                if 'gt_names_3d' in results:
                    results['gt_labels_3d'] = np.array([
                        self.class_names.index(n)
                        for n in results['gt_names_3d']
                    ],
                                                       dtype=np.int64)
        results = super(DefaultFormatBundle3D, self).__call__(results)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str
    
@TRANSFORMS.register_module()
class CustomDefaultFormatBundle3D(DefaultFormatBundle3D):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor, (3)to BaseDataElement (stack=True)
    - proposals: (1)to tensor, (2)to BaseDataElement
    - gt_bboxes: (1)to tensor, (2)to BaseDataElement
    - gt_bboxes_ignore: (1)to tensor, (2)to BaseDataElement
    - gt_labels: (1)to tensor, (2)to BaseDataElement
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        results = super(CustomDefaultFormatBundle3D, self).__call__(results)
        results['gt_map_masks'] = DC(
            to_tensor(results['gt_map_masks']), stack=True)

        return results

@TRANSFORMS.register_module()
class VADFormatBundle3D(DefaultFormatBundle3D):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor, (3)to BaseDataElement (stack=True)
    - proposals: (1)to tensor, (2)to BaseDataElement
    - gt_bboxes: (1)to tensor, (2)to BaseDataElement
    - gt_bboxes_ignore: (1)to tensor, (2)to BaseDataElement
    - gt_labels: (1)to tensor, (2)to BaseDataElement
    """
    def __init__(self, class_names, with_gt=True, with_label=True, with_ego=True):
        super(VADFormatBundle3D, self).__init__(class_names, with_gt, with_label)
        self.with_ego = with_ego


    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        results = super(VADFormatBundle3D, self).__call__(results)
        # results['gt_map_masks'] = DC(to_tensor(results['gt_map_masks']), stack=True)
        if self.with_ego:
            if 'ego_his_trajs' in results:
                results['ego_his_trajs'] = DC(to_tensor(results['ego_his_trajs'][None, ...]), stack=True)
            if 'ego_fut_trajs' in results:
                results['ego_fut_trajs'] = DC(to_tensor(results['ego_fut_trajs'][None, ...]), stack=True)
            if 'ego_fut_masks' in results:
                results['ego_fut_masks'] = DC(to_tensor(results['ego_fut_masks'][None, None, ...]), stack=True)
            if 'ego_fut_cmd' in results:
                results['ego_fut_cmd'] = DC(to_tensor(results['ego_fut_cmd'][None, None, ...]), stack=True)
            if 'ego_lcf_feat' in results:
                results['ego_lcf_feat'] = DC(to_tensor(results['ego_lcf_feat'][None, None, ...]), stack=True)
            if 'gt_attr_labels' in results:
                results['gt_attr_labels'] = DC(to_tensor(results['gt_attr_labels']), cpu_only=False)
                
        return results

