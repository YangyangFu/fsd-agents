from typing import Optional, Union, List, Tuple
from collections import defaultdict
import torch 
import torch.nn as nn
import numpy as np

from mmengine.utils import is_seq_of
from mmengine.model import BaseDataPreprocessor
from mmengine.structures import BaseDataElement
from fsd.structures import TrajectoryData
from fsd.registry import MODELS

def stack_batch(data):
    """Stack a sequence of data of the same type/size at the new first dimension
    Currently supports list of torch.Tensor, list of numpy.ndarray and list of TrajectoryData.
    
    """
    if len(data) == 1:
        return data[0]
    
    if is_seq_of(data, torch.Tensor):
        return torch.stack(data, dim=0)
    elif is_seq_of(data, np.ndarray):
        return np.stack(data, axis=0)
    elif is_seq_of(data, TrajectoryData):
        return stack_batch_trajectory_data(data)
    elif is_seq_of(data, BaseDataElement):
        return stack_batch_data_element(data)
    else:
        raise ValueError(f"Unsupported data type {type(data)} for stacking")
    
def stack_batch_trajectory_data(data: List[TrajectoryData]) -> TrajectoryData:
    """ Stack a list of trajectory data to a single trajectory data
    
    Typically used for stacking trajectory data
    
    """
    assert is_seq_of(data, TrajectoryData), f"Expecting a list of TrajectoryData, \
                but got {type(data)}."
    
    keys = data[0].all_keys()
    metainfo_fields = data[0]._metainfo_fields 
    data_fields = data[0]._data_fields
    stack_ = TrajectoryData()
    stack_meta = defaultdict(list)
    stack_data = defaultdict(list)
    for d in data:
        for key in keys:
            if key in metainfo_fields:
                stack_meta[key].append(d.get(key))
            elif key in data_fields:
                v = d.get(key)
                if isinstance(v, torch.Tensor) or \
                        isinstance(v, np.ndarray):
                    stack_data[key].append(v)
                else:
                    raise ValueError(f"Unsupported data type {type(v)} for stacking")
                
    for key in stack_meta:
        stack_.set_metainfo({key: stack_meta[key]})
    for key in stack_data:
        stack_.set_data({key: stack_batch(stack_data[key])})
    
    return stack_

def stack_batch_data_element(data_element: List[BaseDataElement]) -> BaseDataElement:
    """ Stack a list of data element to a single data element
    
    Typically used for stacking image data or point cloud data 
    
    """
    assert is_seq_of(data_element, BaseDataElement), f"Expecting a list of BaseDataElement, \
                but got {type(data_element)}."
    
    keys = data_element[0].all_keys()
    metainfo_fields = data_element[0]._metainfo_fields 
    data_fields = data_element[0]._data_fields
    stack_ = BaseDataElement()
    stack_meta = defaultdict(list)
    stack_data = defaultdict(list)
    for d in data_element:
        for key in keys:
            if key in metainfo_fields:
                stack_meta[key].append(d.get(key))
            elif key in data_fields:
                v = d.get(key)
                if isinstance(v, torch.Tensor) or \
                        isinstance(v, np.ndarray) or \
                            isinstance(v, TrajectoryData):
                    stack_data[key].append(v)
                else:
                    raise ValueError(f"Unsupported data type {type(v)} for stacking")
                
    for key in stack_meta:
        stack_.set_metainfo({key: stack_meta[key]})
    for key in stack_data:
        stack_.set_data({key: stack_batch(stack_data[key])})
    
    return stack_


@MODELS.register_module()
class PlanningDataPreprocessor(BaseDataPreprocessor):
    """Image and LiDAR data preprocessor for planning models
    
    It will perform the following processing:
    
    - Collate and move image and point cloud data to the target device.

    - 1) For image data:
      - Stack images in inputs to batch_imgs.
      - Do batch augmentations during training.

    - 2) For point cloud data:

      - If no voxelization, directly return list of point cloud data.
      - If voxelization is applied, voxelize point cloud according to
        ``voxel_type`` and obtain ``voxels``.

    """
    def __init__(self, 
                 non_blocking: Optional[bool] = False,
                 batch_augments: Optional[List[dict]] = None) -> None:
        super().__init__(non_blocking=non_blocking)

        
        # any batch augumentations
        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None
         
    def forward(self,
                data: Union[dict, List[dict]],
                training: bool = False) -> Union[dict, List[dict]]:
        """Perform data processing before sending to model:
            - cast data to the target device.
            - processing image data, such as generating images, etc
            - processing point data, such as generating voxels, 
            - stacking batch data etc.

        Args:
            data (dict or List[dict]): Data from dataloader. The dict contains
                the whole batch data, when it is a list[dict], the list
                indicates test time augmentation.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict or List[dict]: Data in the same format as the model input.
        """
        # process img 
        data = self.process_images(data, training)
        # process pts 
        data = self.process_points(data, training)
        
        # process annotation
        
        # stack
        data = self.collate_data(data)
        # cast data
        
        if self.batch_augments is not None:
            for aug in self.batch_augments:
                data = aug(data)
        
        return {'inputs': data['inputs'], 'data_samples': data['data_samples']} 

    def process_images(self, 
                       data : Union[dict, List[dict]], 
                       training : bool = False) -> Union[dict, List[dict]]:
        """ Image related preprocessing
        """
        return data
    
    def process_points(self, 
                       data : Union[dict, List[dict]], 
                       training : bool = False) -> Union[dict, List[dict]]:
        """ Point cloud related preprocessing
        """
        return data

    def collate_data(self, data: dict) -> dict:
        """Copy data to the target device and perform normalization, padding
        and bgr2rgb conversion and stack based on ``BaseDataPreprocessor``.

        Collates the data sampled from dataloader into a list of dict and list
        of labels, and then copies tensor to the target device.

        Args:
            data (dict): Data sampled from dataloader.

        Returns:
            dict: Data in the same format as the model input.
        """
        # TODO: cast data after or before stacking
        data = self.cast_data(data)  # type: ignore

        if 'img' in data['inputs']:
            data['inputs']['img'] = stack_batch(data['inputs']['img'])
        
        if 'pts' in data['inputs']:
            processed = stack_batch(data['inputs']['pts'])
            processed.data = processed.data.float()/255.0
            data['inputs']['pts'] = processed
            
        return data


 