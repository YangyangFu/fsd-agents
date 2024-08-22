from collections.abc import Sized
from typing import Any, List, Union
from fsd.utils.type import Array 

import torch
import numpy as np 
import itertools

from mmengine.structures import BaseDataElement, InstanceData

BoolTypeTensor: Union[Any]
LongTypeTensor: Union[Any]

IndexType: Union[Any] = Union[str, slice, int, list, np.ndarray]

#TODO: add rotation info/timestamps
#TODO: add mask update support for data pipeline, e.g., TrajectoryData.merge_mask(mask)
#TODO: better support for timestams. considering distinguish between timestamps and timestamps_diff

class TrajectoryData(BaseDataElement):
    """ Data structure for trajectory annotations or predictions
    
    For one frame, the trajectory data can be used to store annotations or predictions for all instances in the frame.
    This typically lead to a 2D tensor with shape (2, N) for xy coordinates, a 1D tensor with shape (N,) for mask, 
    and a 1D tensor with shape (N,) for timestamps, where N represents the number of instances in the frame.
    To encode the temporal information, multiple trajectory can be stacked along the last dimension, leading to 
    a 3D tensor with shape (N, 2, T) for xy coordinates, a 2D tensor with shape (N, T) for mask, and a 2D tensor with shape (N, T) for timestamps.
    
    Subclass of :class:`BaseDataElement`. All data items in `data_fields` should have the same length (use the last dimension here).
    TrajectoryData also supports slicing, indexing and arithmetic addition and subtraction.
    
    Attributes:
        xy (torch.Tensor): The xy coordinates of the trajectory. Shape (2, N).
        mask (torch.Tensor): mask with 1 for valid points and 0 for invalid points. Shape (N,).
    
    """
    
    def __setattr__(self, name: str, value: Union[torch.Tensor, np.ndarray]):
        """setattr is only used to set data.

        The value must have the attribute of `__len__` and have the same length
        of `InstanceData`.
        """
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a '
                                        'private attribute, which is immutable.')

        else:
            assert isinstance(value, (torch.Tensor, np.ndarray)), \
                f'Can not set {type(value)}, only support' \
                f' {(torch.Tensor, np.ndarray)}'

            if len(self) > 0:
                assert len(value) == len(self), 'The length of ' \
                                                f'values {len(value)} is ' \
                                                'not consistent with ' \
                                                'the length of this ' \
                                                ':obj:`InstanceData` ' \
                                                f'{len(self)}'
            super().__setattr__(name, value)
    
    __setitem__ = __setattr__
        
    def __getitem__(self, item: IndexType) -> 'TrajectoryData':
        """
        Args:
            item (str, int, list, :obj:`slice`, :obj:`numpy.ndarray`,
                :obj:`torch.LongTensor`, :obj:`torch.BoolTensor`):
                Get the corresponding values according to item along the first dimension.

        Returns:
            :obj:`PointData`: Corresponding values.
        """
        if isinstance(item, list):
            item = np.array(item)
            
        if isinstance(item, np.ndarray):
            # The default int type of numpy is platform dependent, int32 for
            # windows and int64 for linux. `torch.Tensor` requires the index
            # should be int64, therefore we simply convert it to int64 here.
            # Mode details in https://github.com/numpy/numpy/issues/9464
            item = item.astype(np.int64) if item.dtype == np.int32 else item
            item = torch.from_numpy(item)
        assert isinstance(
            item, (str, slice, int, torch.LongTensor, torch.cuda.LongTensor,
                   torch.BoolTensor, torch.cuda.BoolTensor))

        if isinstance(item, str):
            return getattr(self, item)

        if isinstance(item, int):
            if item >= len(self) or item < -len(self):  # type: ignore
                raise IndexError(f'Index {item} out of range!')
            else:
                # keep the dimension
                item = slice(item, None, len(self))

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, torch.Tensor):
            assert item.dim() == 1, 'Only support to get the' \
                                    ' values along the first dimension.'
            if isinstance(item, (torch.BoolTensor, torch.cuda.BoolTensor)):
                assert len(item) == len(self), 'The shape of the ' \
                                               'input(BoolTensor) ' \
                                               f'{len(item)} ' \
                                               'does not match the shape ' \
                                               'of the indexed tensor ' \
                                               'in results_field ' \
                                               f'{len(self)} at ' \
                                               'first dimension.'

            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    new_data[k] = v[item]
                elif isinstance(v, np.ndarray):
                    new_data[k] = v[item.cpu().numpy()]
                elif isinstance(
                        v, (str, list, tuple)) or (hasattr(v, '__getitem__')
                                                   and hasattr(v, 'cat')):
                    # convert to indexes from BoolTensor
                    if isinstance(item,
                                  (torch.BoolTensor, torch.cuda.BoolTensor)):
                        indexes = torch.nonzero(item).view(
                            -1).cpu().numpy().tolist()
                    else:
                        indexes = item.cpu().numpy().tolist()
                    slice_list = []
                    if indexes:
                        for index in indexes:
                            slice_list.append(slice(index, None, len(v)))
                    else:
                        slice_list.append(slice(None, 0, None))
                    r_list = [v[s] for s in slice_list]
                    if isinstance(v, (str, list, tuple)):
                        new_value = r_list[0]
                        for r in r_list[1:]:
                            new_value = new_value + r
                    else:
                        new_value = v.cat(r_list)
                    new_data[k] = new_value
                else:
                    raise ValueError(
                        f'The type of `{k}` is `{type(v)}`, which has no '
                        'attribute of `cat`, so it does not '
                        'support slice with `bool`')
        else:
            # item is a slice
            for k, v in self.items():
                new_data[k] = v[item]
        return new_data  # type: ignore
    
    
    def __len__(self) -> int:
        """Get the length of the attribute.

        Returns:
            int: The length of the attribute.
        """
        if len(self._data_fields) > 0:
            return len(self.values()[0])
        
        return 0
    
    ### ----------------------------------------------
    ### Properties
    @property 
    def num_steps(self) -> int:
        """The number of steps in the trajectory
        
        Returns:
            int: The number of steps in the trajectory
        """
        if len(self) > 0:
            if self.xy.dim() == 3:
                return self.xy.shape[2]
            elif self.xy.dim() == 2:
                return 1
        return 0
        
    @property
    def num_instances(self) -> int:
        """The number of instances in the trajectory
        
        Returns:
            int: The number of instances in the trajectory
        """
        return len(self)
    
    @property
    def xy(self) -> Array:
        return self._xy
    
    @xy.setter
    def xy(self, value: Array):
        """xy coordinates of the trajectory
        
        Args:
            value (torch.Tensor): The xy coordinates of the trajectory with a dim of 2 or 3. 
                Shape (N, 2, ...). 
                if dim == 2, then shape is (N, 2). 
                if dim == 3, then shape is (N, 2, T).
                if dim == 4, then shape is (B, N, 2, T).
        """
        if isinstance(value, (torch.Tensor, np.ndarray)) and (value.ndim == 1):
            value = value[None, ...]
            
        assert isinstance(value, (torch.Tensor, np.ndarray)) and (value.ndim == 2 or value.ndim == 3 or value.ndim==4), \
            "xy coordinates should be a 2D or 3D or 3D tensor"
        if value.ndim == 2:
            assert value.shape[1] == 2, "The last dimension of xy should be 2"
        elif value.ndim > 2:
            assert value.shape[-2] == 2, "The second last dimension of xy should be 2"
            
        if hasattr(self, 'mask'):
            assert value.ndim - self.mask.ndim == 1, "xy dimension has to be greater than mask dimension"
            
        self.set_field(value, '_xy', dtype=type(value))
        
    @xy.deleter
    def xy(self):
        del self._xy
    
    @property
    def mask(self) -> torch.Tensor:
        """ mask with 1 for valid points and 0 for invalid points 
        """
        return self._mask
    @mask.setter
    def mask(self, value: torch.Tensor):
        """mask of the trajectory
        
        Args:
            value (torch.Tensor): The mask of the trajectory with a dim of 1 or 2 or 3.
                Shape (N, ...). 
                If dim == 1, then shape is (N,). 
                If dim == 2, then shape is (N, T).
                if dim == 3, then shape is (B, N, T).
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)) and (value.ndim == 1 or value.ndim == 2 or value.ndim == 3), \
            "mask should be a 1D or 2D or 3D tensor"
        if hasattr(self, 'xy'):
            assert self.xy.ndim - value.ndim == 1, "mask dimension has to be less than xy dimension"
        
        self.set_field(value, '_mask', dtype=type(value))
    @mask.deleter
    def mask(self):
        del self._mask
    
    ### ----------------------------------------------
    ### overloaded operators
    def __add__(self, other: 'TrajectoryData') -> 'TrajectoryData':
        """Add two TrajectoryData objects
        
        Args:
            other (TrajectoryData): Another TrajectoryData object
        
        Returns:
            TrajectoryData: A new TrajectoryData object with the added data.
        """
        raise NotImplementedError("Addition of TrajectoryData objects is not supported")
     
    def __sub__(self, other: 'TrajectoryData') -> 'TrajectoryData':
        """Subtract two TrajectoryData objects
        
        Args:
            other (TrajectoryData): Another TrajectoryData object
        
        Returns:
            TrajectoryData: A new TrajectoryData object with the subtracted data.
        """
        # at least one of them has a length of 1 or same length
        assert len(self) == 1 or len(other) == 1 or (len(self) == len(other)), \
            "At least one of the TrajectoryData objects should have a length of 1, or \
                they should have the same length"
                
        # subtract the data
        xy = self.xy - other.xy 
        # merge mask
        mask = self.mask * other.mask
        
        return TrajectoryData(xy=xy, mask=mask)
    
    ### ----------------------------------------------
    ### Methods   
    @staticmethod
    def cat(trajectory_list: List['TrajectoryData']) -> 'TrajectoryData':
        """Concat the instances of all :obj:`TrajectoryData`.

        Note: To ensure that cat returns as expected, make sure that
        all elements in the list must have exactly the same keys.

        Args:
            instances_list (list[:obj:`TrajectoryData`]): A list
                of :obj:`TrajectoryData`.

        Returns:
            :obj:`TrajectoryData`
        """
        if type(trajectory_list) is not list:
            raise TypeError('Input must be a list')
                
        assert all(
            isinstance(results, TrajectoryData) for results in trajectory_list)
        assert len(trajectory_list) > 0
        
        if len(trajectory_list) == 1:
            return trajectory_list[0]

        # metainfo and data_fields must be exactly the
        # same for each element to avoid exceptions.
        field_keys_list = [
            instances.all_keys() for instances in trajectory_list
        ]
        assert len({len(field_keys) for field_keys in field_keys_list}) \
               == 1 and len(set(itertools.chain(*field_keys_list))) \
               == len(field_keys_list[0]), 'There are different keys in ' \
                                           '`trajectory_list`, which may ' \
                                           'cause the cat operation ' \
                                           'to fail. Please make sure all ' \
                                           'elements in `trajectory_list` ' \
                                           'have the exact same key.'
        # must have the same steps in temporal dimension
        assert all(trajectory_list[0].num_steps == trajectory.num_steps \
                for trajectory in trajectory_list), \
                    "All TrajectoryData objects should have the same number of steps"
        
        new_data = trajectory_list[0].__class__(
            metainfo=trajectory_list[0].metainfo)
        for k in trajectory_list[0].keys():
            values = [results[k] for results in trajectory_list]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                new_values = torch.cat(values, dim=0)
            elif isinstance(v0, np.ndarray):
                new_values = np.concatenate(values, axis=0)
            elif isinstance(v0, (str, list, tuple)):
                new_values = v0[:]
                for v in values[1:]:
                    new_values += v
            else:
                raise ValueError(
                    f'The type of `{k}` is `{type(v0)}` which has no '
                    'attribute of `cat`')
            new_data[k] = new_values
        return new_data

    @staticmethod
    def stack(trajectory_list: List['TrajectoryData']) -> 'TrajectoryData':
        """Concat the instances of all :obj:`TrajectoryData`along temporal dimension.
        

        Note: To ensure that cat returns as expected, make sure that
        all elements in the list must have exactly the same keys.

        Args:
            instances_list (list[:obj:`TrajectoryData`]): A list
                of :obj:`TrajectoryData`.

        Returns:
            :obj:`TrajectoryData`
        """
        if type(trajectory_list) is not list:
            raise TypeError('Input must be a list')
                
        assert all(
            isinstance(results, TrajectoryData) for results in trajectory_list)
        assert len(trajectory_list) > 0
        
        if len(trajectory_list) == 1:
            return trajectory_list[0]

        # metainfo and data_fields must be exactly the
        # same for each element to avoid exceptions.
        field_keys_list = [
            instances.all_keys() for instances in trajectory_list
        ]
        assert len({len(field_keys) for field_keys in field_keys_list}) \
               == 1 and len(set(itertools.chain(*field_keys_list))) \
               == len(field_keys_list[0]), 'There are different keys in ' \
                                           '`trajectory_list`, which may ' \
                                           'cause the cat operation ' \
                                           'to fail. Please make sure all ' \
                                           'elements in `trajectory_list` ' \
                                           'have the exact same key.'
        # must have the same number of instances
        assert all(len(trajectory_list[0]) == len(traj) for traj in trajectory_list), \
            "All TrajectoryData objects should have the same number of instances"
            
        new_data = trajectory_list[0].__class__(
            metainfo=trajectory_list[0].metainfo)
        for k in trajectory_list[0].keys():
            values = [results[k] for results in trajectory_list]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                if trajectory_list[0].num_steps == 1: # no temporal dimension
                    new_values = torch.stack(values, dim=-1)
                else:
                    new_values = torch.cat(values, dim=-1)
                    
            elif isinstance(v0, np.ndarray):
                if trajectory_list[0].num_steps == 1:
                    new_values = np.stack(values, axis=-1)
                else:    
                    new_values = np.concatenate(values, axis=-1)
                    
            elif isinstance(v0, (str, list, tuple)):
                new_values = v0[:]
                for v in values[1:]:
                    new_values += v
            else:
                raise ValueError(
                    f'The type of `{k}` is `{type(v0)}` which has no '
                    'attribute of `cat`')
            new_data[k] = new_values
        return new_data
    
    @classmethod
    def interpolate(self, timestamps)-> 'TrajectoryData':
        """Interpolate the trajectory data to the given timestamps
        
        Args:
            timestamps (torch.Tensor): The timestamps to interpolate the trajectory data to.
        
        Returns:
            TrajectoryData: The interpolated trajectory data.
        """
        raise NotImplementedError("Interpolation of TrajectoryData is not supported")