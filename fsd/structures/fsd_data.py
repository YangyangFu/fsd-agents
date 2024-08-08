import torch
import numpy as np 
from collections.abc import Sized
from typing import Any, List, Union

from mmengine.device import get_device
from mmengine.structures import BaseDataElement, InstanceData

BoolTypeTensor: Union[Any]
LongTypeTensor: Union[Any]

if get_device() == 'npu':
    BoolTypeTensor = Union[torch.BoolTensor, torch.npu.BoolTensor]
    LongTypeTensor = Union[torch.LongTensor, torch.npu.LongTensor]
elif get_device() == 'mlu':
    BoolTypeTensor = Union[torch.BoolTensor, torch.mlu.BoolTensor]
    LongTypeTensor = Union[torch.LongTensor, torch.mlu.LongTensor]
elif get_device() == 'musa':
    BoolTypeTensor = Union[torch.BoolTensor, torch.musa.BoolTensor]
    LongTypeTensor = Union[torch.LongTensor, torch.musa.LongTensor]
else:
    BoolTypeTensor = Union[torch.BoolTensor, torch.cuda.BoolTensor]
    LongTypeTensor = Union[torch.LongTensor, torch.cuda.LongTensor]

IndexType: Union[Any] = Union[str, slice, int, list, LongTypeTensor,
                              BoolTypeTensor, np.ndarray]

#TODO: add rotation info
#TODO: better support for timestams. considering distinguish between timestamps and timestamps_diff

class TrajectoryData(BaseDataElement):
    """ Data structure for trajectory annotations or predictions
    
    Subclass of :class:`BaseDataElement`. All data items in `data_fields` should have the same length.
    TrajectoryData also supports slicing, indexing and arithmetic addition and subtraction.
    
    Attributes:
        xy (torch.Tensor): The xy coordinates of the trajectory. Shape (2, N).
        mask (torch.Tensor): mask with 1 for valid points and 0 for invalid points. Shape (N,).
        timestamps (torch.Tensor): The timestamps of the trajectory. Shape
    
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
                Get the corresponding values according to item.

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
    def xy(self) -> torch.Tensor:
        return self._xy
    @xy.setter
    def xy(self, value: torch.Tensor):
        """xy coordinates of the trajectory
        
        Args:
            value (torch.Tensor): The xy coordinates of the trajectory. Shape (N, 2).
        """
        if isinstance(value, torch.Tensor) and value.dim() == 1:
            value = value.unsqueeze(0)
            
        assert isinstance(value, torch.Tensor) and value.dim() == 2 and value.shape[-1] == 2, \
            "xy coordinates should be a 2D tensor with shape (N, 2)"
        
        self.set_field(value, '_xy', dtype=torch.Tensor)
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
            value (torch.Tensor): The mask of the trajectory. Shape (N,).
        """
        assert isinstance(value, torch.Tensor) and value.dim() == 1, \
            "mask should be a 1D tensor"
        
        self.set_field(value, '_mask', dtype=torch.Tensor)
    @mask.deleter
    def mask(self):
        del self._mask
    
    @property
    def timestamps(self) -> torch.Tensor:
        return self._timestamps
    @timestamps.setter
    def timestamps(self, value: torch.Tensor):
        """timestamps of the trajectory
        
        Args:
            value (torch.Tensor): The timestamps of the trajectory. Shape (N,).
        """
        assert isinstance(value, torch.Tensor) and value.dim() == 1, \
            "timestamps should be a 1D tensor"
        
        self.set_field(value, '_timestamps', dtype=torch.Tensor)
    @timestamps.deleter
    def timestamps(self):
        del self._timestamps
        
    
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
        timestamps = self.timestamps - other.timestamps
        # merge mask
        mask = self.mask * other.mask
        
        return TrajectoryData(xy=xy, mask=mask, timestamps=timestamps)
    
    ### ----------------------------------------------
    ### Methods
    
