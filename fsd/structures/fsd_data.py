from collections.abc import Sized
from typing import Any, List, Union
import warnings
from fsd.utils.type import Array 

import torch
import numpy as np 

from mmengine.structures import BaseDataElement

BoolTypeTensor: Union[Any]
LongTypeTensor: Union[Any]
IndexType: Union[Any] = Union[str, slice, int, list, np.ndarray]

#TODO: (1) May need better implementaion to support concatenation of TrajectoryData,
# invoving the merging of different timestamps, num_past_steps, etc.
# (2) assertion of consistency among the data, mask and meta
class TrajectoryData(BaseDataElement):
    """ Data structure for trajectory annotations or predictions for one agent
    
    For one frame, the trajectory data can be used to store annotations or predictions for each in the frame.
    This typically lead to a 2D tensor with shape (T, d), a 1D tensor with shape (T,) for mask.
    Attributes are:
        - time: float, the timestamp of current step
        - time_step: float, the time step between two steps
        - past_steps: int, the number of past steps in the trajectory
        - future_steps: int, the number of future steps in the trajectory
        - data (torch.Tensor): The data coordinates of the trajectory. Shape (T, 4).
        - mask (torch.Tensor): mask with 1 for valid points and 0 for invalid points. Shape (T,).
    
    Subclass of :class:`BaseDataElement`. All data items in `data_fields` should have the same length (use the last dimension here).
    TrajectoryData also supports slicing, indexing and arithmetic addition and subtraction.
    
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
    def data(self) -> Array:
        if hasattr(self, '_data'):
            return self._data
        
        return None
        
    @data.setter
    def data(self, value: Array):
        """ Trajectory data
        
        Args:
            value (torch.Tensor): The data coordinates of the trajectory has to be a 2D tensor/array.
                Shape (T, d). 
        """
        if isinstance(value, (torch.Tensor, np.ndarray)) and (value.ndim == 1):
            value = value[None, ...]
            
        assert isinstance(value, (torch.Tensor, np.ndarray)) and (value.ndim == 2), \
            "data coordinates should be 2D"
        
        # TODO: avoid checking with meta info as slicing/indexing will lead to inconsistency
        # assertion on length
        #if hasattr(self, 'num_past_steps') or hasattr(self, 'num_future_steps'):
        #    npast = self.num_past_steps if hasattr(self, 'num_past_steps') else 0
        #    nfuture = self.num_future_steps if hasattr(self, 'num_future_steps') else 0
        #    nsteps = npast + nfuture + 1
        #    assert value.shape[0] == nsteps, \
        #        "The trajectory steps in the data coordinates is not consistent with the meta info"
        
        self.set_field(value, '_data', dtype=type(value))
        
    @data.deleter
    def data(self):
        del self._data
    
    @property
    def mask(self) -> torch.Tensor:
        """ mask with 1 for valid points and 0 for invalid points 
        """
        if hasattr(self, '_mask'):
            return self._mask
        return np.ones(len(self))
    
    @mask.setter
    def mask(self, value: torch.Tensor):
        """mask of the trajectory.
        
        Mask can be 1D or 2D tensor.
            if 1D tensor, the mask is applied to the first dimension of the trajectory.
            if 2D tensor, the mask is applied to the first two dimension of the trajectory.
        
        Args:
            value (torch.Tensor): The mask of the trajectory with a dim of 1 or 2.

        """
        assert isinstance(value, (torch.Tensor, np.ndarray)) and (value.ndim == 1 or value.ndim == 2), \
            "mask should be a 1D or 2D tensor"
        
        # assertion on length
        #if hasattr(self, 'num_past_steps') or hasattr(self, 'num_future_steps'):
        #    npast = self.num_past_steps if hasattr(self, 'num_past_steps') else 0
        #    nfuture = self.num_future_steps if hasattr(self, 'num_future_steps') else 0
        #    nsteps = npast + nfuture + 1
        #    assert value.shape[0] == nsteps, \
        #        "The trajectory steps in the data coordinates is not consistent with the meta info"
        
        self.set_field(value, '_mask', dtype=type(value))
        
    @mask.deleter
    def mask(self):
        del self._mask
    
    ### ----------------------------------------------
    ### Methods   
    @staticmethod
    def cat(trajectory_list: List['TrajectoryData']) -> 'TrajectoryData':
        """Concat a list of TrajectoryData

        Returns:
            :obj:`TrajectoryData`
        """
        NotImplementedError("Concatenation of TrajectoryData is not supported")

    @classmethod
    def interpolate(self, timestamps)-> 'TrajectoryData':
        """Interpolate the trajectory data to the given timestamps
        
        Args:
            timestamps (torch.Tensor): The timestamps to interpolate the trajectory data to.
        
        Returns:
            TrajectoryData: The interpolated trajectory data.
        """
        raise NotImplementedError("Interpolation of TrajectoryData is not supported")


class MultiModalTrajectoryData(TrajectoryData):
    """Multi-modal trajectory data structure to support multiple trajectories for one instance
    
    Shape of the trajectory data is (T, M, 4) for [x,y,z, yaw], and (T, M) for mask.

    Args:
        TrajectoryData (_type_): _description_
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    
    @property
    def num_modalities(self) -> int:
        """The number of modalities in the trajectory
        
        Returns:
            int: The number of modalities in the trajectory
        """
        if hasattr(self, '_data'):
            if self._data.ndim == 3:
                return self._data.shape[1]
            elif self._data.ndim == 2:
                return 1
        warnings.warn("No data in the TrajectoryData object for inferring the number of modalities")
        return 0
        
    @num_modalities.setter
    def num_modalities(self, value: int):
        
        if hasattr(self, '_data'):
            if self._data.ndim == 3:
                assert value == self._data.shape[1], "The number of modalities is not consistent with the data coordinates"
            elif self._data.ndim == 2:
                assert value == 1, "The number of modalities is not consistent with the data coordinates"
        
        self.set_field(value, '_num_modalities', dtype=int)
    
    @property
    def data(self) -> Array:
        if hasattr(self, '_data'):
            return self._data
        
        return None
        
    @data.setter
    def data(self, value: Array):
        """ Multi-modal trajectory data
        
        Args:
            value (torch.Tensor): The data coordinates of the trajectory has to be a 3D tensor/array.
                Shape (T, M, d). 
        """
        if isinstance(value, (torch.Tensor, np.ndarray)) and (value.ndim == 1):
            value = value[None, None, ...]
        if isinstance(value, (torch.Tensor, np.ndarray)) and (value.ndim == 2):
            value = value[None, ...]
            
        assert isinstance(value, (torch.Tensor, np.ndarray)) and (value.ndim == 3), \
            "data coordinates should be 3D"
            
        if value.ndim == 3:
            if hasattr(self, '_num_modalities'):
                assert value.shape[1] == self.num_modalities, "The number of modalities is not consistent with the data coordinates"
            
        self.set_field(value, '_data', dtype=type(value))
        
    @data.deleter
    def data(self):
        del self._data
    
    @property
    def mask(self) -> torch.Tensor:
        """ mask with 1 for valid points and 0 for invalid points 
        """
        if hasattr(self, '_mask'):
            return self._mask
        return np.ones(len(self))
    
    @mask.setter
    def mask(self, value: torch.Tensor):
        """mask of the trajectory.
        
        Mask can be 1D or 2D tensor.
            if 1D tensor, the mask is applied to the first dimension of the trajectory.
            if 2D tensor, the mask is applied to the first two dimension of the trajectory.
            if 3D tensor, the mask is applied to the first three dimension of the trajectory.
        
        Args:
            value (torch.Tensor): The mask of the trajectory with a dim of 1 or 2 or 3.

        """
        assert isinstance(value, (torch.Tensor, np.ndarray)) and (value.ndim == 1 or value.ndim == 2 or value.ndim==3), \
            "mask should be a 1D or 2D or 3D tensor"
        
        self.set_field(value, '_mask', dtype=type(value))
        
    @mask.deleter
    def mask(self):
        del self._mask
    
    @property
    def scores(self) -> torch.Tensor:
        """The scores of the trajectory
        
        Returns:
            torch.Tensor: The scores of the trajectory
        """
        if hasattr(self, '_scores'):
            return self._scores
        return None
    
    @scores.setter
    def scores(self, value: torch.Tensor):
        """Scores of the trajectory
        
        Args:
            value (torch.Tensor): The scores of the trajectory
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Scores should be a tensor"
        
        assert value.ndim == 1, "Scores should be a 1D tensor"
        assert value.shape[0] == self.num_modalities, \
            "The number of scores are not consistent with the number of modalities in the trajectory"
        
        self.set_field(value, '_scores', dtype=type(value))
    
    @scores.deleter
    def scores(self):
        del self._scores
    