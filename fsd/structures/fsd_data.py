from collections.abc import Sized
from typing import Any, List, Union
import warnings

import torch
import numpy as np 

from mmengine.utils import is_seq_of
from mmengine.structures import BaseDataElement, InstanceData
from mmdet3d.structures import BaseInstance3DBoxes

Array = Union[torch.Tensor, np.ndarray]
BoolTypeTensor: Union[Any]
LongTypeTensor: Union[Any]
IndexType: Union[Any] = Union[str, slice, int, list, np.ndarray]

#TODO: (1) May need better implementaion to support concatenation of TrajectoryData,
# invoving the merging of different timestamps, num_past_steps, etc.
# (2) assertion of consistency among the data, mask and meta
class Trajectory(BaseDataElement):
    """ Data structure for trajectory annotations or predictions
    
    The default shape of the trajectory data is (N, M, T, 4) for [x,y,z, yaw], where
        - N is the number of instances
        - M is the number of modalities
        - T is the number of steps in the trajectory
    The default shape of the mask is (T, ).
     
    Key attributes
        - metainfo (dict): The meta information of the trajectory.
            - steps (int): The number of steps in the trajectory.
            - modalities (int): The number of modalities in the trajectory.
            - dim (int): The dimension of the trajectory.
    
    Subclass of :class:`BaseDataElement`. All data items in `data_fields` should have the same length (use the last dimension here).
    TrajectoryData also supports slicing, indexing and arithmetic addition and subtraction.
    
    """
    def __init__(self, *args, **kwargs):
        """Initialize the TrajectoryData.
        Args:
            mode (str): The mode of the trajectory data.
                - 'accumulated': The trajectory data is accumulated over time, 
                    i.e., the absolute position of the trajectory.
                - 'difference': The trajectory data is the difference 
                    between two consecutive points with 0 as the given center.
                    i.e., the relative position of the trajectory.
        """
        warnings.warn(
            f"{self.__class__.__name__} is deprecated and will be removed in a future version.",
            category=DeprecationWarning,
            stacklevel=2
        )
                
        super().__init__(*args, **kwargs)


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
        
    def __getitem__(self, item: IndexType) -> 'Trajectory':
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
        assert isinstance(value, (torch.Tensor, np.ndarray)) and \
            (value.ndim == 2 or value.ndim == 3 or value.ndim == 4), \
            f"data coordinates should be either 2D or 3D or 4D, but got {value.ndim}D."
        
        # to (N, M, T, d)
        if value.ndim == 2:
            value = value[None, None, ...]
        elif value.ndim == 3:
            value = value[None, ...]
        
        # if accumulated, fill the nan values: mask=false    
        if self.get('mode') == 'accumulated' and self.get('mask') is not None:
            value = self._fill_nan(value, self.mask)
        # save to data
        self.set_field(value, '_data', dtype=type(value)) 
    @data.deleter
    def data(self):
        del self._data
    
    @property 
    def num_steps(self) -> int:
        return self._data.shape[-2] if hasattr(self, '_data') else 0
    @property
    def num_modalities(self) -> int:
        """The number of modalities in the trajectory
        
        Returns:
            int: The number of modalities in the trajectory
        """
        return self._data.shape[1] if hasattr(self, '_data') else 0
    @property
    def num_dims(self) -> int:
        """The number of dimensions in the trajectory
        
        Returns:
            int: The number of dimensions in the trajectory
        """
        return self._data.shape[-1] if hasattr(self, '_data') else 0
    
    ## ----------------------------------------------
    ## Methods
    def cumsum(self, start: Union[torch.Tensor, np.ndarray] = None, 
               dim: int = -2) -> 'Trajectory':
        """Cumulative sum of the trajectory data along the given dimension.
        
        Args:
            dim (int): The dimension to perform the cumulative sum. Default is -2.
        
        Returns:
            Trajectory: The cumulative sum of the trajectory data.
        """
        if hasattr(self, '_data'):
            acc = self._data.cumsum(dim=dim)
        if start is not None:
            acc += start        
        
        # save to data
        self.data = acc 
        
        return self
        
    #TODO: need refine
    def _fill_nan(self, data, mask)-> 'Trajectory':
        """Fill the missing trajectory data between steps using interpolation
        """
        data_type = type(data)
        mask = mask.astype(bool)
        
        if data is None or mask is None:
            return 
        
        # to numpy
        if data_type == torch.Tensor:
            data = data.cpu().numpy()

        rows, cols = data.shape[:2]
        index = np.arange(rows)
        index_valid = index[mask]
        
        # fill the invalid data with interpolation/extrapolation
        for col in range(cols):
            col_data = data[:, col]        
            data[:, col] = np.interp(index, index_valid, col_data[mask])
            
        # save back to the data
        if data_type == torch.Tensor:
            data = torch.from_numpy(data)
        
        return data
        
    def set_mode(self, mode: str):
        """Set the mode of the trajectory data
        
        Args:
            mode (str): The mode of the trajectory data.
                - 'accumulated': The trajectory data is accumulated over time, 
                    i.e., the absolute position of the trajectory.
                - 'difference': The trajectory data is the difference 
                    between two consecutive points with 0 as the given center.
                    i.e., the relative position of the trajectory.
        """
        if mode not in ['accumulated', 'difference']:
            raise ValueError("The mode of the trajectory data should be either 'accumulated' or 'difference'")
        
        self.set_field(mode, 'mode', field_type='metainfo')
      
    def convert_to_mode(self, target_mode) -> 'Trajectory':
        """Convert the trajectory data to the target mode
        
        Args:
            target_mode (str): The target mode of the trajectory data.
                - 'accumulated': The trajectory data is accumulated over time, 
                    i.e., the absolute position of the trajectory.
                - 'difference': The trajectory data is the difference 
                    between two consecutive points with 0 as the given center.
                    i.e., the relative position of the trajectory.
        
        Returns:
            TrajectoryData: The converted trajectory data.
        """
        if self.get('mode') == target_mode:
            return 
        
        if self.get('mode') == 'accumulated':
            self.set_mode(target_mode)
            orig_center = self.get('center')
            # convert to difference
            if isinstance(self.data, np.ndarray):
                past_diff = np.diff(self.data[:self.num_past_steps+1, :], axis=0)
                future_diff = np.diff(self.data[self.num_past_steps:, :], axis=0)
                data = np.concatenate((past_diff, np.zeros_like(orig_center).reshape(1,-1), future_diff), axis=0)
                # keep the center position for converting to accumulated mode
                self.data = data
                self.set_center(orig_center)
            elif isinstance(self.data, torch.Tensor):
                past_diff = torch.diff(self.data[:self.num_past_steps+1, :], dim=0)
                future_diff = torch.diff(self.data[self.num_past_steps:, :], dim=0)
                data = torch.cat((past_diff, torch.zeros_like(orig_center)[None, ...], future_diff), dim=0)
                
                # keep the center position for converting to accumulated mode
                self.data = data
                self.set_center(orig_center)
                
        elif self.get('mode') == 'difference':
            self.set_mode(target_mode)
            # convert to accumulated
            data = self.data
            data[self.num_past_steps, :] = self.get('center')
            
            if isinstance(self.data, np.ndarray):
                past_diff = -data[:self.num_past_steps, :][::-1, :]
                past_accum = np.cumsum(np.concatenate((self.center.reshape(1, -1), past_diff), axis=0), axis=0)[::-1, :]
                future_accum = np.cumsum(data[self.num_past_steps:, :], axis=0) 
                data = np.concatenate((past_accum[:self.num_past_steps], future_accum), axis=0)
                self.data = data
            elif isinstance(self.data, torch.Tensor):
                past_diff = -data[:self.num_past_steps, :][::-1, :]
                past_accum = torch.cumsum(torch.cat((self.center.reshape(1, -1), past_diff), dim=0), dim=0)[::-1, :]
                future_accum = torch.cumsum(data[self.num_past_steps:, :], dim=0) 
                data = torch.cat((past_accum[:self.num_past_steps], future_accum), dim=0)
                self.data = data

class Ego(BaseDataElement):
    """ Data structure for ego vehicle information
    
    Attributes:
        - pose (torch.Tensor): The transformation matrix from ego to world coordinates.
        - traj (TrajectoryData): The trajectory of the ego vehicle.
        - goal (torch.Tensor): The goal point of the ego vehicle.
        - context (torch.Tensor): The local context features of the ego vehicle.
    """
    
    def __setattr__(self, name: str, value: Sized):
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
            assert isinstance(value,
                              Sized), 'value must contain `__len__` attribute'

            super().__setattr__(name, value)

    __setitem__ = __setattr__
    
    @property
    def goal(self) -> torch.Tensor:
        """The goal point of the ego vehicle
        
        Returns:
            torch.Tensor: The goal point of the ego vehicle
        """
        if hasattr(self, '_goal'):
            return self._goal_point
        return None
    
    @goal.setter
    def goal(self, value: torch.Tensor):
        """Goal point of the ego vehicle
        
        Args:
            value (torch.Tensor): The goal point of the ego vehicle
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Goal point should be a tensor"
        
        assert value.ndim == 1, "Goal point should be a 1D tensor"        
        self.set_field(value, '_goal', dtype=type(value))
    
    @goal.deleter
    def goal(self):
        del self._goal
    
    @property
    def pose(self) -> torch.Tensor:
        """The transformation matrix from ego to world coordinates
        
        Returns:
            torch.Tensor: The transformation matrix from ego to world coordinates
        """
        if hasattr(self, '_pose'):
            return self._pose
        return None
    
    @pose.setter
    def pose(self, value: torch.Tensor):
        """The transformation matrix from ego to world coordinates
        
        Args:
            value (torch.Tensor): The transformation matrix from ego to world coordinates
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Transformation matrix should be a tensor"
        
        assert value.ndim == 2, "Transformation matrix should be a 2D tensor"
        assert value.shape[0] == 4 and value.shape[1] == 4, "Transformation matrix should be a 4x4 tensor"
        
        self.set_field(value, '_pose', dtype=type(value))
    
    @pose.deleter
    def pose(self):
        del self._pose
    
    # trajectories
    @property
    def traj(self) -> Trajectory:
        """The trajectory of the ego vehicle
        
        Returns:
            Trajectory: The trajectory of the ego vehicle
        """
        if hasattr(self, '_traj'):
            return self._traj
        return None
    @traj.setter
    def traj(self, value: Union[np.ndarray, torch.Tensor, Trajectory]):
        """The trajectory of the ego vehicle
        
        Args:
            value (Trajectory): The trajectory of the ego vehicle
        """
        assert isinstance(value, (np.ndarray, torch.Tensor)) \
            or isinstance(value, Trajectory), \
            "Trajectory should be a Trajectory object"
        
        self.set_field(value, '_traj', dtype=type(value))
    @traj.deleter
    def traj(self):
        del self._traj
    
    # local context
    @property
    def context(self) -> torch.Tensor:
        """The local context features of the ego vehicle
        
        Returns:
            torch.Tensor: The local context features of the ego vehicle
        """
        if hasattr(self, '_context'):
            return self._context
        return None
    @context.setter
    def context(self, value: torch.Tensor):
        """The local context features of the ego vehicle
        
        Args:
            value (torch.Tensor): The local context features of the ego vehicle
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Local context should be a tensor"
        
        self.set_field(value, '_context', dtype=type(value))
    @context.deleter
    def context(self):
        del self._context
    
    # commmand: turn left/right, go straight, etc
    @property
    def command(self) -> torch.Tensor:
        """The command of the ego vehicle
        
        Returns:
            torch.Tensor: The command of the ego vehicle
        """
        if hasattr(self, '_command'):
            return self._command
        return None
    @command.setter
    def command(self, value: torch.Tensor):
        """The command of the ego vehicle
        
        Args:
            value (torch.Tensor): The command of the ego vehicle
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Command should be a tensor"
        
        self.set_field(value, '_command', dtype=type(value))
    @command.deleter
    def command(self):
        del self._command
    
    
    # 
    def __len__(self) -> int:
        """int: The length of Ego."""
        if len(self._data_fields) > 0:
            return len(self.values()[0])
        else:
            return 0
class Instances(InstanceData):
    """ Data structure for instance annotations
    
    Attributes:
        - ids (torch.Tensor): The instance ids of the instances.
        - mask (torch.Tensor): mask of the instances to be ignored.
        - pose (torch.Tensor): The transformation matrix from the instances to world coordinates.
        
        - bboxes (torch.Tensor): The bounding boxes of the instances, typically in lidary coord.
        - labels (torch.Tensor): The class labels of the instances.
        - scores (torch.Tensor): The predicted scores of the instances.
        - traj (TrajectoryData): The trajectory of the instances, typically in lidar coord.
        - context (torch.Tensor): The local context features of the instances.  
    """

    @property
    def id(self) -> torch.Tensor:
        """The instance ids of the instances
        
        Returns:
            torch.Tensor: The instance ids of the instances
        """
        if hasattr(self, '_id'):
            return self._id
        return None
    
    @id.setter
    def id(self, value: torch.Tensor):
        """The instance ids of the instances
        
        Args:
            value (torch.Tensor): The instance ids of the instances
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Instance ids should be a tensor"
        
        assert value.ndim == 1, "Instance ids should be a 1D tensor"
        
        self.set_field(value, '_id', dtype=type(value))
    
    @id.deleter
    def id(self):
        del self._id
    
    @property
    def mask(self) -> torch.Tensor:
        """The mask of the instances
        
        Returns:
            torch.Tensor: The mask of the instances
        """
        if hasattr(self, '_bboxes_mask'):
            return self._bboxes_mask
        return None
    
    @mask.setter
    def mask(self, value: torch.Tensor):
        """The mask of the instances
        
        Args:
            value (torch.Tensor): The mask of the instances
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Mask should be a tensor"
        
        assert value.ndim == 1, "Mask should be a 1D tensor"
        
        self.set_field(value, '_bboxes_mask', dtype=type(value))
    
    @mask.deleter
    def mask(self):
        del self._bboxes_mask
    
    @property
    def pose(self) -> torch.Tensor:
        """The transformation matrix from the instances to world coordinates
        
        Returns:
            torch.Tensor: The transformation matrix from the instances to world coordinates
        """
        if hasattr(self, '_pose'):
            return self._pose
        return None

    @pose.setter
    def pose(self, value: torch.Tensor):
        """The transformation matrix from the instances to world coordinates
        
        Args:
            value (torch.Tensor): The transformation matrix from the instances to world coordinates
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Transformation matrix should be a tensor"
        
        assert value.shape[-2:] == (4, 4), "Transformation matrix for each isntance should be a 4x4 tensor"

        self.set_field(value, '_pose', dtype=type(value))
    
    @pose.deleter
    def pose(self):
        del self._pose
    
    @property
    def bbox(self) -> torch.Tensor:
        """The 3D bounding boxes of the instances
        
        Returns:
            torch.Tensor: The bounding boxes of the instances
        """
        if hasattr(self, '_bbox'):
            return self._bbox
        return None
    
    @bbox.setter
    def bbox(self, value: torch.Tensor):
        """The bounding boxes of the instances
        
        Args:
            value (torch.Tensor): The bounding boxes of the instances
        """
        assert isinstance(value, BaseInstance3DBoxes), \
            "Bounding boxes should be a BaseInstance3DBoxes object"
        
        self.set_field(value, '_bbox', dtype=type(value))
        # for backward compatibility
        self.set_field(value, 'bboxes_3d', dtype=type(value))
    
    @bbox.deleter
    def bbox(self):
        del self._bbox
        del self.bboxes_3d
    
    # gt labels
    @property
    def label(self) -> torch.Tensor:
        """The class labels of the instances
        
        Returns:
            torch.Tensor: The class labels of the instances
        """
        if hasattr(self, '_label'):
            return self._label
        return None

    @label.setter
    def label(self, value: torch.Tensor):
        """The class labels of the instances
        
        Args:
            value (torch.Tensor): The class labels of the instances
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Class labels should be a tensor"
        
        assert value.ndim == 1, "Class labels should be a 1D tensor"
        
        self.set_field(value, '_label', dtype=type(value))
        # for backward compatibility
        self.set_field(value, 'labels_3d', dtype=type(value))
        
    @label.deleter
    def label(self):
        del self._label
        del self.labels_3d
        
    # pred scores of the labels
    @property
    def score(self) -> torch.Tensor:
        """The scores of the instances
        
        Returns:
            torch.Tensor: The scores of the instances
        """
        if hasattr(self, '_score'):
            return self._score
        return None
    
    @score.setter
    def score(self, value: torch.Tensor):
        """The scores of the instances
        
        Args:
            value (torch.Tensor): The scores of the instances
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Scores should be a tensor"
                
        self.set_field(value, '_score', dtype=type(value))
        # for backward compatibility
        self.set_field(value, 'scores_3d', dtype=type(value))
    
    @score.deleter
    def score(self):
        del self._score 
        del self.scores_3d
    
    
    # trajectories
    @property
    def traj(self) -> Trajectory:
        """The trajectory of the instances
        
        Returns:
            TrajectoryData: The trajectory of the instances
        """
        if hasattr(self, '_traj'):
            return self._traj
        return None

    @traj.setter
    def traj(self, value: Union[Array, Trajectory]):
        """The trajectory of the instances
        
        Args:
            value (TrajectoryData): The trajectory of the instances
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)) \
            or (isinstance(value, list) and (len(value) == 0 or isinstance(value[0], Trajectory))), \
            "trajectory should be an array-like or TrajectoryData object or empty"
        
        self.set_field(value, '_traj', dtype=type(value))
    
    @traj.deleter
    def traj(self):
        del self._traj
    
    @property
    def traj_mask(self, value: Union[torch.Tensor, np.ndarray]):
        """The mask of the trajectory of the instances
        
        Returns:
            torch.Tensor: The mask of the trajectory of the instances
        """
        if hasattr(self, '_traj_mask'):
            return self._traj_mask
        return None
    @traj_mask.setter
    def traj_mask(self, value: Union[torch.Tensor, np.ndarray]):
        """The mask of the trajectory of the instances
        
        Args:
            value (torch.Tensor): The mask of the trajectory of the instances
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Mask should be a tensor"        
        self.set_field(value, '_traj_mask', dtype=type(value))
    @traj_mask.deleter
    def traj_mask(self):
        del self._traj_mask
    
    
    @property
    def context(self) -> torch.Tensor:
        """The local context features of the instances
        
        Returns:
            torch.Tensor: The local context features of the instances
        """
        if hasattr(self, '_context'):
            return self._context
        return None
    
    @context.setter
    def context(self, value: torch.Tensor):
        """The local context features of the instances
        
        Args:
            value (torch.Tensor): The local context features of the instances
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Context should be a tensor"
                
        self.set_field(value, '_context', dtype=type(value))
    
    @context.deleter
    def context(self):
        del self._context
 
    ## ====================================
    #TODO: use recursion to supported nested sequence of data
    # Mainly to support convert a list of trajectory data
    def to(self, *args, **kwargs) -> 'BaseDataElement':
        """Apply same name function to all tensors in data_fields.
        """
        new_data = self.new()
        for k, v in self.items():
            if hasattr(v, 'to'):
                v = v.to(*args, **kwargs)
                data = {k: v}
                new_data.set_data(data)
            elif is_seq_of(v, torch.Tensor) or is_seq_of(v, BaseDataElement):
                v = [x.to(*args, **kwargs) for x in v]
                data = {k: v}
                new_data.set_data(data)
                
        return new_data
    
    # Tensor-like methods
    def cpu(self) -> 'BaseDataElement':
        """Convert all tensors to CPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cpu()
                data = {k: v}
                new_data.set_data(data)
            elif is_seq_of(v, torch.Tensor) or is_seq_of(v, BaseDataElement):
                v = [x.cpu() for x in v]
                data = {k: v}
                new_data.set_data(data)
                
        return new_data

    # Tensor-like methods
    def cuda(self) -> 'BaseDataElement':
        """Convert all tensors to GPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cuda()
                data = {k: v}
                new_data.set_data(data)
            elif is_seq_of(v, torch.Tensor) or is_seq_of(v, BaseDataElement):
                v = [x.cuda() for x in v]
                data = {k: v}
                new_data.set_data(data)
                
        return new_data    
    
class Grids(BaseDataElement):
    """Data structure for grid-map like data annontation
    
    Attributes:
        - occupancy_mask (torch.Tensor): The mask of the occupancy grid map.
        - density_mask (torch.Tensor): The mask of the density grid map.
        - occupancy (torch.Tensor): The occupancy of the grid map.
        - density (torch.Tensor): The density of the grid map.
    """
    
    @property
    def occupancy_mask(self) -> torch.Tensor:
        """The mask of the occupancy grid map
        
        Returns:
            torch.Tensor: The mask of the occupancy grid map
        """
        if hasattr(self, '_occupancy_mask'):
            return self._occupancy_mask
        return None
    
    @occupancy_mask.setter
    def occupancy_mask(self, value: torch.Tensor):
        """The mask of the occupancy grid map
        
        Args:
            value (torch.Tensor): The mask of the occupancy grid map
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Mask should be a tensor"
                
        self.set_field(value, '_occupancy_mask', dtype=type(value))
    
    @occupancy_mask.deleter
    def occupancy_mask(self):
        del self._occupancy_mask
    
    @property
    def density_mask(self) -> torch.Tensor:
        """The mask of the density grid map
        
        Returns:
            torch.Tensor: The mask of the density grid map
        """
        if hasattr(self, '_density_mask'):
            return self._density_mask
        return None

    @density_mask.setter
    def density_mask(self, value: torch.Tensor):
        """The mask of the density grid map
        
        Args:
            value (torch.Tensor): The mask of the density grid map
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Mask should be a tensor"
                
        self.set_field(value, '_density_mask', dtype=type(value))
    
    @density_mask.deleter
    def density_mask(self):
        del self._density_mask
    
    @property
    def occupancy(self) -> torch.Tensor:
        """The occupancy of the grid map
        
        Returns:
            torch.Tensor: The occupancy of the grid map
        """
        if hasattr(self, '_occupancy'):
            return self._occupancy
        return None

    @occupancy.setter
    def occupancy(self, value: torch.Tensor):
        """The occupancy of the grid map
        
        Args:
            value (torch.Tensor): The occupancy of the grid map
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Occupancy should be a tensor"
                
        self.set_field(value, '_occupancy', dtype=type(value))
    
    @occupancy.deleter
    def occupancy(self):
        del self._occupancy

    @property
    def density(self) -> torch.Tensor:
        """The density of the grid map
        
        Returns:
            torch.Tensor: The density of the grid map
        """
        if hasattr(self, '_density'):
            return self._density
        return None
    
    @density.setter
    def density(self, value: torch.Tensor):
        """The density of the grid map
        
        Args:
            value (torch.Tensor): The density of the grid map
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Density should be a tensor"
                
        self.set_field(value, '_density', dtype=type(value))
    
    @density.deleter
    def density(self):
        del self._density

class BaseMap(BaseDataElement):
    """Base class for map-like data structure
    
    Attributes:
        - map (torch.Tensor): The map of the data
        - mask (torch.Tensor): The mask of the data
    """
    
    def __setattr__(self, name: str, value: Sized):
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
            #assert isinstance(value,
            #                  Sized), 'value must contain `__len__` attribute'

            super().__setattr__(name, value)

    # this enables the use of __setitem__ to set attributes
    __setitem__ = __setattr__

class VectorMap(BaseMap):
    """VectorMap data structure
    """
    # do nothing for now
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
class DenseMap(BaseMap):
    pass