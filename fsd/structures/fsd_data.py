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

class Ego(BaseDataElement):
    """ Data structure for ego vehicle information
    
    Attributes:
        - goal_point (torch.Tensor): The goal point of the ego vehicle.
        - ego2world (torch.Tensor): The transformation matrix from ego to world coordinates.
        - gt_traj (TrajectoryData): The ground truth trajectory of the ego vehicle.
        - pred_traj (TrajectoryData): The predicted trajectory of the ego vehicle.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def goal_point(self) -> torch.Tensor:
        """The goal point of the ego vehicle
        
        Returns:
            torch.Tensor: The goal point of the ego vehicle
        """
        if hasattr(self, '_goal_point'):
            return self._goal_point
        return None
    
    @goal_point.setter
    def goal_point(self, value: torch.Tensor):
        """Goal point of the ego vehicle
        
        Args:
            value (torch.Tensor): The goal point of the ego vehicle
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Goal point should be a tensor"
        
        assert value.ndim == 1, "Goal point should be a 1D tensor"        
        self.set_field(value, '_goal_point', dtype=type(value))
    
    @goal_point.deleter
    def goal_point(self):
        del self._goal_point
    
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
    def gt_traj(self) -> TrajectoryData:
        """The ground truth trajectory of the ego vehicle
        
        Returns:
            TrajectoryData: The ground truth trajectory of the ego vehicle
        """
        if hasattr(self, '_gt_traj'):
            return self._gt_traj
        return None
    
    @gt_traj.setter
    def gt_traj(self, value: Union[TrajectoryData, MultiModalTrajectoryData]):
        """The ground truth trajectory of the ego vehicle
        
        Args:
            value (TrajectoryData): The ground truth trajectory of the ego vehicle
        """
        assert isinstance(value, TrajectoryData), \
            "Ground truth trajectory should be a TrajectoryData object"
        
        self.set_field(value, '_gt_traj', dtype=type(value))
    
    @gt_traj.deleter
    def gt_traj(self):
        del self._gt_traj
    
    @property
    def pred_traj(self) -> TrajectoryData:
        """The predicted trajectory of the ego vehicle
        
        Returns:
            TrajectoryData: The predicted trajectory of the ego vehicle
        """
        if hasattr(self, '_pred_traj'):
            return self._pred_traj
        return None
    
    @pred_traj.setter
    def pred_traj(self, value: Union[TrajectoryData, MultiModalTrajectoryData]):
        """The predicted trajectory of the ego vehicle
        
        Args:
            value (TrajectoryData): The predicted trajectory of the ego vehicle
        """
        assert isinstance(value, TrajectoryData), \
            "Predicted trajectory should be a TrajectoryData object"
        
        self.set_field(value, '_pred_traj', dtype=type(value))
        
    @pred_traj.deleter
    def pred_traj(self):
        del self._pred_traj
        
class Instances(InstanceData):
    """ Data structure for instance annotations
    
    Attributes:
        - ids (torch.Tensor): The instance ids of the instances.
        - bboxes_mask (torch.Tensor): mask of the instances to be ignored.
        - pose (torch.Tensor): The transformation matrix from the instances to world coordinates.
        
        - gt_bboxes_3d (torch.Tensor): The bounding boxes of the instances, typically in lidary coord
        - pred_bboxes_3d (torch.Tensor): The predicted bounding boxes of the instances, typically in lidar coord
        - gt_labels (torch.Tensor): The class labels of the instances.
        - pred_labels (torch.Tensor): The predicted class labels of the instances.
        - pred_scores (torch.Tensor): The predicted scores of the instances.
        - gt_traj (TrajectoryData): The past trajectory of the instances, typically in lidar coord
        - pred_traj (TrajectoryData): The future trajectory of the instances, typicall in lidar coord
    """

    @property
    def ids(self) -> torch.Tensor:
        """The instance ids of the instances
        
        Returns:
            torch.Tensor: The instance ids of the instances
        """
        if hasattr(self, '_ids'):
            return self._ids
        return None
    
    @ids.setter
    def ids(self, value: torch.Tensor):
        """The instance ids of the instances
        
        Args:
            value (torch.Tensor): The instance ids of the instances
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Instance ids should be a tensor"
        
        assert value.ndim == 1, "Instance ids should be a 1D tensor"
        
        self.set_field(value, '_ids', dtype=type(value))
    
    @ids.deleter
    def ids(self):
        del self._ids
    
    @property
    def bboxes_mask(self) -> torch.Tensor:
        """The mask of the instances
        
        Returns:
            torch.Tensor: The mask of the instances
        """
        if hasattr(self, '_bboxes_mask'):
            return self._bboxes_mask
        return None
    
    @bboxes_mask.setter
    def bboxes_mask(self, value: torch.Tensor):
        """The mask of the instances
        
        Args:
            value (torch.Tensor): The mask of the instances
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Mask should be a tensor"
        
        assert value.ndim == 1, "Mask should be a 1D tensor"
        
        self.set_field(value, '_bboxes_mask', dtype=type(value))
    
    @bboxes_mask.deleter
    def bboxes_mask(self):
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
        
        assert value.ndim == 2, "Transformation matrix should be a 2D tensor"
        assert value.shape[0] == 4 and value.shape[1] == 4, "Transformation matrix should be a 4x4 tensor"
        
        self.set_field(value, '_pose', dtype=type(value))
    
    @pose.deleter
    def pose(self):
        del self._pose
    
    @property
    def gt_bboxes_3d(self) -> torch.Tensor:
        """The bounding boxes of the instances
        
        Returns:
            torch.Tensor: The bounding boxes of the instances
        """
        if hasattr(self, '_gt_bboxes_3d'):
            return self._gt_bboxes_3d
        return None
    
    @gt_bboxes_3d.setter
    def gt_bboxes_3d(self, value: torch.Tensor):
        """The bounding boxes of the instances
        
        Args:
            value (torch.Tensor): The bounding boxes of the instances
        """
        assert isinstance(value, BaseInstance3DBoxes), \
            "Bounding boxes should be a tensor"
        
        self.set_field(value, '_gt_bboxes_3d', dtype=type(value))
    
    @gt_bboxes_3d.deleter
    def gt_bboxes_3d(self):
        del self._gt_bboxes_3d
    
    
    @property
    def pred_bboxes_3d(self) -> torch.Tensor:
        """The predicted bounding boxes of the instances
        
        Returns:
            torch.Tensor: The predicted bounding boxes of the instances
        """
        if hasattr(self, '_pred_bboxes_3d'):
            return self._pred_bboxes_3d
        return None
    
    @pred_bboxes_3d.setter
    def pred_bboxes_3d(self, value: torch.Tensor):
        """The predicted bounding boxes of the instances
        
        Args:
            value (torch.Tensor): The predicted bounding boxes of the instances
        """
        assert isinstance(value, BaseInstance3DBoxes), \
            "Bounding boxes should be a tensor"
        
        assert value.tensor.shape == self.gt_bboxes_3d.tensor.shape, \
            "The shape of the predicted bounding boxes is not consistent with the ground truth"
        
        self.set_field(value, '_pred_bboxes_3d', dtype=type(value))
    
    @pred_bboxes_3d.deleter
    def pred_bboxes_3d(self):
        del self._pred_bboxes_3d
    
    # gt labels
    @property
    def gt_labels(self) -> torch.Tensor:
        """The class labels of the instances
        
        Returns:
            torch.Tensor: The class labels of the instances
        """
        if hasattr(self, '_gt_labels'):
            return self._gt_labels
        return None

    @gt_labels.setter
    def gt_labels(self, value: torch.Tensor):
        """The class labels of the instances
        
        Args:
            value (torch.Tensor): The class labels of the instances
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Class labels should be a tensor"
        
        assert value.ndim == 1, "Class labels should be a 1D tensor"
        
        self.set_field(value, '_gt_labels', dtype=type(value))
    
    @gt_labels.deleter
    def gt_labels(self):
        del self._gt_labels
    
    # pred labels
    @property
    def pred_labels(self) -> torch.Tensor:
        """The predicted class labels of the instances
        
        Returns:
            torch.Tensor: The predicted class labels of the instances
        """
        if hasattr(self, '_pred_labels'):
            return self._pred_labels
        return None

    @pred_labels.setter
    def pred_labels(self, value: torch.Tensor):
        """The predicted class labels of the instances
        
        Args:
            value (torch.Tensor): The predicted class labels of the instances
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Class labels should be a tensor"
        
        assert value.shape == self.gt_labels.shape, \
            "The shape of the predicted class labels is not consistent with the ground truth"
        
        self.set_field(value, '_pred_labels', dtype=type(value))
    
    @pred_labels.deleter
    def pred_labels(self):
        del self._pred_labels
    
    # pred scores of the labels
    @property
    def pred_scores(self) -> torch.Tensor:
        """The predicted scores of the instances
        
        Returns:
            torch.Tensor: The predicted scores of the instances
        """
        if hasattr(self, '_pred_scores'):
            return self._pred_scores
        return None
    
    @pred_scores.setter
    def pred_scores(self, value: torch.Tensor):
        """The predicted scores of the instances
        
        Args:
            value (torch.Tensor): The predicted scores of the instances
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Scores should be a tensor"
                
        self.set_field(value, '_pred_scores', dtype=type(value))
    
    @pred_scores.deleter
    def pred_scores(self):
        del self._pred_scores
    
    
    # trajectories
    @property
    def gt_traj(self) -> TrajectoryData:
        """The ground truth trajectory of the instances
        
        Returns:
            TrajectoryData: The ground truth trajectory of the instances
        """
        if hasattr(self, '_gt_traj'):
            return self._gt_traj
        return None

    @gt_traj.setter
    def gt_traj(self, value: Union[TrajectoryData, MultiModalTrajectoryData]):
        """The ground truth trajectory of the instances
        
        Args:
            value (TrajectoryData): The ground truth trajectory of the instances
        """
        assert isinstance(value, list) and isinstance(value[0], TrajectoryData), \
            "Ground truth trajectory should be a TrajectoryData object"
        
        self.set_field(value, '_gt_traj', dtype=type(value))
    
    @gt_traj.deleter
    def gt_traj(self):
        del self._gt_traj
    
    @property
    def pred_traj(self) -> TrajectoryData:
        """The predicted trajectory of the instances
        
        Returns:
            TrajectoryData: The predicted trajectory of the instances
        """
        if hasattr(self, '_pred_traj'):
            return self._pred_traj
        return None
    
    @pred_traj.setter
    def pred_traj(self, value: Union[TrajectoryData, MultiModalTrajectoryData]):
        """The predicted trajectory of the instances
        
        Args:
            value (TrajectoryData): The predicted trajectory of the instances
        """
        assert isinstance(value, list) and isinstance(value[0], TrajectoryData), \
            "Predicted trajectory should be a list of TrajectoryData object"
        
        self.set_field(value, '_pred_traj', dtype=type(value))
    
    @pred_traj.deleter
    def pred_traj(self):
        del self._pred_traj
    
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
        - gt_occupancy (torch.Tensor): The ground truth occupancy of the grid map.
        - pred_occupancy (torch.Tensor): The predicted occupancy of the grid map.
        - gt_density (torch.Tensor): The ground truth density of the grid map.
        - pred_density (torch.Tensor): The predicted density of the grid map.
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
    def gt_occupancy(self) -> torch.Tensor:
        """The ground truth occupancy of the grid map
        
        Returns:
            torch.Tensor: The ground truth occupancy of the grid map
        """
        if hasattr(self, '_gt_occupancy'):
            return self._gt_occupancy
        return None

    @gt_occupancy.setter
    def gt_occupancy(self, value: torch.Tensor):
        """The ground truth occupancy of the grid map
        
        Args:
            value (torch.Tensor): The ground truth occupancy of the grid map
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Occupancy should be a tensor"
                
        self.set_field(value, '_gt_occupancy', dtype=type(value))
    
    @gt_occupancy.deleter
    def gt_occupancy(self):
        del self._gt_occupancy
        
    @property
    def pred_occupancy(self) -> torch.Tensor:
        """The predicted occupancy of the grid map
        
        Returns:
            torch.Tensor: The predicted occupancy of the grid map
        """
        if hasattr(self, '_pred_occupancy'):
            return self._pred_occupancy
        return None
    
    @pred_occupancy.setter
    def pred_occupancy(self, value: torch.Tensor):
        """The predicted occupancy of the grid map
        
        Args:
            value (torch.Tensor): The predicted occupancy of the grid map
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Occupancy should be a tensor"
        
        assert value.shape == self.gt_occupancy.shape, \
            "The shape of the predicted occupancy is not consistent with the ground truth occupancy"
            
        self.set_field(value, '_pred_occupancy', dtype=type(value))
    
    @pred_occupancy.deleter
    def pred_occupancy(self):
        del self._pred_occupancy
    
    @property
    def gt_density(self) -> torch.Tensor:
        """The ground truth density of the grid map
        
        Returns:
            torch.Tensor: The ground truth density of the grid map
        """
        if hasattr(self, '_gt_density'):
            return self._gt_density
        return None
    
    @gt_density.setter
    def gt_density(self, value: torch.Tensor):
        """The ground truth density of the grid map
        
        Args:
            value (torch.Tensor): The ground truth density of the grid map
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Density should be a tensor"
                
        self.set_field(value, '_gt_density', dtype=type(value))
    
    @gt_density.deleter
    def gt_density(self):
        del self._gt_density
    
    @property
    def pred_density(self) -> torch.Tensor:
        """The predicted density of the grid map
        
        Returns:
            torch.Tensor: The predicted density of the grid map
        """
        if hasattr(self, '_pred_density'):
            return self._pred_density
        return None

    @pred_density.setter
    def pred_density(self, value: torch.Tensor):
        """The predicted density of the grid map
        
        Args:
            value (torch.Tensor): The predicted density of the grid map
        """
        assert isinstance(value, (torch.Tensor, np.ndarray)), \
            "Density should be a tensor"
        
        assert value.shape == self.gt_density.shape, \
            "The shape of the predicted density is not consistent with the ground truth density"
            
        self.set_field(value, '_pred_density', dtype=type(value))
    
    @pred_density.deleter
    def pred_density(self):
        del self._pred_density