from typing import List, Optional, Sequence, Tuple, Union

import torch
import numpy as np 

from mmengine.config import ConfigDict
from mmengine.structures import InstanceData, PixelData

Array = Union[torch.Tensor, np.ndarray]

# TODO: Need to avoid circular import with assigner and sampler
# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]
# Type hint of one or more config data
MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

InstanceList = List[InstanceData]
OptInstanceList = Optional[InstanceList]

PixelList = List[PixelData]
OptPixelList = Optional[PixelList]

RangeType = Sequence[Tuple[int, int]]


# type hinting for data sample
from fsd.structures import PlanningDataSample

DataSampleType = Union[PlanningDataSample, dict]
OptDataSampleType = Optional[DataSampleType]
DataSampleList = List[DataSampleType]
OptDataSampleList = Optional[DataSampleList]