from typing import List, Union

import torch 
import numpy as np

def one_hot_encoding(labels: Union[np.array, torch.Tensor], 
                     num_classes: int) -> Union[np.array, torch.Tensor]:
    """
    Convert a tensor of labels to one-hot encoding.

    Args:
        labels (torch.Tensor): A tensor of labels.
        num_classes (int): The number of classes.

    Returns:
        torch.Tensor: A tensor of one-hot encoding.
    """
    if isinstance(labels, np.ndarray):
        return np.eye(num_classes)[labels]
    elif isinstance(labels, torch.Tensor):
        return torch.eye(num_classes)[labels]
    else:
        raise ValueError(f"Unsupported data type {type(labels)} for one-hot encoding")
