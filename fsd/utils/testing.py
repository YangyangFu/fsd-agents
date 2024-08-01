import torch
import numpy as np 
import random, os
import copy 

from mmengine.config import Config
from fsd.utils import ConfigType

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def _get_root_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source fsdagent repo
        repo_dpath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    except NameError:
        # For IPython development when this __file__ is not defined
        import fsd
        repo_dpath = os.path.dirname(os.path.dirname(fsd.__file__))

    if not os.path.exists(repo_dpath):
        raise Exception('Cannot find repository root path')
    return repo_dpath

def _get_config_module(fname):
    """Load a configuration as a python module."""
    root_dpath = _get_root_directory()
    config_fpath = os.path.join(root_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod

def get_agent_cfg(path: str)-> ConfigType:
    config = _get_config_module(path)
    
    model = copy.deepcopy(config.model)
    return model