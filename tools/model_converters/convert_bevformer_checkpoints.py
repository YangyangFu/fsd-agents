import argparse
import tempfile
import torch
from mmengine import Config
from mmengine.runner import load_state_dict
from mmengine.registry import init_default_scope
from fsd.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert keys in original checkpoints for BEVFormer')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='path of the output checkpoint file')
    args = parser.parse_args()
    return args


def validate_model_converted(model, checkpoint):
    """Validate the converted model with the checkpoint.

    Args:
        model (nn.Module): The model to be validated.
        checkpoint (dict): The checkpoint to be validated.

    Returns:
        bool: True if the model is valid, False otherwise.
    """
    try:
        load_state_dict(model, checkpoint)
        return True
    except Exception as e:
        print(f'Error loading state dict: {e}')
        return False
    
def main():
    """Convert keys in checkpoints for VoteNet.

    There can be some breaking changes during the development of mmdetection3d,
    and this tool is used for upgrading checkpoints trained with old versions
    (before v0.6.0) to the latest one.
    """
    args = parse_args()
    checkpoint = torch.load(args.checkpoint)

    # ckpts
    orig_ckpt = checkpoint['state_dict']
    converted_ckpt = orig_ckpt.copy()


    RENAME_PREFIX = {
        'pts_bbox_head.transformer': 'pts_bbox_head',
    }

    DEL_KEYS = []

    EXTRACT_KEYS = {}

    # Delete some useless keys
    for key in DEL_KEYS:
        converted_ckpt.pop(key)

    # Rename keys with specific prefix
    RENAME_KEYS = dict()
    for old_key in converted_ckpt.keys():
        for rename_prefix in RENAME_PREFIX.keys():
            if rename_prefix in old_key:
                new_key = old_key.replace(rename_prefix,
                                          RENAME_PREFIX[rename_prefix])
                RENAME_KEYS[new_key] = old_key
    for new_key, old_key in RENAME_KEYS.items():
        converted_ckpt[new_key] = converted_ckpt.pop(old_key)

    # Extract weights and rename the keys
    for new_key, (old_key, indices) in EXTRACT_KEYS.items():
        cur_layers = orig_ckpt[old_key]
        converted_layers = []
        for (start, end) in indices:
            if end != -1:
                converted_layers.append(cur_layers[start:end])
            else:
                converted_layers.append(cur_layers[start:])
        converted_layers = torch.cat(converted_layers, 0)
        converted_ckpt[new_key] = converted_layers
        if old_key in converted_ckpt.keys():
            converted_ckpt.pop(old_key)

    # Check the converted checkpoint by loading to the model
    checkpoint['state_dict'] = converted_ckpt
    torch.save(checkpoint, args.out)


if __name__ == '__main__':
    main()
