import torch

from .example_utils import Inferencer


def read_weights(exp_weights_root, idx=None):
    if idx is not None:
        model_path = exp_weights_root / f'models/models_{idx}.pt'
        return torch.load(model_path)
    models_path = list((exp_weights_root / 'models').iterdir())    
    model_path = sorted(models_path, key=lambda x: int(x.stem.split('_')[1]))[-1]
    return torch.load(model_path)


def get_model(exp_weights_root, idx=None):    
    ckpt = read_weights(exp_weights_root, idx)
    return Inferencer(ckpt, device)