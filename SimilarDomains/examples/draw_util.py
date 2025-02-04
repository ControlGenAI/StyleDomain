import torch
import random
import numpy as np

from pathlib import Path
from copy import deepcopy
from collections import OrderedDict


class IdentityEditor:    
    def __call__(self, style_space, **kwargs):
        return style_space

    def __add__(self, style_editor, **kwargs):
        new_editor = deepcopy(style_editor)
        return new_editor


class StyleEditor:
    offsetconv_to_style = dict([
        (0, 0),
        (1, 2),
        (2, 3),
        (3, 5),
        (4, 6),
        (5, 8),
        (6, 9),
        (7, 11),
        (8, 12),
        (9, 14),
        (10, 15),
        (11, 17),
        (12, 18),
        (13, 20),
        (14, 21),
        (15, 23),
        (16, 24)
    ])
    
    def __init__(self, ckpt=None, device='cuda:0', img_size=1024):
        self.device = device
        if ckpt is None:
            return
        self._construct_from_ckpt(ckpt)
    
    def _construct_from_ckpt(self, ckpt):
        last_layer_index = max([int(k.split('.')[1].split('_')[-1]) for k in ckpt['state_dict'].keys()])
        
        self.shifts = {
              StyleEditor.offsetconv_to_style[off_idx]: ckpt['state_dict'][f'heads.conv_{off_idx}.params_in'].to(self.device) 
                    for off_idx in range(last_layer_index)
        }
    
    def __call__(self, stspace, power=1.):
        answer = {}
        for idx, value in enumerate(stspace):
            if idx in self.shifts:
                answer[idx] = stspace[idx].clone() + power * self.shifts[idx]
            else:
                answer[idx] = stspace[idx].clone()            
        
        return list(answer.values())
    
    def __mul__(self, alpha):
        answer = deepcopy(self)
        for st_idx in answer.shifts:
            answer.shifts[st_idx] = self.shifts[st_idx] * alpha
        return answer
    
    def __add__(self, other):
        answer = deepcopy(self)
        for st_idx in other.shifts:
            answer.shifts[st_idx] = self.shifts[st_idx] + other.shifts[st_idx]
        return answer
    
    def to(self, device):
        for st_idx in self.shifts:
            self.shifts[st_idx] = self.shifts[st_idx].to(device)
        self.device = device
        return self


def set_seed(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    

def morph_g_ema(ckpt1, ckpt2, alpha):
    final_ckpt = OrderedDict()
    
    for key in ckpt1['g_ema']:
        final_ckpt[key] = alpha * ckpt1['g_ema'][key] + (1 - alpha) * ckpt2['g_ema'][key]
        
    return {'g_ema': final_ckpt}
    

w_style_pair = [
    (0, 0),
    (1, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (3, 5),
    (4, 6),
    (5, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (7, 11),
    (8, 12),
    (9, 13),
    (9, 14),
    (10, 15),
    (11, 16),
    (11, 17),
    (12, 18),
    (13, 19),
    (13, 20),
    (14, 21),
    (15, 22),
    (15, 23),
    (16, 24),
    (17, 25)
]


p_root = Path(__file__).resolve().parent.parent / 'pretrained'

weights = {p.name.rsplit('_', 1)[0]: p for p in (p_root / 'checkpoints_iccv').iterdir()}

weights.update({p.name.split('-')[1]: p for p in (p_root / 'StyleGAN2').iterdir()})
weights.update({'_'.join(p.stem.split('_')[1:3]): p for p in (p_root / 'StyleGAN2_ADA').iterdir()})
