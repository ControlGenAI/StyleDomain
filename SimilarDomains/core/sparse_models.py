import torch
import torch.nn as nn

from torch.autograd import Function
from collections import OrderedDict
from core.utils.common import get_stylegan_conv_dimensions


def cat_stylespace(style_space):
    return torch.cat(style_space, dim=1)


def split_stylespace(style_space, img_size=1024):
    prev = 0
    result = []
    for cin, _ in get_stylegan_conv_dimensions(img_size):
        result.append(style_space[:, prev:prev + cin])
        prev = prev + cin
    
    return result


def to_dict_input(deltas, img_size=1024):    
    dict_input = OrderedDict()
    start_idx = 0

    for idx, (in_d, out_d) in enumerate(get_stylegan_conv_dimensions(img_size)):
        dict_input[f'conv_{idx}'] = {
            'in': deltas[:, start_idx: start_idx + in_d]
        }
        start_idx += in_d
        
    return dict_input


def to_tensor(dict_input):
    return torch.cat([v['in'] for k, v in dict_input.items()], dim=1)
    
    
def ckpt_to_tensor(ckpt):
    state_dict = ckpt['state_dict']
    n = len(state_dict)
    return torch.cat([state_dict[f'heads.conv_{i}.params_in'] for i in range(n)], dim=1)
    

class SparsedModel(nn.Module):
    def __init__(self, device, ckpt=None):
        super().__init__()
        
        self.device = device
        self.convid_to_st = dict([
            (0, 0), (1, 2), (2, 3), (3, 5), 
            (4, 6), (5, 8), (6, 9), (7, 11), 
            (8, 12), (9, 14), (10, 15), (11, 17), 
            (12, 18), (13, 20), (14, 21), (15, 23), 
            (16, 24)
        ])

        self.s_to_conv_id = {v:k for k, v in self.convid_to_st.items()}
        self.input_keys = sorted(self.s_to_conv_id.keys())
        
        self.deltas = nn.Parameter(torch.zeros(1, 6048))
        self.register_buffer('grad_mask', torch.ones(1, 6048))
        
        if ckpt is not None:
            self._deltas_from_ckpt(ckpt)
        
    def forward(self, style_space):
        st = torch.cat([style_space[i] for i in self.input_keys], dim=1)
        st_shifted = st + self.deltas * self.grad_mask

        splited_st = split_stylespace(st_shifted)
        answer = [
            splited_st[self.s_to_conv_id[i]] if i in self.input_keys else style_space[i] for i in range(len(style_space))
        ]
        
        return answer, to_dict_input(self.deltas)
    
    def offsets(self):
        return to_dict_input(self.deltas)
    
    def pruned_offsets(self, perc):
        deltas_pruned = torch.clone(self.deltas.data)
        top = torch.abs(deltas_pruned.squeeze()).argsort() # top to lower
        chosen_idxes = int(6048 * perc)
        deltas_pruned[:, top[:chosen_idxes]] = 0.
        return to_dict_input(deltas_pruned)
    
    def _deltas_from_ckpt(self, ckpt):
        self.deltas = nn.Parameter(ckpt_to_tensor(ckpt).to(self.device))
        return self
    
    def pruned_forward(self, style_space, perc):
        deltas_pruned = torch.clone(self.deltas.data)
        top = torch.abs(deltas_pruned.squeeze()).argsort().flipud()
        chosen_idxes = int(6048 * perc)
        deltas_pruned = deltas_pruned[:, top[-chosen_idxes:]] = 0.
        st = cat_stylespace(style_space)
        st_shifted = self.fn(st, deltas_pruned, torch.ones(1, 6048))
        return split_stylespace(st_shifted)
