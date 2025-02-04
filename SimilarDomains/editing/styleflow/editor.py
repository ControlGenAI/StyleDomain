import torch
import pickle
import numpy as np

from pathlib import Path
from .flow import build_styleflow_model


class StyleFlowEditor:
    keep_indexes = [
        2, 5, 25, 28, 16, 32, 33, 34, 55, 75, 79, 162, 177, 196, 
        160, 212, 246, 285, 300, 329, 362, 369, 462, 460, 478, 
        551, 583, 643, 879, 852, 914, 999, 976, 627, 844, 237, 52, 301, 599
    ]
    attr_order = ['Gender', 'Glasses', 'Yaw', 'Pitch', 'Baldness', 'Beard', 'Age', 'Expression']
    lighting_order = ['Left->Right', 'Right->Left', 'Down->Up', 'Up->Down', 'No light', 'Front light']
    attr_degree_list = [1.5, 2.5, 1., 1., 2, 1.7,0.93, 1.]
    min_dic = {
        'Gender': 0, 'Glasses': 0, 'Yaw': -20, 'Pitch': -20, 
        'Baldness': 0, 'Beard': 0.0, 'Age': 0, 'Expression': 0
    }
    max_dic = {
        'Gender': 1, 'Glasses': 1, 'Yaw': 20, 'Pitch': 20, 
        'Baldness': 1, 'Beard': 1, 'Age': 65, 'Expression': 1
    }
    
    def __init__(
        self, data_path, weight_path, device
    ):
        
        self.model = build_styleflow_model(weight_path, device)
        self.device = device
        
        raw_w = pickle.load(open(Path(data_path) / 'sg2latents.pickle', 'rb'))
        raw_attr = np.load(Path(data_path) / 'attributes.npy')
        raw_lights = np.load(Path(data_path) / 'light.npy')
        
        self.all_w = np.array(raw_w['Latent'])[self.keep_indexes]
        self.all_attr = raw_attr[self.keep_indexes]
        self.all_lights = raw_lights[self.keep_indexes]
        
        self.zero_padding = torch.zeros(1, 18, 1).to(self.device)
        self.gap_dic = {i: StyleFlowEditor.max_dic[i] - StyleFlowEditor.min_dic[i] for i in StyleFlowEditor.max_dic}
        
    def _allocate_entity(self, idx):
        self.w_current_np = self.all_w[idx].copy()
        self.attr_current = self.all_attr[idx].copy()
        self.light_current = self.all_lights[idx].copy()

        self.attr_current_list = [self.attr_current[i][0] for i in range(len(self.attr_order))]
        self.light_current_list = [0 for i in range(len(self.lighting_order))]
        
        array_source = torch.from_numpy(self.attr_current).type(torch.FloatTensor).to(self.device)
        array_light = torch.from_numpy(self.light_current).type(torch.FloatTensor).to(self.device)
        self.final_array_target_ = torch.cat([array_light, array_source.unsqueeze(0).unsqueeze(-1)], dim=1)
        self.initial_w = torch.from_numpy(self.w_current_np).to(self.device)
        
    def _invert_to_real(self, name, edit_power):
        return float(float(edit_power) * self.gap_dic[name] + StyleFlowEditor.min_dic[name])
    
    def get_edited_pair(self, attr_idx, edit_power):
        fws = self.model(self.initial_w, self.final_array_target_, self.zero_padding)
        
        real_value = self._invert_to_real(StyleFlowEditor.attr_order[attr_idx], edit_power)
        attr_change = real_value - self.attr_current_list[attr_idx]
        attr_final = StyleFlowEditor.attr_degree_list[attr_idx] * attr_change + self.attr_current_list[attr_idx]

        final_array_target = self.final_array_target_.clone()
        final_array_target[0, attr_idx + 9, 0, 0] = attr_final

        rev = self.model(fws[0], final_array_target, self.zero_padding, True)
        
        if attr_idx == 0:
            rev[0][0][8:] = self.initial_w[0][8:]
        elif attr_idx == 1:
            rev[0][0][:2] = self.initial_w[0][:2]
            rev[0][0][4:] = self.initial_w[0][4:]
        elif attr_idx == 2:
            rev[0][0][4:] = self.initial_w[0][4:]
        elif attr_idx == 3:
            rev[0][0][4:] = self.initial_w[0][4:]
        elif attr_idx == 4:
            rev[0][0][6:] = self.initial_w[0][6:]
        elif attr_idx == 5:
            rev[0][0][:5] = self.initial_w[0][:5]
            rev[0][0][10:] = self.initial_w[0][10:]
        elif attr_idx == 6:
            rev[0][0][0:4] = self.initial_w[0][0:4]
            rev[0][0][8:] = self.initial_w[0][8:]
        elif attr_idx == 7:
            rev[0][0][:4] = self.initial_w[0][:4]
            rev[0][0][6:] = self.initial_w[0][6:]
        
        return self.initial_w, rev[0].clone()