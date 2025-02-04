import torch
import pickle
import numpy as np

from pathlib import Path

from .flow import cnf


def _build_styleflow_model(ckpt_path, device, input_dim=512, dims='512-512-512-512-512', zdim=17, num_blocks=1):
    model = cnf(input_dim=input_dim, dims=dims, zdim=zdim, num_blocks=num_blocks)

    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.to(device).eval()

    return model


class StyleFlowEditor:
    keep_indexes = [
        2, 5, 25, 28, 16, 32, 33, 34, 55, 75, 79, 162, 177, 196, 160, 212, 246, 285, 300, 329, 362,
        369, 462, 460, 478, 551, 583, 643, 879, 852, 914, 999, 976, 627, 844, 237, 52, 301, 599
    ]
    attr_order = ['Gender', 'Glasses', 'Yaw', 'Pitch', 'Baldness', 'Beard', 'Age', 'Expression']
    lighting_order = ['Left->Right', 'Right->Left', 'Down->Up', 'Up->Down', 'No light', 'Front light']

    attr_degree_list = [1.5, 2.5, 1., 1., 2, 1.7, 0.93, 1.]

    min_dic = {
        'Gender': 0, 'Glasses': 0, 'Yaw': -20, 'Pitch': -20,
        'Baldness': 0, 'Beard': 0.0, 'Age': 0, 'Expression': 0
    }
    max_dic = {
        'Gender': 1, 'Glasses': 1, 'Yaw': 20, 'Pitch': 20,
        'Baldness': 1, 'Beard': 1, 'Age': 65, 'Expression': 1
    }

    light_degree = 1.
    light_min_dic = {
        'Left->Right': 0, 'Right->Left': 0, 'Down->Up': 0,
        'Up->Down': 0, 'No light': 0, 'Front light': 0
    }
    light_max_dic = {
        'Left->Right': light_degree, 'Right->Left': light_degree, 'Down->Up': light_degree,
        'Up->Down': light_degree, 'No light': light_degree, 'Front light': light_degree
    }

    def __init__(self, data_path, weight_path, device):
        self.device = device
        self.model = _build_styleflow_model(weight_path, device)

        raw_w = pickle.load(open(Path(data_path) / 'sg2latents.pickle', 'rb'))
        self.raw_attr = np.load(Path(data_path) / 'attributes.npy')
        self.raw_lights = np.load(Path(data_path) / 'light.npy')

        self.all_w = np.array(raw_w['Latent'])[self.keep_indexes]
        self.all_attr = self.raw_attr[self.keep_indexes]
        self.all_lights = self.raw_lights[self.keep_indexes]

        light0 = torch.from_numpy(self.raw_lights[8]).type(torch.FloatTensor).cuda()
        light1 = torch.from_numpy(self.raw_lights[33]).type(torch.FloatTensor).cuda()
        light2 = torch.from_numpy(self.raw_lights[641]).type(torch.FloatTensor).cuda()
        light3 = torch.from_numpy(self.raw_lights[547]).type(torch.FloatTensor).cuda()
        light4 = torch.from_numpy(self.raw_lights[28]).type(torch.FloatTensor).cuda()
        light5 = torch.from_numpy(self.raw_lights[34]).type(torch.FloatTensor).cuda()
        self.pre_lighting = [light0, light1, light2, light3, light4, light5]
        
        self.zero_padding = torch.zeros(1, 18, 1).to(self.device)

        self.gap_dic = {
            i: StyleFlowEditor.max_dic[i] - StyleFlowEditor.min_dic[i]
            for i in StyleFlowEditor.max_dic
        }
        self.light_gap_dic = {
            i: StyleFlowEditor.light_max_dic[i] - StyleFlowEditor.light_min_dic[i]
            for i in StyleFlowEditor.light_max_dic
        }

    def _allocate_entity(self, idx):
        self.w_current_np = self.all_w[idx].copy()
        self.attr_current = self.all_attr[idx].copy()
        self.light_current = self.all_lights[idx].copy()

        self.attr_current_list = [self.attr_current[i][0] for i in range(len(self.attr_order))]
        self.light_current_list = [0.0 for _ in range(len(self.lighting_order))]

        array_source = torch.from_numpy(self.attr_current).type(torch.FloatTensor).to(self.device)
        self.array_light = torch.from_numpy(self.light_current).type(torch.FloatTensor).to(self.device)
        self.pre_lighting_distance = [self.pre_lighting[i] - self.array_light for i in range(len(self.lighting_order))]

        self.final_array_target = torch.cat([self.array_light, array_source.unsqueeze(0).unsqueeze(-1)], dim=1)

        self.initial_w = torch.from_numpy(self.w_current_np).to(self.device)

    def _invert_to_real(self, name, edit_power):
        return float(float(edit_power) * self.gap_dic[name] + StyleFlowEditor.min_dic[name])

    def _light_invert_to_real(self, name, edit_power):
        return float(float(edit_power) * self.light_gap_dic[name] + StyleFlowEditor.light_min_dic[name])

    def real_time_editing(self, attr_index, edit_power):
        """
        :param int attr_index: modification index
        :param float edit_power: modification magnitude.
            0 corresponds to minimum value of attribute and 1 corresponds to maximum value
        :return Tuple[torch.Tensor, torch.Tensor]: initial and modified latent vectors
        """
        fws = self.model(self.initial_w, self.final_array_target, self.zero_padding, reverse=False)

        real_value = self._invert_to_real(StyleFlowEditor.attr_order[attr_index], edit_power)
        attr_change = real_value - self.attr_current_list[attr_index]
        attr_final = StyleFlowEditor.attr_degree_list[attr_index] * attr_change + self.attr_current_list[attr_index]

        final_array_target = self.final_array_target.clone()
        final_array_target[0, attr_index + 9, 0, 0] = attr_final

        rev = self.model(fws[0], final_array_target, self.zero_padding, reverse=True)

        if attr_index == 0:
            rev[0][0][8:] = self.initial_w[0][8:]
        elif attr_index == 1:
            rev[0][0][:2] = self.initial_w[0][:2]
            rev[0][0][4:] = self.initial_w[0][4:]
        elif attr_index == 2:
            rev[0][0][4:] = self.initial_w[0][4:]
        elif attr_index == 3:
            rev[0][0][4:] = self.initial_w[0][4:]
        elif attr_index == 4:
            rev[0][0][6:] = self.initial_w[0][6:]
        elif attr_index == 5:
            rev[0][0][:5] = self.initial_w[0][:5]
            rev[0][0][10:] = self.initial_w[0][10:]
        elif attr_index == 6:
            rev[0][0][0:4] = self.initial_w[0][0:4]
            rev[0][0][8:] = self.initial_w[0][8:]
        elif attr_index == 7:
            rev[0][0][:4] = self.initial_w[0][:4]
            rev[0][0][6:] = self.initial_w[0][6:]

        return self.initial_w, rev[0].detach().clone()

    def real_time_lighting(self, light_index, edit_power):
        """
        :param int light_index: modification index
        :param float edit_power: modification magnitude.
            0 corresponds to minimum value of attribute and 1 corresponds to maximum value
        :return Tuple[torch.Tensor, torch.Tensor]: initial and modified latent vectors
        """
        fws = self.model(self.initial_w, self.final_array_target, self.zero_padding, reverse=False)

        real_value = self._light_invert_to_real(self.lighting_order[light_index], edit_power)

        self.light_current_list[light_index] = real_value

        # calculate attributes array first, then change the values of attributes
        lighting_final = self.array_light.clone().detach()
        for i in range(len(self.lighting_order)):
            lighting_final += self.light_current_list[i] * self.pre_lighting_distance[i]

        final_array_target = self.final_array_target.clone()
        final_array_target[:, :9] = lighting_final

        rev = self.model(fws[0], final_array_target, self.zero_padding, reverse=True)

        rev[0][0][0:7] = self.initial_w[0][0:7]
        rev[0][0][12:18] = self.initial_w[0][12:18]

        return self.initial_w, rev[0].detach().clone()

