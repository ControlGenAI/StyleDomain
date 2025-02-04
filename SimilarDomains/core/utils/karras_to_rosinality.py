import sys
import click
import pickle
import math
import torch
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from pathlib import Path
from collections import OrderedDict
from gan_models.StyleGAN2.model import Generator


def get_const_input(ada_g_state):
    return {'input.input': ada_g_state['synthesis.b4.const'].unsqueeze(0)}


def get_mapping_state(state_dict, n_mlp):
    od = OrderedDict()
    for layer_idx in range(n_mlp):
        for module in ['weight', 'bias']:
            ada_key = f'mapping.fc{layer_idx}.{module}'
            new_key = f'style.{layer_idx + 1}.{module}'
            od[new_key] = state_dict[ada_key]
    return od


def get_to_rgbs(ada_g_state, img_size):
    to_rgbs = OrderedDict()
    keys = []
    
    log_img = int(math.log(img_size, 2)) + 1
    for ros_i, ada_i in enumerate(range(3, log_img)):        
        img_res = 2 ** ada_i
        
        to_rgbs[f'to_rgbs.{ros_i}.conv.weight'] = ada_g_state[f'synthesis.b{img_res}.torgb.weight'].unsqueeze(0)
        r = torch.numel(ada_g_state[f'synthesis.b{img_res}.torgb.bias'])
        to_rgbs[f'to_rgbs.{ros_i}.bias'] = ada_g_state[f'synthesis.b{img_res}.torgb.bias'].view(1, r, 1, 1)
        to_rgbs[f'to_rgbs.{ros_i}.conv.modulation.weight'] = ada_g_state[f'synthesis.b{img_res}.torgb.affine.weight']
        to_rgbs[f'to_rgbs.{ros_i}.conv.modulation.bias'] = ada_g_state[f'synthesis.b{img_res}.torgb.affine.bias']
    
    to_rgbs[f'to_rgb1.conv.weight'] = ada_g_state[f'synthesis.b4.torgb.weight'].unsqueeze(0)
    r = torch.numel(ada_g_state[f'synthesis.b4.torgb.bias'])
    to_rgbs[f'to_rgb1.bias'] = ada_g_state[f'synthesis.b4.torgb.bias'].view(1, r, 1, 1)
    to_rgbs[f'to_rgb1.conv.modulation.weight'] = ada_g_state[f'synthesis.b4.torgb.affine.weight']
    to_rgbs[f'to_rgb1.conv.modulation.bias'] = ada_g_state[f'synthesis.b4.torgb.affine.bias']
    
    return to_rgbs


def get_convs(ada_g_state, img_size):    
    convs = OrderedDict()
    noises = OrderedDict()
    keys = []
    
    log_img = int(math.log(img_size, 2)) + 1
    ros_i = 0
    for ada_i in range(3, log_img):
        img_res = 2 ** ada_i
        
        for ada_conv_idx in range(2):
            # modConv
            convs[f'convs.{ros_i}.conv.weight'] = ada_g_state[f'synthesis.b{img_res}.conv{ada_conv_idx}.weight'].unsqueeze(0)
            convs[f'convs.{ros_i}.activate.bias'] = ada_g_state[f'synthesis.b{img_res}.conv{ada_conv_idx}.bias']
            
            # W to Style Space
            convs[f'convs.{ros_i}.conv.modulation.weight'] = ada_g_state[f'synthesis.b{img_res}.conv{ada_conv_idx}.affine.weight']
            convs[f'convs.{ros_i}.conv.modulation.bias'] = ada_g_state[f'synthesis.b{img_res}.conv{ada_conv_idx}.affine.bias']
            
            # Noise
            convs[f'convs.{ros_i}.noise.weight'] = ada_g_state[f'synthesis.b{img_res}.conv{ada_conv_idx}.noise_strength'].unsqueeze(0)
            if ros_i % 2 == 0:
                convs[f'convs.{ros_i}.conv.blur.kernel'] = ada_g_state[f'synthesis.b{img_res}.conv{ada_conv_idx}.resample_filter'] * 4
            
            ros_i += 1
            
            noises[f'noises.noise_{ros_i}'] = ada_g_state[f'synthesis.b{img_res}.conv{ada_conv_idx}.noise_const'].unsqueeze(0).unsqueeze(0)
    
    convs[f'conv1.conv.weight'] = ada_g_state[f'synthesis.b4.conv1.weight'].unsqueeze(0)
    convs[f'conv1.activate.bias'] = ada_g_state[f'synthesis.b4.conv1.bias']

    # W to Style Space
    convs[f'conv1.conv.modulation.weight'] = ada_g_state[f'synthesis.b4.conv1.affine.weight']
    convs[f'conv1.conv.modulation.bias'] = ada_g_state[f'synthesis.b4.conv1.affine.bias']

    # Noise
    convs[f'conv1.noise.weight'] = ada_g_state[f'synthesis.b4.conv1.noise_strength'].unsqueeze(0)

    noises[f'noises.noise_0'] = ada_g_state[f'synthesis.b4.conv1.noise_const'].unsqueeze(0).unsqueeze(0)
    
    return convs, noises
    
    
@click.command()
@click.option('--src', help='Path to pkl.')
@click.option('--dst', help='Path to result file.')
@click.option('--img_size', help='final img size of generated images', type=int)
def main(src, dst, img_size):
    print('[+] Reading checkpoint...')
    with open(src, "rb") as f:
        data = pickle.load(f)

    ada_g_state = data['G_ema'].state_dict()
    
    print('[+] Parsing...')
    c, n = get_convs(ada_g_state, img_size=img_size)
    st = get_mapping_state(ada_g_state, n_mlp=8)
    con = get_const_input(ada_g_state)
    r = get_to_rgbs(ada_g_state, img_size=img_size)
    
    state = {
        **c, **n, **st, **con, ** r
    }

    g = Generator(img_size, 512, 8)
    print('[+] Checkpoint verified, saving...')
    g.load_state_dict(state, strict=False)
    torch.save({'g_ema': g.state_dict()}, dst)
    print('[+] Saved. OK')

    
if __name__ == '__main__':
    main()
    