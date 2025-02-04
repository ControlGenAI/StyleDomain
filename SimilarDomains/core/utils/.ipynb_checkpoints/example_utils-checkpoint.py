import os
import numpy as np
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
import PIL


from argparse import Namespace
from torchvision import transforms
from torchvision.utils import make_grid
from core.uda_models import uda_models
from core.utils.common import (
    get_trainable_model_state, get_stylegan_conv_dimensions, align_face,
    compose_text_with_templates, load_clip
)
from core.parametrizations import BaseParametrization
from core.mappers import mapper_registry
from restyle_encoders.psp import pSp
from restyle_encoders.e4e import e4e


def read_img(img_path, align_input=False):
    if align_input:
        return run_alignment(img_path)
    else:
        return PIL.Image.open(img_path)
    
    
class Inferencer(nn.Module):
    def __init__(self, ckpt, device):
        super().__init__()
        
        self.device = device
        
        if len(ckpt) > 1:
            self._reset_trainable(ckpt)    
        
        self.sg2_source = uda_models['stylegan2'](
            **ckpt['sg2_params']
        )
        
        if 'patch_key' in ckpt:
            self.sg2_source.patch_layers(ckpt['patch_key'])
            
        self.sg2_source.freeze_layers()
        self.sg2_source.to(self.device)
        
    @torch.no_grad()
    def forward(self, latents, text_description=None, mx_n=1, no_mixing=False, **kwargs):
        src_imgs, _ = self.sg2_source(latents, **kwargs)
        
        if not kwargs.get('input_is_latent', False):
            if self.model_type == 'original':
                latents = self.model_da.style(latents)
            else:
                latents = self.sg2_source.style(latents)
            kwargs['input_is_latent'] = True
                
        if not kwargs.get('truncation', False):
            kwargs['truncation'] = 1
        
        if self.da_type == 'im2im' and not no_mixing:
            latents = self._mtg_mixing_noise(latents, truncation=kwargs['truncation'], pw=mx_n)
            kwargs.pop('truncation')
        
        if self.model_type == 'original':
            trg_imgs, _ = self.model_da(latents, **kwargs)
        elif self.model_type == 'mapper':
            if text_description is not None:
                text_encoded = self._encode_text(text_description)
                offsets = self.model_da(text_encoded)
            else:
                offsets = None
                
            trg_imgs, _ = self.sg2_source(latents, offsets=offsets, **kwargs)
        elif self.model_type == 'parametrization':
            trg_imgs, _ = self.sg2_source(latents, offsets=self.model_da(), **kwargs)
            
        else:
            trg_imgs = None
        
        return src_imgs, trg_imgs
    
    def _mtg_mixing_noise(self, latents, truncation=1, pw=1.):
        w_styles = latents[0]
        n_lat = self.sg2_source.generator.n_latent
        if w_styles.ndim == 2:
            w_styles = w_styles.unsqueeze(1).repeat(1, n_lat, 1)
        
        gen_mean = self.sg2_source.mean_latent.unsqueeze(1).repeat(1, n_lat, 1)
        style_mixing_latents = truncation * (w_styles - gen_mean) + gen_mean
        style_mixing_latents[:, 7:, :] = pw * self.style_latents + (1 - pw) * style_mixing_latents[:, 7:, :]
        return [style_mixing_latents]
    
    @torch.no_grad()
    def _encode_text(self, text, templates=("{}", )):
        text = compose_text_with_templates(text, templates=templates)
        tokens = clip.tokenize(text).to(self.device)
        text_features = self.encoder.encode_text(tokens).detach().float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def _reset_trainable(self, ckpt):
        self.model_type = ckpt['model_type']
        self.da_type = ckpt['da_type']
        
        if self.model_type == 'original':
            self.model_da = uda_models['stylegan2'](
                **ckpt['sg2_params']
            )
            self.model_da.freeze_layers()
        elif self.model_type == 'mapper':
            self.encoder, _ = load_clip(ckpt['clip_encoder'], self.device)
            self.model_da = mapper_registry[
                ckpt['mapper_config']['mapper_type']
            ](
                ckpt['mapper_config'],
                get_stylegan_conv_dimensions(ckpt['sg2_params']['img_size']),
            )
        else:
            self.model_da = BaseParametrization(
                ckpt['patch_key'],
                get_stylegan_conv_dimensions(ckpt['sg2_params']['img_size']),
            )
        self.model_da.load_state_dict(ckpt['state_dict'], strict=False)
        self.model_da.to(self.device).eval()
        
        if self.da_type == 'im2im':
            self.style_latents = ckpt['style_latents'].to(self.device)
        
    
@torch.no_grad()
def get_avg_image(net):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    return avg_image.to('cuda').float().detach()


def run_on_batch(inputs, net, opts):
    y_hat, latent = None, None
    results_batch = {idx: [] for idx in range(inputs.shape[0])}
    results_latent = {idx: [] for idx in range(inputs.shape[0])}
    
    avg_image = get_avg_image(net)
    avg_image_for_batch = avg_image.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
    x_input = torch.cat([inputs, avg_image_for_batch], dim=1)
    
    for iter in range(opts.n_iters_per_batch):
        y_hat, latent = net.forward(
            x_input,
            latent=latent,
            randomize_noise=False,
            return_latents=True,
            resize=opts.resize_outputs
        )

        for idx in range(inputs.shape[0]):
            results_batch[idx].append(y_hat[idx])
            results_latent[idx].append(latent[idx].unsqueeze(0))

        # resize input to 256 before feeding into next iteration
        y_hat = net.face_pool(y_hat)
        x_input = torch.cat([inputs, y_hat], dim=1)

    return results_batch, results_latent


def load_latent(path):
    return torch.from_numpy(np.load(path)).unsqueeze(0)


def get_celeb_latents(names):
    if not isinstance(names, list):
        return load_latent(f'examples/celeb_latents/{names}.npy')
    
    return torch.cat([
        load_latent(f'examples/celeb_latents/{name}.npy') for name in names
    ], dim=0)


def to_im(torch_image, **kwargs):
    return transforms.ToPILImage()(
        make_grid(torch_image, value_range=(-1, 1), normalize=True, **kwargs)
    )


@torch.no_grad()
def project_e4e(img, model_path, name=None, device='cuda:0'):
    # model_path = 'models/e4e_ffhq_encode.pt'
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = e4e(opts).eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    img = transform(img).unsqueeze(0).to(device)
    images, w_plus = net(img, randomize_noise=False, return_latents=True)
    if name is not None:
        result_file = {}
        result_file['latent'] = w_plus[0]
        torch.save(result_file, name)
    return images, w_plus



@torch.no_grad()
def project_restyle_psp(img, model_path, name=None, device='cuda:0'):
    ### load_model
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)
    opts.n_iters_per_batch = 5
    opts.resize_outputs = False
    ### 
    
    net = pSp(opts).eval().to(device)
    
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    img = transform(img).unsqueeze(0).to(device)
    
    results_im, result_latents = run_on_batch(img, net, opts)
    if name is not None:
        result_file = {}
        result_file['latent'] = result_latents[0]
        torch.save(result_file, name)
    return results_im[0], result_latents[0]


def run_alignment(image_path, predictor_path="pretrained/shape_predictor_68_face_landmarks.dat"):
    import dlib
    if not os.path.exists(predictor_path):
        print('dlib shape predictor is not downloaded; launch `python download.py --load_type=dlib`')
    predictor = dlib.shape_predictor(predictor_path)
    aligned_image = align_face(filepath=image_path, predictor=predictor) 
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def hstack_with_lines(list_of_images, width):
    h, w, _ = np.array(list_of_images[0]).shape
    
    new_list = []
    
    for im in list_of_images:
        new_list.append(im)
        new_list.append(np.ones((h, width, 3), dtype=np.uint8) * 255)
    return np.hstack(new_list[:-1])


def vstack_with_lines(list_of_images, height):
    h, w, _ = np.array(list_of_images[0]).shape
    new_list = []
    
    for im in list_of_images:
        new_list.append(im)
        new_list.append(np.ones((height, w, 3), dtype=np.uint8) * 255)
    return np.vstack(new_list[:-1])


def insert_image(image, desired_size):
    s, _, _ = image.shape
    desired_image = np.ones((desired_size, desired_size, 3), dtype=np.uint8) * 255
    delta = (desired_size - s) // 2
    
    desired_image[delta: desired_size - delta, delta: desired_size - delta] = image
    return desired_image
