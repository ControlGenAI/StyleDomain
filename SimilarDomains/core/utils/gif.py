import torch
import numpy as np


from copy import deepcopy
from core.utils.example_utils import to_im
from torchvision.transforms import Resize
from PIL import ImageFont, Image
from collections import OrderedDict


def morph_checkpoints(ckpt_l, ckpt_r, alpha):
    new_checkpoint = {k: v for k, v in ckpt_l.items() if k not in ('state_dict', 'style_latents')}
    new_checkpoint['state_dict'] = OrderedDict()
    
    for k in ckpt_l['state_dict']:
        v1, v2 = ckpt_l['state_dict'][k].clone(), ckpt_r['state_dict'][k].clone()
        new_checkpoint['state_dict'][k] = alpha * v1 + (1 - alpha) * v2
    
    if 'style_latents' in ckpt_l or 'style_latents' in ckpt_r:
        new_checkpoint['da_type'] = 'im2im'
    
    
    if 'style_latents' in ckpt_l and 'style_latents' in ckpt_r:
        v1, v2 = ckpt_l['style_latents'].clone(), ckpt_r['style_latents'].clone()
        new_checkpoint['style_latents'] = alpha * v1 + (1 - alpha) * v2
    elif 'style_latents' in ckpt_l:
        new_checkpoint['style_latents'] = ckpt_l['style_latents'].clone()
    elif 'style_latents' in ckpt_r:
        new_checkpoint['style_latents'] = ckpt_r['style_latents'].clone()
    
    return new_checkpoint


def text_centered(image, message, font, fontColor):
    W, H = image.size
    image = image.copy()
    draw = ImageDraw.Draw(image)
    _, _, w, h = draw.textbbox((0, 0), message, font=font)
    draw.text(((W - w) / 2, (H - h) / 2), message, fill=fontColor, font=myfont)
    return image


class GIF:
    def __init__(
        self, model, initial_latents, caption_mode=None,
        steps_per_stage=20, steps_between=20, randomize_noise=True, img_size=256,
        font_style_p="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        font_size=16
    ):
        self.stack = []
        self.model = model
        self.lats = initial_latents
        self.step_stage = steps_per_stage
        self.step_between = steps_between
        self.randomize_noise = randomize_noise
        
        self.font = ImageFont.truetype(font_style_p, font_size)
        self.img_size = img_size
        
        self.stylized = False
        self.current_style = None
        self.caption_mode = caption_mode
    
    def caption_image(self, image, info):
        if self.caption_mode == 'text':
            image = self._text_caption(image, info)
        elif self.caption_mode == 'images_arrow':
            image = self._im_arrow_caption(image, info)
        return image
    
    def _text_caption(self, image, caption, fontColor=(0, 0, 0)):
        W, H = im.size

        image = Image.fromarray(np.ones((H // 3, W, 3), np.uint8) * 255)
        capt = text_centered(image, caption, self.font, fontColor)
        return Image.fromarray(np.vstack([im, capt]))
    
    def add_stage(
        self, stage_type, stage_info
    ):        
        tmp_stage = []
        
        for self.step, self.alpha in enumerate(np.linspace(0, 1, self.step_stage)):
            if stage_type == 'unstylize':
                im = self.unstylize_stage(stage_info)
            elif stage_type == 'stylize':
                im = self.stylize_stage(stage_info)
                capt_info = {
                    'stylization': stage_info
                }
            elif stage_type == 'style_morphing':
                im = self.style_morphing_stage(stage_info)
            elif stage_type == 'lat_morphing':
                im = self.lat_morphing_stage(stage_info)
            
            im = to_im(Resize(self.img_size, antialias=True)(im))
            
            capt_info = None
            self.caption_image(im, capt_info)
            
            tmp_stage.append(im)
        
        self.stack.extend(tmp_stage)

        if self.step_between > 0:
            t = [deepcopy(im) for _ in range(self.step_between)]
            self.stack.extend(t)
            
        return self
        
    def style_morphing_stage(self, info):
        if not self.stylized:
            raise ValueError("To Launch style morphing stylisation stage is needed")
        
        if self.step == 0:
            self.next_style_ckpt = torch.load(info)
        
        tmp_ckpt = morph_checkpoints(self.next_style_ckpt, self.current_style_ckpt, self.alpha)
        self.model._reset_trainable(tmp_ckpt)
        
        if self.next_style_ckpt['da_type'] == 'im2im' and self.current_style_ckpt['da_type'] == 'td':
            mn = self.alpha
        elif self.next_style_ckpt['da_type'] == 'td' and self.current_style_ckpt['da_type'] == 'im2im':
            mn = 1 - self.alpha
        elif self.next_style_ckpt['da_type'] == 'im2im' and self.current_style_ckpt['da_type'] == 'im2im':
            mn = 1
        else:
            mn = 0
        
        _, t_im = self.model(
            self.lats, input_is_latent=True, 
            offset_power=1., mx_n=mn,
            randomize_noise=self.randomize_noise
        )
        
        if self.step + 1 == self.step_stage:
            self.current_style_ckpt = self.next_style_ckpt
            
        return t_im
    
    def lat_morphing_stage(self, info):
        
        if self.step == 0:
            self.next_person, self.next_lat = info
        
        tmp_lat = [
            self.next_lat[0] * self.alpha + self.lats[0] * (1 - self.alpha) 
        ]
        _, t_im = self.model(
            tmp_lat, input_is_latent=True, 
            offset_power=1., 
            randomize_noise=self.randomize_noise
        )
        
        if self.step + 1 == self.step_stage:
            self.lats = self.next_lat
        
        return t_im
    
    def unstylize_stage(self, info):
        if self.step == self.step_stage - 1:
            self.stylized = False
        
        if self.current_style_ckpt['da_type'] == 'td':
            mn = 0
        else:
            mn = 1 - self.alpha
        
        _, t_im = self.model(
            self.lats, input_is_latent=True, 
            offset_power=(1 - self.alpha), mx_n=mn,
            randomize_noise=self.randomize_noise
        )
        return t_im
    
    def stylize_stage(self, info):
        if self.step == 0:
            self.current_style_ckpt = torch.load(info)
            self.model._reset_trainable(
                self.current_style_ckpt
            )
            self.stylized = True
        
        _, t_im = self.model(
            self.lats, input_is_latent=True, 
            offset_power=self.alpha, mx_n=self.alpha,
            randomize_noise=self.randomize_noise
        )
        
        return t_im
    
    def render(self, fp_out, fps=25):
        self.stack[0].save(
            fp_out,
            save_all=True, 
            append_images=self.stack[1:], 
            optimize=False, duration=1000/fps, loop=0
        )
        

def get_white(size_h, size_w):
    return np.ones((size_h, size_w, 3), dtype=np.uint8) * 255


def text_centered(image, message, font, fontColor):
    W, H = image.size
    image = image.copy()
    draw = ImageDraw.Draw(image)
    _, _, w, h = draw.textbbox((0, 0), message, font=font)
    draw.text(((W - w) / 2, (H - h) / 2), message, fill=fontColor, font=myfont)
    return image