import torch
import numpy as np


import torch

from torch_utils import misc
from gan_models.StyleGAN2.nvidia import (
    MappingNetwork, upfirdn2d, ToRGBLayer, bias_act, FullyConnectedLayer, modulated_conv2d
)



class SynthesisLayerOffsets(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, offsets, noise_mode='random', fused_modconv=True, gain=1, patch='s_delta'):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.in_channels, in_resolution, in_resolution])
        
        styles = self.affine(w)
        
        assert patch in ['s_delta', 'cin_mult'], 'incorrect patch'
        
        if offsets is not None and patch == 's_delta':
            styles = styles + offsets
        elif offsets is not None and patch == 'cin_mult':
            styles = styles * (torch.ones_like(offsets) + offsets)
        
        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


class SynthesisBlockOffsets(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels, 0 = first block.
        out_channels,                           # Number of output channels.
        w_dim,                                  # Intermediate latent (W) dimensionality.
        resolution,                             # Resolution of this block.
        img_channels,                           # Number of output color channels.
        is_last,                                # Is this the last block?
        architecture            = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter         = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp              = 256,          # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16                = False,        # Use FP16 for this block?
        fp16_channels_last      = False,        # Use channels-last memory format with FP16?
        fused_modconv_default   = True,         # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
        **layer_kwargs,                         # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayerOffsets(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayerOffsets(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, offsets, force_fp32=False, fused_modconv=None, update_emas=False, **layer_kwargs):
        _ = update_emas # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            if offsets is not None:
                x = self.conv1(x, next(w_iter), offsets['conv1'], fused_modconv=fused_modconv, **layer_kwargs)
            else:
                x = self.conv1(x, next(w_iter), None, fused_modconv=fused_modconv, **layer_kwargs)
                
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            if offsets is not None:
                x = self.conv0(x, next(w_iter), offsets['conv0'], fused_modconv=fused_modconv, **layer_kwargs)
            else:
                x = self.conv0(x, next(w_iter), None, fused_modconv=fused_modconv, **layer_kwargs)
            
            
            if offsets is not None:
                x = self.conv1(x, next(w_iter), offsets['conv1'], fused_modconv=fused_modconv, **layer_kwargs)
            else:
                x = self.conv1(x, next(w_iter), None, fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'


class SynthesisNetworkOffsets(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 4,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlockOffsets(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, offsets, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            
            if offsets is not None:
                x, img = block(x, img, cur_ws, offsets[f"b{res}"], **block_kwargs)
            else:
                x, img = block(x, img, cur_ws, None, **block_kwargs)
        
        return img

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])


class GeneratorOffsets(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetworkOffsets(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, offsets=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, offsets, update_emas=update_emas, **synthesis_kwargs)
        return img