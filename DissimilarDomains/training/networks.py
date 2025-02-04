# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import regex
from typing import Union

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma


# ----------------------------------------------------------------------------

# Full list of available weights offsets
out_in_base_parametrizations = [
    'in', 'out', 'spatial', 'in_spatial', 'out_spatial', 'out_in',
]
out_in_base_parametrizations += ['_'.join([_, 'additive']) for _ in out_in_base_parametrizations]
out_in_decomposition_parametrizations = [
    'out+in',
    *[f'out_in_{k}' for k in [1, 5, 10, 20, 50, 100, 256, 512]],
    *[f'out_in_{k}_dual' for k in [1, 5, 10, 20, 50, 100, 256, 512]],
    *[f'out_in_{k}_{t}' for k in [1, 5, 10, 20, 50, 100, 256, 512] for t in range(1, 512 // k + 1)],
    *[f'out_in_{k}_{t}_train_in' for k in [1, 5, 10, 20, 50, 100, 256, 512] for t in range(1, 512 // k + 1)],
    *[f'out_in_{k}_{t}_train_out' for k in [1, 5, 10, 20, 50, 100, 256, 512] for t in range(1, 512 // k + 1)],
]
out_in_decomposition_parametrizations += ['_'.join([_, 'additive']) for _ in out_in_decomposition_parametrizations]

_plus_pattern = regex.compile(r'^out\+in(?:_additive)?$')
_single_pattern = regex.compile(r'^out_in_([0-9]+)(?:_additive)?$')
_dual_pattern = regex.compile(r'^out_in_([0-9]+)_dual(?:_additive)?$')
_train_pattern = regex.compile(r'^out_in_([0-9]+)_([0-9]+)(?:_additive)?$')
_train_in_pattern = regex.compile(r'^out_in_([0-9]+)_([0-9]+)_train_in(?:_additive)?$')
_train_out_pattern = regex.compile(r'^out_in_([0-9]+)_([0-9]+)_train_out(?:_additive)?$')

# Full list of available weights offsets for affine layers
affine_out_in_decomposition_parametrizations = [
    *[f'affine_out_in_{k}_{t}' for k in [1, 5, 10, 20, 50, 100, 256, 512] for t in range(1, 512 // k + 1)],
]
affine_out_in_decomposition_parametrizations += [
    '_'.join([_, 'additive']) for _ in affine_out_in_decomposition_parametrizations
]
_affine_train_pattern = regex.compile(r'^affine_out_in_([0-9]+)_([0-9]+)(?:_additive)?$')


def get_style_space_modifications(layer_idx, styles, style_space_modifications):
    r"""Generate Style Space modification vector for a given layer.
        Also generate mask for Style Space Directions

    Args:
        layer_idx (int): Layer number
        styles (torch.Tensor): Style latents for the current layer of size [batch, channels] 
        style_space_modifications (List[Tuple[Tuple[int, int], float, float]]): list of modifications
            to apply in form ((layer, channel), magnitude, offset_factor). So, given Style Space latents S 
            they will be modified as following: S[layer, channel] += magnitude. 
            If Style Space Directions \Delta are given, and Style Space modifications are applied before Style Space Directions
            then they will be adjusted: \Delta[layer, channel] *= offset_factor
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            Style Space modification to the current layer and Style Space Directions mask
    """
    if style_space_modifications is None:
        return None, None

    modifications_shape = [1] * styles.ndim
    modifications_shape[1] = styles.shape[1]
    modifications = torch.zeros(modifications_shape, device=styles.device, dtype=styles.dtype)
    offset_factors = torch.ones_like(modifications)
    for (layer, channel), magnitude, offset_factor in style_space_modifications:
        if layer_idx == layer:
            modifications[:, channel] = magnitude
            offset_factors[:, channel] = offset_factor

    return modifications, offset_factors


def split_parameterization(use_domain_modulation, domain_modulation_parametrization):
    if use_domain_modulation:
        domain_modulation_parametrizations = np.array(domain_modulation_parametrization.split(','))

        # Split parametrizations on domain modulation and weights modulation
        is_weights_modulation = np.isin(
            domain_modulation_parametrizations,
            out_in_base_parametrizations + out_in_decomposition_parametrizations
        )
        is_affine_weights_modulation = np.isin(
            domain_modulation_parametrizations,
            affine_out_in_decomposition_parametrizations
        )

        weights_domain_modulation_parametrization = (
            domain_modulation_parametrizations[is_weights_modulation]
        )
        affine_weights_domain_modulation_parametrization = (
            domain_modulation_parametrizations[is_affine_weights_modulation]
        )
        domain_modulation_parametrization = (
            domain_modulation_parametrizations[(~is_weights_modulation) & (~is_affine_weights_modulation)]
        )

        def _unpack_at_most_one(values, message):
            if len(values) == 0:
                return None
            elif len(values) == 1:
                return values[0]
            else:
                raise ValueError(f'{message}: {values}')

        # Allow at most one domain modulation parametrization
        domain_modulation_parametrization = _unpack_at_most_one(
            domain_modulation_parametrization, 'Too many types of domain modulation'
        )
        # Allow at most one weights modulation parametrization
        weights_domain_modulation_parametrization = _unpack_at_most_one(
            weights_domain_modulation_parametrization, 'Too many types of domain modulation'
        )
        # Allow at most one affine weights modulation parametrization
        affine_weights_domain_modulation_parametrization = _unpack_at_most_one(
            affine_weights_domain_modulation_parametrization, 'Too many types of domain modulation'
        )

        return (
            domain_modulation_parametrization,
            weights_domain_modulation_parametrization,
            affine_weights_domain_modulation_parametrization
        )
    return None, None, None


def register_domain_modulation(self: "Union[SynthesisLayer, ToRGBLayer]", in_channels, w_dim):
    print(
        'Register domain modulation for {0}. IN: {1:d}, W: {2:d}'.format(
            ('b{0:d}.conv' if self.is_rgb else 'b{0:d}.torgb').format(self.resolution), in_channels, w_dim
        )
    )

    if self.domain_modulation_parametrization == 'multiplicative':
        self.register_buffer('ones', torch.ones((1, in_channels)))
        self.offset = torch.nn.Parameter(torch.zeros(1, in_channels))
    elif self.domain_modulation_parametrization == 'additive':
        self.offset = torch.nn.Parameter(torch.zeros(1, in_channels))
    elif self.domain_modulation_parametrization == 'multiplicative_w_space':
        self.register_buffer('ones', torch.ones((1, w_dim)))
        self.offset = torch.nn.Parameter(torch.zeros(1, w_dim))
    elif self.domain_modulation_parametrization == 'additive_w_space':
        self.offset = torch.nn.Parameter(torch.zeros(1, w_dim))
    else:
        raise ValueError(f'Unknown type of domain modulation: {self.domain_modulation_parametrization}')

    self.register_buffer('offset_mask', torch.ones_like(self.offset))


def register_affine_weights_domain_modulation(self: "FullyConnectedLayer"):
    out_features, in_features = self.weight.shape

    if self.affine_weights_domain_modulation_parametrization in affine_out_in_decomposition_parametrizations:
        if regex.match(_affine_train_pattern, self.affine_weights_domain_modulation_parametrization):
            [(weights_offset_rank, terms_num)] = regex.findall(
                _affine_train_pattern, self.affine_weights_domain_modulation_parametrization
            )
            weights_offset_rank, terms_num = int(weights_offset_rank), int(terms_num)

            self.weights_offset_in = []
            self.weights_offset_out = []
            for idx in range(terms_num):
                self.register_parameter(
                    f'weights_offset_in_{idx}',
                    torch.nn.Parameter(torch.randn(weights_offset_rank, in_features))
                )
                self.register_parameter(
                    f'weights_offset_out_{idx}',
                    torch.nn.Parameter(torch.zeros(out_features, weights_offset_rank))
                )

                self.weights_offset_in.append(
                    (self.get_parameter, f'weights_offset_in_{idx}'),
                )
                self.weights_offset_out.append(
                    (self.get_parameter, f'weights_offset_out_{idx}'),
                )

            self.register_buffer('ones_weights', self.weight.new_ones(out_features, in_features))

            print('Register affine weights offsets. OUT: {0:d}, K: {1:d}, IN: {2:d}, T: {3}'.format(
                out_features, weights_offset_rank, in_features, self.affine_weights_domain_modulation_parametrization
            ))
        else:
            raise ValueError(
                f'Unknown type of domain modulation: {self.affine_weights_domain_modulation_parametrization}'
            )

        for getter, name in self.weights_offset_in:
            tensor = getter(name)
            if torch.linalg.norm(tensor.data).item() > 1e-6:
                tensor.data /= torch.linalg.norm(
                    tensor.data, dim=1, keepdim=True
                )

        for getter, name in self.weights_offset_out:
            tensor = getter(name)
            if torch.linalg.norm(tensor.data).item() > 1e-6:
                tensor.data /= torch.linalg.norm(
                    tensor.data, dim=0, keepdim=True
                )
    else:
        raise ValueError(f'Unknown type of domain modulation: {self.affine_weights_domain_modulation_parametrization}')


def register_weights_domain_modulation(self: "Union[SynthesisLayer, ToRGBLayer]"):
    out_channels, in_channels, kernel_size_w, kernel_size_h = self.weight.shape
    if self.weights_domain_modulation_parametrization in ['in', 'in_additive']:
        self.weights_offset = torch.nn.Parameter(torch.zeros(1, in_channels, 1, 1))
    elif self.weights_domain_modulation_parametrization in ['out', 'out_additive']:
        self.weights_offset = torch.nn.Parameter(torch.zeros(out_channels, 1, 1, 1))
    elif self.weights_domain_modulation_parametrization in ['spatial', 'spatial_additive']:
        self.weights_offset = torch.nn.Parameter(torch.zeros(1, 1, kernel_size_w, kernel_size_h))
    elif self.weights_domain_modulation_parametrization in ['in_spatial', 'in_spatial_additive']:
        self.weights_offset = torch.nn.Parameter(torch.zeros(1, in_channels, kernel_size_w, kernel_size_h))
    elif self.weights_domain_modulation_parametrization in ['out_spatial', 'out_spatial_additive']:
        self.weights_offset = torch.nn.Parameter(torch.zeros(out_channels, 1, kernel_size_w, kernel_size_h))
    elif self.weights_domain_modulation_parametrization in ['out_in', 'out_in_additive']:
        self.weights_offset = torch.nn.Parameter(torch.zeros(out_channels, in_channels, 1, 1))
    elif self.weights_domain_modulation_parametrization in out_in_decomposition_parametrizations:
        if regex.match(_plus_pattern, self.weights_domain_modulation_parametrization):
            self.register_parameter(
                f'weights_offset_in_0',
                torch.nn.Parameter(torch.zeros(1, in_channels))
            )
            self.register_parameter(
                f'weights_offset_out_0',
                torch.nn.Parameter(torch.zeros(out_channels, 1))
            )

            self.weights_offset_in = [
                (self.get_parameter, 'weights_offset_in_0'),
            ]
            self.weights_offset_out = [
                (self.get_parameter, 'weights_offset_out_0'),
            ]

            print('Register weights offsets. OUT: {0:d}, K: {1:d}, IN: {2:d}, T: {3}'.format(
                out_channels, -1, in_channels, self.weights_domain_modulation_parametrization
            ))
        elif regex.match(_single_pattern, self.weights_domain_modulation_parametrization):
            [weights_offset_rank] = regex.findall(
                _single_pattern, self.weights_domain_modulation_parametrization
            )
            weights_offset_rank = int(weights_offset_rank)

            self.register_parameter(
                f'weights_offset_in_0',
                torch.nn.Parameter(torch.zeros(weights_offset_rank, in_channels))
            )
            self.register_parameter(
                f'weights_offset_out_0',
                torch.nn.Parameter(torch.zeros(out_channels, weights_offset_rank))
            )

            self.weights_offset_in = [
                (self.get_parameter, 'weights_offset_in_0'),
            ]
            self.weights_offset_out = [
                (self.get_parameter, 'weights_offset_out_0'),
            ]

            print('Register weights offsets. OUT: {0:d}, K: {1:d}, IN: {2:d}, T: {3}'.format(
                out_channels, weights_offset_rank, in_channels, self.weights_domain_modulation_parametrization
            ))
        elif regex.match(_dual_pattern, self.weights_domain_modulation_parametrization):
            # Here we parameterize WO = A_{1} @ B_{1} + A_{2} @ B_{2}, where only A_{1} and B_{2} is trained
            # B_{1} and A_{2} are initialized as orthonormal matrices (w.r.t. columns and rows correspondingly)
            [weights_offset_rank] = regex.findall(
                _dual_pattern, self.weights_domain_modulation_parametrization
            )
            weights_offset_rank = int(weights_offset_rank)

            self.register_buffer('weights_offset_in_0', torch.randn(weights_offset_rank, in_channels))
            self.register_parameter(
                'weights_offset_in_1', torch.nn.Parameter(torch.zeros(weights_offset_rank, in_channels))
            )

            self.register_parameter(
                'weights_offset_out_0', torch.nn.Parameter(torch.zeros(out_channels, weights_offset_rank))
            )
            self.register_buffer('weights_offset_out_1', torch.randn(out_channels, weights_offset_rank))

            self.weights_offset_in = [
                (self.get_buffer, 'weights_offset_in_0'),
                (self.get_parameter, 'weights_offset_in_1'),
            ]
            self.weights_offset_out = [
                (self.get_parameter, 'weights_offset_out_0'),
                (self.get_buffer, 'weights_offset_out_1'),
            ]

            print('Register weights offsets. OUT: {0:d}, K: {1:d}, IN: {2:d}, T: {3}'.format(
                out_channels, weights_offset_rank, in_channels, self.weights_domain_modulation_parametrization
            ))
        elif regex.match(_train_pattern, self.weights_domain_modulation_parametrization):
            [(weights_offset_rank, terms_num)] = regex.findall(
                _train_pattern, self.weights_domain_modulation_parametrization
            )
            weights_offset_rank, terms_num = int(weights_offset_rank), int(terms_num)

            self.weights_offset_in = []
            self.weights_offset_out = []
            for idx in range(terms_num):
                self.register_parameter(
                    f'weights_offset_in_{idx}',
                    torch.nn.Parameter(torch.randn(weights_offset_rank, in_channels))
                )
                self.register_parameter(
                    f'weights_offset_out_{idx}',
                    torch.nn.Parameter(torch.zeros(out_channels, weights_offset_rank))
                )

                self.weights_offset_in.append(
                    (self.get_parameter, f'weights_offset_in_{idx}'),
                )
                self.weights_offset_out.append(
                    (self.get_parameter, f'weights_offset_out_{idx}'),
                )

            print('Register weights offsets. OUT: {0:d}, K: {1:d}, IN: {2:d}, T: {3}'.format(
                out_channels, weights_offset_rank, in_channels, self.weights_domain_modulation_parametrization
            ))
        elif regex.match(_train_in_pattern, self.weights_domain_modulation_parametrization):
            [(weights_offset_rank, terms_num)] = regex.findall(
                _train_in_pattern, self.weights_domain_modulation_parametrization
            )
            weights_offset_rank, terms_num = int(weights_offset_rank), int(terms_num)

            self.weights_offset_in = []
            self.weights_offset_out = []
            for idx in range(terms_num):
                self.register_parameter(
                    f'weights_offset_in_{idx}',
                    torch.nn.Parameter(torch.zeros(weights_offset_rank, in_channels))
                )
                self.register_buffer(
                    f'weights_offset_out_{idx}',
                    torch.randn(out_channels, weights_offset_rank)
                )

                self.weights_offset_in.append(
                    (self.get_parameter, f'weights_offset_in_{idx}'),
                )
                self.weights_offset_out.append(
                    (self.get_buffer, f'weights_offset_out_{idx}'),
                )

            print('Register weights offsets. OUT: {0:d}, K: {1:d}, IN: {2:d}, T: {3}'.format(
                out_channels, weights_offset_rank, in_channels, self.weights_domain_modulation_parametrization
            ))
        elif regex.match(_train_out_pattern, self.weights_domain_modulation_parametrization):
            [(weights_offset_rank, terms_num)] = regex.findall(
                _train_out_pattern, self.weights_domain_modulation_parametrization
            )
            weights_offset_rank, terms_num = int(weights_offset_rank), int(terms_num)

            self.weights_offset_in = []
            self.weights_offset_out = []
            for idx in range(terms_num):
                self.register_buffer(
                    f'weights_offset_in_{idx}',
                    torch.randn(weights_offset_rank, in_channels)
                )
                self.register_parameter(
                    f'weights_offset_out_{idx}',
                    torch.nn.Parameter(torch.zeros(out_channels, weights_offset_rank))
                )

                self.weights_offset_in.append(
                    (self.get_buffer, f'weights_offset_in_{idx}'),
                )
                self.weights_offset_out.append(
                    (self.get_parameter, f'weights_offset_out_{idx}'),
                )

            print('Register weights offsets. OUT: {0:d}, K: {1:d}, IN: {2:d}, T: {3}'.format(
                out_channels, weights_offset_rank, in_channels, self.weights_domain_modulation_parametrization
            ))
        else:
            raise ValueError(
                f'Unknown type of domain modulation: {self.weights_domain_modulation_parametrization}'
            )

        for getter, name in self.weights_offset_in:
            tensor = getter(name)
            if torch.linalg.norm(tensor.data).item() > 1e-6:
                tensor.data /= torch.linalg.norm(
                    tensor.data, dim=1, keepdim=True
                )

        for getter, name in self.weights_offset_out:
            tensor = getter(name)
            if torch.linalg.norm(tensor.data).item() > 1e-6:
                tensor.data /= torch.linalg.norm(
                    tensor.data, dim=0, keepdim=True
                )
    else:
        raise ValueError(f'Unknown type of domain modulation: {self.weights_domain_modulation_parametrization}')

    if self.weights_domain_modulation_parametrization in out_in_decomposition_parametrizations:
        self.register_buffer('ones_weights', self.weight.new_ones(out_channels, in_channels, 1, 1))
    else:
        print('Register weights offsets of shape: {0}, T: {1}'.format(
            list(self.weights_offset.shape), self.weights_domain_modulation_parametrization
        ))
        self.register_buffer('ones_weights', torch.ones_like(self.weights_offset))


def register_modulation(
        self: "Union[SynthesisLayer, ToRGBLayer]", in_channels, w_dim,
        use_domain_modulation, domain_modulation_parametrization, generator_requires_grad_parts
):
    """Create necessary parameters and buffers for domain adaptation using Style Space and Weights offsets

    Args:
        self:
        in_channels (int): S space dimension for a given layer
        w_dim (int): W space dimension for a given layer
        use_domain_modulation (bool): Whether to use Style Space or Weights offsets
        domain_modulation_parametrization (str): Comma separated list of selected offsets
            Should contain at most one type of Style Space Offsets and at most one type of Weights Offsets
        generator_requires_grad_parts (List[str]):
    """
    self.use_domain_modulation = use_domain_modulation
    (
        self.domain_modulation_parametrization,
        self.weights_domain_modulation_parametrization,
        self.affine.affine_weights_domain_modulation_parametrization
    ) = split_parameterization(
        use_domain_modulation, domain_modulation_parametrization
    )

    # Register parameters for domain modulation
    if self.domain_modulation_parametrization is not None:
        register_domain_modulation(self, in_channels, w_dim)

    # Register parameters for weights modulation
    if self.is_rgb:
        add_modulation = (
            'all' in generator_requires_grad_parts or
            'tRGB_weights_offset' in generator_requires_grad_parts or
            'tRGB_weights_offset.b{0:d}'.format(self.resolution) in generator_requires_grad_parts
        )
    else:
        add_modulation = (
            'all' in generator_requires_grad_parts or
            'synt_weights_offset' in generator_requires_grad_parts or
            'synt_weights_offset.b{0:d}'.format(self.resolution) in generator_requires_grad_parts
        )

    if add_modulation and self.weights_domain_modulation_parametrization is not None:
        register_weights_domain_modulation(self)
    else:
        self.weights_domain_modulation_parametrization = None

    if self.affine.affine_weights_domain_modulation_parametrization is not None:
        register_affine_weights_domain_modulation(self.affine)


def w_to_s(self, w, weight_gain, style_space_modifications=None, style_space_modifications_first=False, saved_styles=None):
    """Apply Affine layer, Style Space modifications and Style Space offsets

    Args:
        self (torch.nn.Module): Generator model
        w (torch.Tensor): W Space latents
        weight_gain (float):
        style_space_modifications (List[Tuple[Tuple[int, int], float, float]], optional): 
            List of modifications. Defaults to None.
        style_space_modifications_first (bool, optional): 
            Whether to apply Style Space modification before Style Space offsets. Defaults to False.
        saved_styles (defaultdict(dict), optional): 
            Buffer to save intermediate Style Space latents. Defaults to None.
    Returns:
        torch.Tensor: Style Space latents
    """
    use_w_space_modulation = self.domain_modulation_parametrization in ['multiplicative_w_space', 'additive_w_space']

    # Perform domain modulation in W space
    if self.use_domain_modulation and self.domain_modulation_parametrization is not None and use_w_space_modulation:
        if self.domain_modulation_parametrization == 'multiplicative_w_space':
            w = (self.ones + self.offset * self.offset_mask) * w
        elif self.domain_modulation_parametrization == 'additive_w_space':
            w = w + self.offset * self.offset_mask
        else:
            raise ValueError(f'Unknown type of domain modulation in W+ space: {self.domain_modulation_parametrization}')

    # Convert weights from W to S space
    styles = self.affine(w) * weight_gain
    if saved_styles is not None:
        saved_styles['initial'][self.layer_idx] = styles.detach().cpu()

    # Get StyleSpace semantic modifications vector
    modifications, offset_factors = get_style_space_modifications(
        self.layer_idx, styles, style_space_modifications
    )
    # Apply StyleSpace modifications
    if style_space_modifications is not None and style_space_modifications_first:
        styles = styles + modifications

    # Perform domain modulation in S space
    if self.use_domain_modulation and self.domain_modulation_parametrization is not None and not use_w_space_modulation:
        offset_mask = self.offset_mask if style_space_modifications is None else self.offset_mask * offset_factors

        if self.domain_modulation_parametrization == 'multiplicative':
            styles = (self.ones + self.offset * offset_mask) * styles
        elif self.domain_modulation_parametrization == 'additive':
            styles = styles + self.offset * offset_mask
        else:
            raise ValueError(f'Unknown type of domain modulation in S space: {self.domain_modulation_parametrization}')

    # Apply StyleSpace modifications
    if style_space_modifications is not None and not style_space_modifications_first:
        styles = styles + modifications

    if saved_styles is not None:
        saved_styles['final'][self.layer_idx] = styles.detach().cpu()

    return styles


def weight_to_weight(self, weight):
    """Apply Weights offsets

    Args:
        self (torch.nn.Module):
        weight (torch.Tensor): domain modulation weights

    Returns:
        torch.Tensor: shifted domain modulation weights
    """
    if hasattr(self, 'affine_weights_domain_modulation_parametrization') and self.affine_weights_domain_modulation_parametrization is not None:
        if self.affine_weights_domain_modulation_parametrization in affine_out_in_decomposition_parametrizations:
            if self.affine_weights_domain_modulation_parametrization in ['affine_out+in', 'affine_out+in_additive']:
                weights_offset = (self.weights_offset_out + self.weights_offset_in)
            else:
                weights_offset = sum(
                    lgetter(lname) @ rgetter(rname) for (lgetter, lname), (rgetter, rname) in
                    zip(self.weights_offset_out, self.weights_offset_in)
                ) / len(self.weights_offset_out)
        else:
            weights_offset = self.weights_offset
        if self.affine_weights_domain_modulation_parametrization.endswith('additive'):
            return weight + weights_offset
        else:
            return (self.ones_weights + weights_offset) * weight

    if hasattr(self, 'weights_domain_modulation_parametrization') and self.weights_domain_modulation_parametrization is not None:
        if self.weights_domain_modulation_parametrization in out_in_decomposition_parametrizations:
            if self.weights_domain_modulation_parametrization in ['out+in', 'out+in_additive']:
                weights_offset = (self.weights_offset_out + self.weights_offset_in).unsqueeze(2).unsqueeze(3)
            else:
                weights_offset = sum(
                    lgetter(lname) @ rgetter(rname) for (lgetter, lname), (rgetter, rname) in
                    zip(self.weights_offset_out, self.weights_offset_in)
                ).unsqueeze(2).unsqueeze(3) / len(self.weights_offset_out)
        else:
            weights_offset = self.weights_offset
        if self.weights_domain_modulation_parametrization.endswith('additive'):
            return weight + weights_offset
        else:
            return (self.ones_weights + weights_offset) * weight
    return weight


# ----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


# ----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
        x,  # Input tensor of shape [batch_size, in_channels, in_height, in_width].
        weight,  # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
        styles,  # Modulation coefficients of shape [batch_size, in_channels].
        noise=None,  # Optional noise tensor to add to the output activations.
        up=1,  # Integer upsampling factor.
        down=1,  # Integer downsampling factor.
        padding=0,  # Padding with respect to the upsampled image.
        # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
        resample_filter=None,
        demodulate=True,  # Apply weight demodulation?
        flip_weight=True,  # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
        fused_modconv=True,  # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw])  # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels])  # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (
            1 / np.sqrt(in_channels * kh * kw) /
            weight.norm(float('inf'), dim=[1, 2, 3], keepdim=True)
        )  # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(
            x=x, w=weight.to(x.dtype), f=resample_filter,
            up=up, down=down, padding=padding, flip_weight=flip_weight
        )
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings():  # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(
        x=x, w=w.to(x.dtype), f=resample_filter, 
        up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight
    )
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x


# ----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(
        self,
        in_features,  # Number of input features.
        out_features,  # Number of output features.
        bias=True,  # Apply additive bias before the activation function?
        activation='linear',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=1,  # Learning rate multiplier.
        bias_init=0,  # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = weight_to_weight(self, self.weight.to(x.dtype)) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x


# ----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        kernel_size,  # Width and height of the convolution kernel.
        bias=True,  # Apply additive bias before the activation function?
        activation='linear',  # Activation function: 'relu', 'lrelu', etc.
        up=1,  # Integer upsampling factor.
        down=1,  # Integer downsampling factor.
        resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.
        channels_last=False,  # Expect the input to have memory_format=channels_last?
        trainable=True,  # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1)  # slightly faster
        x = conv2d_resample.conv2d_resample(
            x=x, w=w.to(x.dtype), f=self.resample_filter,
            up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


# ----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,  # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,  # Intermediate latent (W) dimensionality.
        num_ws,  # Number of intermediate latents to output, None = do not broadcast.
        num_layers=8,  # Number of mapping layers.
        embed_features=None,  # Label embedding dimensionality, None = same as w_dim.
        layer_features=None,  # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=0.01,  # Learning rate multiplier for the mapping layers.
        w_avg_beta=0.995,  # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x


# ----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        w_dim,  # Intermediate latent (W) dimensionality.
        resolution,  # Resolution of this layer.
        kernel_size=3,  # Convolution kernel size.
        up=1,  # Integer upsampling factor.
        use_noise=True,  # Enable noise input?
        activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last=False,  # Use channels_last format for the weights?
        use_domain_modulation=False,
        domain_modulation_parametrization='multiplicative',
        generator_requires_grad_parts=('all',),
    ):
        super().__init__()
        self.is_rgb = False
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
        self.weight = torch.nn.Parameter(
            torch.randn(
                [out_channels, in_channels, kernel_size, kernel_size]
            ).to(memory_format=memory_format)
        )
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

        register_modulation(
            self, in_channels, w_dim,
            use_domain_modulation, domain_modulation_parametrization, generator_requires_grad_parts
        )

        self.layer_idx = None

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1, **kwargs):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution])

        styles = w_to_s(self, w=w, weight_gain=1.0, **kwargs)
        weight = weight_to_weight(self, self.weight)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn(
                [x.shape[0], 1, self.resolution, self.resolution], device=x.device
            ) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1)  # slightly faster
        x = modulated_conv2d(
            x=x, weight=weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight,
            fused_modconv=fused_modconv
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


# ----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(
            self, in_channels, out_channels, w_dim, resolution, kernel_size=1, conv_clamp=None, channels_last=False,
            use_domain_modulation=False, domain_modulation_parametrization='multiplicative',
            generator_requires_grad_parts=('all',), activation=None, use_noise=None
    ):
        # use_noise, activation are dummy arguments for legacy loading

        super().__init__()
        self.is_rgb = True
        self.resolution = resolution
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(
            torch.randn(
                [out_channels, in_channels, kernel_size, kernel_size]
            ).to(memory_format=memory_format)
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

        register_modulation(
            self, in_channels, w_dim,
            use_domain_modulation, domain_modulation_parametrization, generator_requires_grad_parts
        )

        self.layer_idx = None

    def forward(self, x, w, fused_modconv=True, noise_mode='random', **kwargs):
        styles = w_to_s(self, w=w, weight_gain=self.weight_gain, **kwargs)
        weight = weight_to_weight(self, self.weight)

        x = modulated_conv2d(x=x, weight=weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x


# ----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels, 0 = first block.
        out_channels,  # Number of output channels.
        w_dim,  # Intermediate latent (W) dimensionality.
        resolution,  # Resolution of this block.
        img_channels,  # Number of output color channels.
        is_last,  # Is this the last block?
        architecture='skip',  # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16=False,  # Use FP16 for this block?
        fp16_channels_last=False,  # Use channels-last memory format with FP16?
        **layer_kwargs,  # Arguments for SynthesisLayer.
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
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(
                in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2, resample_filter=resample_filter,
                conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs
            )
            self.num_conv += 1

        self.conv1 = SynthesisLayer(
            out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs
        )
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(
                out_channels, img_channels, w_dim=w_dim, resolution=resolution,
                conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs
            )
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(
                in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last
            )

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings():  # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img


# ----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(
        self,
        w_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # Output image resolution.
        img_channels,  # Number of color channels.
        channel_base=32768,  # Overall multiplier for the number of channels.
        channel_max=512,  # Maximum number of channels in any layer.
        num_fp16_res=0,  # Use FP16 for the N highest resolutions.
        **block_kwargs,  # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(
                in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16,
                **block_kwargs
            )
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
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
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img


# ----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality.
        c_dim,  # Conditioning label (C) dimensionality.
        w_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # Output resolution.
        img_channels,  # Number of output color channels.
        mapping_kwargs={},  # Arguments for MappingNetwork.
        synthesis_kwargs={},  # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(
            w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs
        )
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

        # Set layer numbers for correct Style Space modifications
        current_layer_idx = 0
        for mname, module in self.named_modules():
            if regex.match('synthesis.b[0-9]*.conv[0-1]$', mname) or regex.match('synthesis.b[0-9]*.torgb$', mname):
                module.layer_idx = current_layer_idx
                current_layer_idx += 1

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, **synthesis_kwargs)
        return img


# ----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels, 0 = first block.
        tmp_channels,  # Number of intermediate channels.
        out_channels,  # Number of output channels.
        resolution,  # Resolution of this block.
        img_channels,  # Number of input color channels.
        first_layer_idx,  # Index of the first layer.
        architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
        activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16=False,  # Use FP16 for this block?
        fp16_channels_last=False,  # Use channels-last memory format with FP16?
        freeze_layers=0,  # Freeze-D: Number of layers to freeze.
        **_
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0

        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable

        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(
                img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp,
                channels_last=self.channels_last
            )

        self.conv0 = Conv2dLayer(
            tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp,
            channels_last=self.channels_last
        )

        self.conv1 = Conv2dLayer(
            tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp,
            channels_last=self.channels_last
        )

        if architecture == 'resnet':
            self.skip = Conv2dLayer(
                tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter,
                channels_last=self.channels_last
            )

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img


# ----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings():  # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(
            G, -1, F, c, H, W
        )  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.
        return x


# ----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        cmap_dim,  # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,  # Resolution of this block.
        img_channels,  # Number of input color channels.
        architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels=1,  # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        **_
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(
            group_size=mbstd_group_size, num_channels=mbstd_num_channels
        ) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(
            in_channels + mbstd_num_channels, in_channels, kernel_size=3,
            activation=activation, conv_clamp=conv_clamp
        )
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])  # [NCHW]
        _ = force_fp32  # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x


# ----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(
        self,
        c_dim,  # Conditioning label (C) dimensionality.
        img_resolution,  # Input resolution.
        img_channels,  # Number of input color channels.
        architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
        channel_base=32768,  # Overall multiplier for the number of channels.
        channel_max=512,  # Maximum number of channels in any layer.
        num_fp16_res=0,  # Use FP16 for the N highest resolutions.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.
        block_kwargs={},  # Arguments for DiscriminatorBlock.
        mapping_kwargs={},  # Arguments for MappingNetwork.
        epilogue_kwargs={},  # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(
                in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs,
                **common_kwargs
            )
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(
                z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None,
                **mapping_kwargs
            )
        self.b4 = DiscriminatorEpilogue(
            channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs,
            **common_kwargs
        )

    def forward(self, img, c, **block_kwargs):
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x


# ----------------------------------------------------------------------------
