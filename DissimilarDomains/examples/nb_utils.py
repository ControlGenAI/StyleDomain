import os
import copy
import json
import glob
import regex
from collections import defaultdict

import torch
import numpy as np

import dnnlib
import legacy
from torch_utils import misc


# noinspection PyPep8Naming
def create(G_kwargs, D_kwargs, common_kwargs, resume_pkl, device, verbose=False):
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if verbose:
        print(f'Resuming from "{resume_pkl}"')
    if resume_pkl is not None:
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=True)

    return G, D, G_ema


# noinspection PyPep8Naming
def load_checkpoint(exp_path, chkpt_idx, device):
    options_path = os.path.join(exp_path, 'training_options.json')
    with open(options_path, 'r', encoding='utf-8') as options_file:
        options = json.load(options_file)

    resume_pkl = os.path.join(exp_path, 'network-snapshot-{0:06d}.pkl'.format(chkpt_idx))

    common_kwargs = dict(
        c_dim=0, img_channels=3,
        img_resolution=options['training_set_kwargs']['resolution']
    )
    options['G_kwargs']['synthesis_kwargs']['generator_requires_grad_parts'] = options['requires_grad_parts']['Gboth']
    G, _, G_ema = create(
        options['G_kwargs'], options['D_kwargs'], common_kwargs, resume_pkl, device, verbose=False
    )

    base_options = copy.deepcopy(options)
    if 'use_domain_modulation' in base_options['G_kwargs']['synthesis_kwargs']:
        del base_options['G_kwargs']['synthesis_kwargs']['use_domain_modulation']
    if 'domain_modulation_parametrization' in base_options['G_kwargs']['synthesis_kwargs']:
        del base_options['G_kwargs']['synthesis_kwargs']['domain_modulation_parametrization']
    G_base, _, G_ema_base = create(
        base_options['G_kwargs'], base_options['D_kwargs'], common_kwargs, base_options['resume_pkl'], device
    )

    metrics = defaultdict(list)
    for metrics_path in glob.glob(os.path.join(exp_path, 'metric-*.jsonl')):
        with open(metrics_path, 'r', encoding='utf-8') as metric_file:
            metric_name, *_ = regex.findall('metric-(.*).jsonl', os.path.basename(metrics_path))
            for line in metric_file:
                entity = json.loads(line)
                metric_name = entity['metric']
                metric_value = entity['results'][metric_name]

                chkpt_idx, *_ = regex.findall('network-snapshot-(.*).pkl', entity['snapshot_pkl'])
                chkpt_idx = int(chkpt_idx)

                metrics[metric_name].append((chkpt_idx, metric_value))

    metrics = {key: sorted(value) for key, value in metrics.items()}

    return G, G_ema, G_base, G_ema_base, options, metrics


# noinspection PyPep8Naming
@torch.no_grad()
def generate_images(
        G, grid_size=(7, 4), seed=42, batch_size=8, device=None,
        *, target_zs=None, target_ws=None, **kwargs
):
    """
    :param torch.nn.Module G: Generator to perform inference
    :param Union[int, List[int]] grid_size: Number of images to generate.
        Could be a list that defines a rectangular grid
    :param int seed: Random seed
    :param int batch_size: Inference batch size
    :param Optional[torch.Device] device: Computing device
    :param Optional[torch.Tensor] target_zs: Tensor of shape [batch_size, G.mapping.z_dim].
        During inference, those latents in Z space will be used instead of randomly generated ones
    :param Optional[torch.Tensor] target_ws: Tensor of shape [*, G.mapping.num_ws, G.mapping.w_dim].
        During inference, those latents in W+ space will be used instead of the Mapping Network output
    """
    assert not ((target_zs is not None) and (target_ws is not None))

    G.to(device)

    def _G_call(z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        if target_zs is not None:
            z = target_zs

        if target_ws is not None:
            ws = target_ws
        else:
            ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

        img = G.synthesis(ws, force_fp32=True, **synthesis_kwargs)
        return img, ws

    # Generate random latent vectors in Z space
    grid_z = np.random.default_rng(seed).normal(size=[np.prod(grid_size), G.z_dim])
    grid_c = np.empty([np.prod(grid_size), 0])

    grid_z = torch.from_numpy(grid_z).to(device).split(batch_size)
    grid_c = torch.from_numpy(grid_c).to(device).split(batch_size)

    images, wss = [], []
    for batch_z, batch_c in zip(grid_z, grid_c):
        batch_images, batch_ws = _G_call(z=batch_z, c=batch_c, noise_mode='const', **kwargs)
        images.append(batch_images.cpu())
        wss.append(batch_ws.cpu())
    images = torch.cat(images).numpy()
    wss = torch.cat(wss).numpy()

    # Normalize image to [0, 255] integer segment
    lo, hi = [-1, 1]
    images = (images - lo) * (255 / (hi - lo))
    images = np.rint(images).clip(0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)

    G.to('cpu')

    return images, wss


def prepare_axes(axes):
    for ax in np.array(axes).reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])
        for key, spine in ax.spines.items():
            spine.set_visible(False)
