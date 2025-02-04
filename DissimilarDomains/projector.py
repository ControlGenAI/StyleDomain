# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import os
import copy
import glob
from time import perf_counter

import tqdm
import click
import imageio
import numpy as np
import PIL.Image

import torch
import torch.utils.data
import torch.nn.functional as F

import dnnlib
import legacy


def generate_image(generator, latent, space, noise_mode='const', truncation_psi=1.0):
    if space == 'w':
        return generator.synthesis(latent, noise_mode=noise_mode)
    elif space == 'z':
        return generator(latent, [], noise_mode=noise_mode, truncation_psi=truncation_psi)


def project(
        G,
        target: torch.Tensor,  # [C, H, W] and dynamic range [0,255], W & H must match G output resolution
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.1,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        verbose=False,
        space='w',
        truncation_psi=1.0,
        device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)  # type: ignore

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    latent_std, latent_opt, latent_out = None, None, None
    if space == 'w':
        # Compute w stats.
        logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')

        z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None, truncation_psi=truncation_psi)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
        latent_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
        latent_std = (np.sum((w_samples - latent_avg) ** 2) / w_avg_samples) ** 0.5

        latent_opt = torch.tensor(
            latent_avg, dtype=torch.float32, device=device, requires_grad=True
        )  # pylint: disable=not-callable
    elif space == 'z':
        latent_std = 1

        latent_opt = torch.zeros(
            [1, G.z_dim], dtype=torch.float32, device=device, requires_grad=True
        )  # pylint: disable=not-callable

    latent_out = torch.zeros([num_steps] + list(latent_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([latent_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in (pbar := tqdm.tqdm(range(num_steps), total=num_steps, leave=True, disable=not verbose)):
        # Learning rate schedule.
        t = step / num_steps
        noise_scale = latent_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from latent_opt.
        latent_noise = torch.randn_like(latent_opt) * noise_scale
        latents = latent_opt + latent_noise
        if space == 'w':
            latents = latents.repeat([1, G.mapping.num_ws, 1])
        synth_images = generate_image(
            G, latents, space, noise_mode='const', truncation_psi=truncation_psi
        )

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        pbar.set_description(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected latent for each optimization step.
        latent_out[step] = latent_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    if space == 'w':
        latent_out = latent_out.repeat([1, G.mapping.num_ws, 1])
    return latent_out


class BunchOfImagesDataset(torch.utils.data.Dataset):
    def __init__(self, target_path, resolution, extensions=('.png', '.jpg', '.jpeg')):
        self.resolution = resolution
        if os.path.isdir(target_path):
            self.files = glob.glob(target_path + '/**', recursive=True)
            self.files = sorted([file for file in self.files if file.endswith(extensions)])
        else:
            self.files = [target_path]

    def __getitem__(self, idx):
        target_path = self.files[idx]
        image_pil = PIL.Image.open(target_path).convert('RGB')
        w, h = image_pil.size
        s = min(w, h)
        image_pil = image_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        image_pil = image_pil.resize((self.resolution, self.resolution), PIL.Image.LANCZOS)
        # noinspection PyTypeChecker
        image_array = np.array(image_pil, dtype=np.uint8)
        return image_pil, image_array

    def __len__(self):
        return len(self.files)

    @staticmethod
    def collate(batch):
        images_pils, images_arrays = list(zip(*batch))
        return images_pils, np.stack(images_arrays)


# ----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--num-steps', help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed', help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video', help='Save an mp4 video of optimization progress', is_flag=True, show_default=True)
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR')
@click.option('--space', help='Space to project', required=True, type=click.Choice(['w', 'z']))
@click.option('--truncation-psi', help='Truncation used for training and generation', type=float, default=1.0, show_default=True)
@click.option('--save-all-steps', help='Save all steps for all images', is_flag=True, show_default=True)
@click.option('--save-image', help='Save image for the end of optimization progress', is_flag=True, show_default=True)
@click.option('--save-n', help='Number of projected images to save', type=int, default=10, show_default=True)
@click.option('--gpu', help='GPU idx', type=int)
def run_projection(
        network_pkl: str,
        target_fname: str,
        outdir: str,
        save_video: bool,
        seed: int,
        num_steps: int,
        space: str,
        truncation_psi: float,
        save_all_steps: bool,
        save_image: bool,
        save_n: int,
        gpu: int
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    gpu = str(gpu)
    print(f'Using GPU with idx: {gpu}')
    if gpu is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    print(f'Cuda is available: {torch.cuda.is_available()}. N devices: {torch.cuda.device_count()}')

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda:0')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)  # type: ignore

    # Load target images.
    dataset = BunchOfImagesDataset(target_path=target_fname, resolution=G.img_resolution)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=dataset.collate
    )
    print(f'Using dataset {target_fname} of size: {len(dataset)}.')

    # Render debug output: optional video and projected image and latent vector.
    os.makedirs(outdir, exist_ok=True)

    all_projected = []
    all_projected_steps = []
    for batch_idx, (target_pils, target_uint8s) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        # Optimize projection.
        start_time = perf_counter()

        target_pil, target_uint8 = target_pils[0], target_uint8s[0]
        projected_steps = project(
            G,
            target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device),
            num_steps=num_steps,
            device=device,
            verbose=True,
            space=space,
            truncation_psi=truncation_psi
        )
        projected = projected_steps[-1]

        print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

        save_batch = (batch_idx % max(1, (len(dataset) // save_n))) == 0
        if save_batch and save_video:
            video = imageio.get_writer(f'{outdir}/{batch_idx}_proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
            print(f'Saving optimization progress video "{outdir}/{batch_idx}_proj.mp4"')
            for projected in projected_steps:
                synth_image = generate_image(
                    G, projected.unsqueeze(0), space, noise_mode='const', truncation_psi=truncation_psi
                )

                synth_image = (synth_image + 1) * (255 / 2)
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
            video.close()

        if save_image:
            target_pil.save(f'{outdir}/{batch_idx}_target.png')

            synth_image = generate_image(
                G, projected.unsqueeze(0), space, noise_mode='const', truncation_psi=truncation_psi
            )

            # Save final projected frame
            synth_image = (synth_image + 1) * (255 / 2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/{batch_idx}_proj.png')

        all_projected.append(projected.cpu().numpy())
        if save_all_steps:
            all_projected_steps.append(projected_steps.cpu().numpy())

        np.savez(
            f'{outdir}/{batch_idx}_projected_{space}.npz',
            **{
                f'{space}': projected.unsqueeze(0).cpu().numpy(),
                f'{space}_steps': projected_steps.cpu().numpy()
            }
        )

    np.savez(
        f'{outdir}/projected_{space}.npz',
        **{
            f'{space}': np.stack(all_projected),
            f'{space}_steps': np.stack(all_projected_steps) if save_all_steps else None
        }
    )


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
