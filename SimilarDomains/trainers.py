import os
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip
import torchvision.transforms as transforms
import torch.distributions as dis
import typing as tp

from pathlib import Path

from core.utils.text_templates import imagenet_templates
from core.utils.train_log import StreamingMeans, TimeLog, Timer
from core.utils.loggers import LoggingManager
from core.utils.class_registry import ClassRegistry
from core.utils.common import (
    mixing_noise, format_device, compose_text_with_templates, load_clip,
    read_domain_list, read_style_images_list, determine_opt_layers,
    get_stylegan_conv_dimensions, DataParallelPassthrough, get_trainable_model_state,
    w_idx_to_style_idx
)

from core.utils.example_utils import read_img, project_e4e

from core.loss import DirectLoss
from core.utils.math_utils import (
    resample_single_vector, resample_batch_vectors, convex_hull,
    resample_batch_templated_embeddings, convex_hull_small
)

from core.utils.image_utils import BicubicDownSample, t2im, construct_paper_image_grid, crop_augmentation
from core.parametrizations import BaseParametrization
from core.mappers import mapper_registry
from core.utils.II2S import II2S
from core.uda_models import uda_models
from core.dataset import ImagesDataset
from tqdm import trange


trainer_registry = ClassRegistry()


def invert_ii2s(image_path, ii2s_options, align_input, device):
    single_image_dataset = ImagesDataset(
        opts=ii2s_options,
        image_path=image_path,
        align_input=align_input
    )
    
    ii2s = II2S(ii2s_options)
    
    image_info = single_image_dataset[0]

    image_full_res = image_info['image_high_res_torch'].unsqueeze(0).to(device)
    image_resized = image_info['image_low_res_torch'].unsqueeze(0).to(device)
    
    latents = ii2s.invert_image(
        image_full_res,
        image_resized
    )
    
    return latents


class BaseDomainAdaptationTrainer:
    def __init__(self, config):
        # common
        self.config = config
        self.trainable = None
        self.source_generator = None

        self.current_step = 0
        self.optimizer = None
        self.loss_function = None
        self.batch_generators = None

        self.zs_for_logging = None

        self.reference_embeddings = {}

        # processed in multiple_domain trainer
        self.domain_embeddings = None
        self.desc_to_embeddings = None

        self.global_metrics = {}

    def _setup_base(self):
        self._setup_device()
        self._setup_logger()
        self._setup_batch_generators()
        self._setup_source_generator()
        self._setup_loss()
        
        if "low_memory" in self.config.logging and self.config.logging.low_memory:
            self._initial_logging_for_low_memory()
        else:
            self._initial_logging()

    def _setup_device(self):
        chosen_device = self.config.training["device"].lower()
        device = format_device(chosen_device)
        self.device = torch.device(device)

    def _setup_source_generator(self):
        self.source_generator = uda_models[self.config.training.generator](
            **self.config.generator_args[self.config.training.generator]
        )
        self.source_generator.patch_layers(self.config.training.patch_key)
        self.source_generator.freeze_layers()
        self.source_generator.to(self.device)

    def _setup_loss(self):
        self.loss_function = DirectLoss(self.config.optimization_setup)
        self.has_clip_loss = len(self.loss_function.clip.funcs) > 0
        
    def _setup_logger(self):
        self.logger = LoggingManager(self.config)

    def _setup_batch_generators(self):
        self.batch_generators = {}

        for visual_encoder in self.config.optimization_setup.visual_encoders:
            self.batch_generators[visual_encoder] = (
                load_clip(visual_encoder, device=self.config.training.device)
            )

        self.reference_embeddings = {k: {} for k in self.batch_generators}

    @torch.no_grad()
    def _initial_logging(self):
        self.zs_for_logging = [
            mixing_noise(16, 512, 0, self.config.training.device)
            for _ in range(self.config.logging.num_grid_outputs)
        ]

        for idx, z in enumerate(self.zs_for_logging):
            images = self.forward_source(z, truncation=self.config.logging.truncation)
            self.logger.log_images(0, {f"src_domain_grids/{idx}": construct_paper_image_grid(images)})
            
    @torch.no_grad()
    def _initial_logging_for_low_memory(self):
        self.zs_for_logging = [
            mixing_noise(2, 512, 0, self.config.training.device)
            for _ in range(self.config.logging.num_grid_outputs)
        ]

        for idx, z in enumerate(self.zs_for_logging):
            images = self.forward_source(z, truncation=self.config.logging.truncation)
            images = torch.cat([images[0], images[1]], dim=2)
            self.logger.log_images(0, {f"src_domain_grids/{idx}": t2im(images)})
    
    def _setup_optimizer(self):
        if self.config.training.patch_key == "original":
            g_reg_every = self.config.optimization_setup.g_reg_every
            lr = self.config.optimization_setup.optimizer.lr

            g_reg_ratio = g_reg_every / (g_reg_every + 1)
            betas = self.config.optimization_setup.optimizer.betas

            self.optimizer = torch.optim.Adam(
                self.trainable.parameters(),
                lr=lr * g_reg_ratio,
                betas=(betas[0] ** g_reg_ratio, betas[1] ** g_reg_ratio),
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.trainable.parameters(), **self.config.optimization_setup.optimizer
            )
        
        if self.config.optimization_setup.get('scheduler', False):
            lr = self.config.optimization_setup.optimizer.lr
            steps = self.config.optimization_setup.scheduler.n_steps
            start_lr = self.config.optimization_setup.scheduler.start_lr
            alpha = lr - start_lr
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda i: start_lr/lr + min(1, i / steps) * alpha / lr
            )
        
    # @classmethod
    # def from_ckpt(cls, ckpt_path):
    #     m = cls(ckpt['config'])
    #     m._setup_base()
    #     return m

    def start_from_checkpoint(self):
        step = 0
        if self.config.checkpointing.start_from:
            state_dict = torch.load(self.config.checkpointing.start_from, map_location='cpu')
            step = state_dict['step']
            self.trainable.load_state_dict(state_dict['trainable'])
            self.optimizer.load_state_dict(state_dict['trainable_optimizer'])
            print('starting from step {}'.format(step))
        # TODO: python main.py --ckpt_path ./.... -> Trainer.from_ckpt()
        return step

    def get_checkpoint(self):
        state_dict = {
            "step": self.current_step,
            "trainable": self.trainable.state_dict(),
            "trainable_optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }
        return state_dict

    # TODO: refactor checkpoint
    def make_checkpoint(self):
        if not self.config.checkpointing.is_on:
            return

        ckpt = self.get_checkpoint()
        torch.save(ckpt, os.path.join(self.logger.checkpoint_dir, "checkpoint.pt"))

    def save_models(self):
        models_dict = get_trainable_model_state(
            self.config, self.trainable.state_dict()
        )
        
        models_dict.update(self.ckpt_info())
        torch.save(models_dict, str(
            Path(self.logger.models_dir) / f"models_{self.current_step}.pt"
        ))
    
    def ckpt_info(self):
        return {}
    
    def all_to_device(self, device):
        self.source_generator.to(device)
        self.trainable.to(device)
        self.loss_function.to(device)

    def train_loop(self):
        self.all_to_device(self.device)
        
        training_time_log = TimeLog(
            self.logger, self.config.training.iter_num + 1, event="training"
        )

        recovered_step = self.start_from_checkpoint()
        iter_info = StreamingMeans()

        for self.current_step in range(recovered_step, self.config.training.iter_num + 1, 1):
            with Timer(iter_info, "train_iter"):
                self.train_step(iter_info)

            if self.current_step % self.config.checkpointing.step_backup == 0:
                self.make_checkpoint()

            if (self.current_step + 1) % self.config.exp.step_save == 0:
                self.save_models()

            if self.current_step % self.config.logging.log_images == 0:
                with Timer(iter_info, "log_images"):
                    if "low_memory" in self.config.logging and self.config.logging.low_memory:
                        self.log_images_for_low_memory()
                    else:
                        self.log_images()
            
            if self.current_step % self.config.logging.log_every == 0:
                self.logger.log_values(
                    self.current_step, self.config.training.iter_num, iter_info
                )
                iter_info.clear()
                training_time_log.now(self.current_step)
            
        training_time_log.end()
        wandb.finish()
    
    @torch.no_grad()
    def encode_text(self, model, text, templates):
        text = compose_text_with_templates(text, templates=templates)
        tokens = clip.tokenize(text).to(self.config.training.device)
        text_features = model.encode_text(tokens).detach()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def clip_encode_image(self, model, image, preprocess, norm=True):
        image_features = model.encode_image(preprocess(image))
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)
        
        return image_features
    
    @torch.no_grad()
    def _mean_clip_image_src_embedding(self, model, preprocess, num=1000):
        mean_clip_value = torch.zeros(512, dtype=torch.float16, device=self.device)
                    
        for i in trange(num // self.config.training.batch_size):
            
            sample_z = mixing_noise(
                self.config.training.batch_size, 512, 
                self.config.training.mixing_noise, 
                self.device
            )
            sampled_src = self.forward_source(
                sample_z, truncation=self.config.emb.online_truncation
            )
            mean_clip_value += self.clip_encode_image(model, sampled_src, preprocess, norm=False).sum(dim=0)
        
        return mean_clip_value / num
    
    def _process_source_embeddings(self):
        self.src_embeddings = {}
        
        for clip_key, (m, p) in self.batch_generators.items():
        
            if self.config.emb.type == 'mean':
                src_dir = Path(self.config.emb.src_emb_dir)
                src_domain = Path(self.config.generator_args['checkpoint_path']).stem.split('-')[1]
                im_size = self.source_generator.generator.size
                src_emb_name = f"{src_domain}_{clip_key.split('/')[1]}.pkl"
                src_emb_path = src_dir / src_emb_name


                if src_emb_path.exists():
                    import pickle as pkl
                    with open(str(src_emb_path), 'rb') as f:
                        X = pkl.load(f)
                        X = np.array(X)
                    mean = torch.from_numpy(np.mean(X, axis=0)).float().to(self.device)
                    mean /= mean.clone().norm(dim=-1, keepdim=True)
                    self.src_embeddings[clip_key] = mean
                else:
                    raise ValueError(f'no mean embedding of Source domain in dir: {src_emb_path}')

            elif self.config.emb.type == 'online':
                self.src_embeddings[clip_key] = self._mean_clip_embedding(m, p, num=self.config.emb.num).unsqueeze(0)
            elif self.config.emb.type == 'projected_target':
                self.src_embeddings[clip_key] = self.clip_encode_image(m, self.style_image_inverted_A, p)
            elif self.config.emb.type == 'text':
                ...
            else:
                raise ValueError('Unknown emb type')
    
    def partial_trainable_model_freeze(self):
        if not hasattr(self.config.training, 'auto_layer_iters'):
            return
        
        if self.config.training.auto_layer_iters == 0:
            return

        train_layers = determine_opt_layers(
            self.source_generator,
            self.trainable,
            self.batch_generators['ViT-B/32'][0],
            self.config,
            self.config.training.target_class,
            self.config.training.auto_layer_iters,
            self.config.training.auto_layer_batch,
            self.config.training.auto_layer_k,
            device=self.device,
        )

        if not isinstance(train_layers, list):
            train_layers = [train_layers]

        self.trainable.freeze_layers()
        self.trainable.unfreeze_layers(train_layers)

    def train_step(self, iter_info):
        self.trainable.train()
        sample_z = mixing_noise(
            self.config.training.batch_size,
            512,
            self.config.training.mixing_noise,
            self.config.training.device,
        )

        self.partial_trainable_model_freeze()

        batch = self.calc_batch(sample_z)
        losses = self.loss_function(batch)
        
        with torch.autograd.set_detect_anomaly(True):
            self.optimizer.zero_grad()
            losses["total"].backward()
            self.optimizer.step()
            if hasattr(self, 'scheduler'):
                self.scheduler.step()

        iter_info.update({f"losses/{k}": v for k, v in losses.items()})

    def forward_trainable(self, latents, *args, **kwargs):
        raise NotImplementedError()

    @torch.no_grad()
    def forward_source(self, latents, **kwargs) -> torch.Tensor:
        sampled_images, _ = self.source_generator(latents, **kwargs)
        return sampled_images.detach()

    def calc_batch(self, sample_z):
        raise NotImplementedError()
    
    @torch.no_grad()
    def log_images(self):
        raise NotImplementedError()

    def to_multi_gpu(self):
        self.source_generator = DataParallelPassthrough(
            self.source_generator, 
            device_ids=self.config.exp.device_ids
        )
        self.trainable = DataParallelPassthrough(self.trainable, device_ids=self.config.exp.device_ids)

    def invert_image_ii2s(self, image_info, ii2s):
        image_full_res = image_info['image_high_res_torch'].unsqueeze(0).to(self.device)
        image_resized = image_info['image_low_res_torch'].unsqueeze(0).to(self.device)
            
        print(f"Shape inside invert_image_ii2s: image_full res: {image_full_res.shape}, resized :{image_resized.shape}")
        
        lam = str(int(ii2s.opts.p_norm_lambda * 1000))
        name = Path(image_info['image_name']).stem + f"_{lam}.npy"
        current_latents_path = self.logger.cached_latents_local_path / name

        if current_latents_path.exists():
            latents = np.load(str(current_latents_path))
            latents = torch.from_numpy(latents).to(self.config.training.device)
        else:
            latents = ii2s.invert_image(
                image_full_res,
                image_resized
            )

            print(f'''
            latents for {image_info['image_name']} cached in 
            {str(current_latents_path.resolve())}
            ''')

            np.save(str(current_latents_path), latents.detach().cpu().numpy())

        return latents


class SingleDomainAdaptationTrainer(BaseDomainAdaptationTrainer):
    def __init__(self, config):
        super().__init__(config)
    
    def _setup_trainable(self):
        if self.config.training.patch_key == 'original':
            self.trainable = uda_models[self.config.training.generator](
                **self.config.generator_args[self.config.training.generator]
            )
            trainable_layers = list(self.trainable.get_training_layers(
                phase=self.config.training.phase
            ))
            self.trainable.freeze_layers()
            self.trainable.unfreeze_layers(trainable_layers)
        elif self.config.training.get('adaptive', False):
            
            from stylespace_core.stylespace_models import AdaptiveModel
            
            self.trainable = AdaptiveModel(
                min_number_ch=self.config.training.adaptive_min_ch,
                cutout_ch=self.config.training.adaptive_cutout_rate,
                prune_method=self.config.training.adaptive_prune_method,
                prune_step=self.config.training.adaptive_prune_step,
            )
            
        else:
            self.trainable = BaseParametrization(
                self.config.training.patch_key,
                get_stylegan_conv_dimensions(self.source_generator.generator.size),
                no_coarse=self.config.training.get('no_coarse', False),
                no_medium=self.config.training.get('no_medium', False),
                no_fine=self.config.training.get('no_fine', False)
            )

        self.trainable.to(self.device)
    
    def forward_trainable(self, latents, **kwargs) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        if self.config.training.patch_key == "original":
            sampled_images, _ = self.trainable(
                latents, **kwargs
            )
            offsets = None
        elif self.config.training.get('adaptive', False):
            if not kwargs.get('is_s_code', False):
                s_code = self.source_generator.get_s_code(latents, **kwargs)
            
            s_code, offsets = self.trainable(s_code)
            sampled_images, _ = self.source_generator(
                s_code, truncation=kwargs.get('truncation', 1), is_s_code=True
            )
        else:
            offsets = self.trainable()
            sampled_images, _ = self.source_generator(
                latents, offsets=offsets, **kwargs
            )

        return sampled_images, offsets
    
    @torch.no_grad()
    def log_images(self):
        self.trainable.eval()
        dict_to_log = {}

        for idx, z in enumerate(self.zs_for_logging):
            sampled_images, _ = self.forward_trainable(z, truncation=self.config.logging.truncation)
            images = construct_paper_image_grid(sampled_images)
            dict_to_log.update({
                f"trg_domain_grids/{self.config.training.target_class}/{idx}": images
            })

        self.logger.log_images(self.current_step, dict_to_log)
        
    
@trainer_registry.add_to_registry("td_single")
class TextDrivenSingleDomainAdaptationTrainer(SingleDomainAdaptationTrainer):
    def __init__(self, config):
        super().__init__(config)
    
    def ckpt_info(self):
        return {
            'da_type': 'td',
        }
    
    def setup(self):
        self._setup_base()
        self._setup_trainable()
        self._setup_optimizer()
        self._setup_text_embeddings()

    def _setup_text_embeddings(self):
        for visual_encoder, (model, preprocess) in self.batch_generators.items():
            self.reference_embeddings[visual_encoder][self.config.training.source_class] = self.encode_text(
                model, self.config.training.source_class, imagenet_templates
            )
            self.reference_embeddings[visual_encoder][self.config.training.target_class] = self.encode_text(
                model, self.config.training.target_class, imagenet_templates
            )

    def calc_batch(self, sample_z):
        clip_data = {
            k: {} for k in self.batch_generators
        }
        
        frozen_img = self.forward_source(sample_z)
        trainable_img, offsets = self.forward_trainable(sample_z)
                
        for visual_encoder_key, (model, preprocess) in self.batch_generators.items():
            
            trg_encoded = self.clip_encode_image(model, trainable_img, preprocess)
            src_encoded = self.clip_encode_image(model, frozen_img, preprocess)
            
            clip_data[visual_encoder_key].update({
                'trg_encoded': trg_encoded,
                'src_encoded': src_encoded,
                'trg_domain_emb': (
                    self.reference_embeddings[visual_encoder_key][self.config.training.target_class].unsqueeze(0)
                ),
                'src_domain_emb': (
                    self.reference_embeddings[visual_encoder_key][self.config.training.source_class].unsqueeze(0)
                )
            })
        
        return {
            'clip_data': clip_data,
            'rec_data': None,
            'offsets': offsets
        }
    

@trainer_registry.add_to_registry("im2im_single")
class Image2ImageSingleDomainAdaptationTrainer(SingleDomainAdaptationTrainer):
    def __init__(self, config):
        super().__init__(config)

        self.style_image_latents = None
        self.style_image_full_res = None
        self.style_image_resized = None
        self.style_image_inverted_A = None
    
    def setup(self):
        self._setup_base()
        self._setup_trainable()
        self._setup_optimizer()

        self._setup_style_image()
        self._log_target_images()
        self._setup_src_embeddings()
    
    def ckpt_info(self):
        return {
            'da_type': 'im2im',
            'style_latents': self.style_image_latents[:, 7:, :].cpu()
        }
    
    def _setup_src_embeddings(self):
        for visual_encoder, (model, preprocess) in self.batch_generators.items():
            self.reference_embeddings[visual_encoder][self.config.training.source_class] = self.encode_text(
                model, self.config.training.source_class, ['A {}']
            )
    
    def _setup_style_image(self):
        im_path = self.config.training.target_class
        inversion_m = self.config.inversion.method
        
        root_latents = Path(self.config.inversion.latents_root)
        root_latents.mkdir(exist_ok=True)
        size = self.source_generator.generator.size
        size_pref = "" if size == 1024 else "_512"
        latent_dump_name = f"{Path(im_path).stem}_{inversion_m}{size_pref}.npy"
    
        
        current_latents_path = root_latents / latent_dump_name
        
        if inversion_m == 'e4e':
            self.style_image_latents = self._setup_style_e4e(current_latents_path)
        elif inversion_m == 'ii2s':
            self.style_image_latents = self._setup_style_ii2s(current_latents_path)
                
        self.bicubic = BicubicDownSample(self.source_generator.generator.size // 256)
        self.style_image_inverted_A = self.forward_source([self.style_image_latents], input_is_latent=True)
        
        from core.style_embed_options import II2S_s_opts
        single_image_dataset = ImagesDataset(
            opts=II2S_s_opts,
            image_path=self.config.training.target_class,
            align_input=self.config.inversion.align_style
        )

        image_info = single_image_dataset[0]

        self.style_image_full_res = image_info['image_high_res_torch'].unsqueeze(0).to(self.device)
        self.style_image_resized = image_info['image_low_res_torch'].unsqueeze(0).to(self.device)
                
    def _setup_style_ii2s(self, latents_path):
        from core.style_embed_options import II2S_s_opts
        II2S_s_opts.size = self.source_generator.generator.size
        II2S_s_opts.ckpt = self.config.generator_args.stylegan2.checkpoint_path
        
        
        if latents_path.exists():
            latents = np.load(str(latents_path))
            latents = torch.from_numpy(latents).to(self.device)
        else:
            latents = invert_ii2s(
                self.config.training.target_class, II2S_s_opts, 
                align_input=self.config.inversion.align_style, device=self.device
            )
            print(f'''
            latents for {self.config.training.target_class} cached in 
            {str(latents_path.resolve())}
            ''')

            np.save(str(latents_path), latents.detach().cpu().numpy())
        
        return latents
    
    def _setup_style_e4e(self, latents_path):
        if latents_path.exists():
            latents = np.load(str(latents_path))
            latents = torch.from_numpy(latents).to(self.device)
            return latents
        
        image = read_img(self.config.training.target_class, self.config.inversion.align_style)
        _, latents = project_e4e(
            image, self.config.inversion.model_path, device=self.device
        )

        print(f'''
        latents for {self.config.training.target_class} cached in 
        {str(latents_path.resolve())}
        ''')

        np.save(str(latents_path), latents.detach().cpu().numpy())
        return latents
    
    
    def _log_target_images(self):
        style_image_resized = t2im(self.style_image_resized.squeeze())
        st_im_inverted_A = t2im(self.style_image_inverted_A.squeeze())
        self.logger.log_images(
            0, {"style_image/orig": style_image_resized, "style_image/projected_A": st_im_inverted_A}
        )

    def calc_batch(self, sample_z):
        clip_data = {k: {} for k in self.batch_generators}

        frozen_img = self.forward_source(sample_z)
        trainable_img, offsets = self.forward_trainable(sample_z)
        style_image_inverted_B, _ = self.forward_trainable(
            [self.style_image_latents], input_is_latent=True
        )
        
        for visual_encoder_key, (model, preprocess) in self.batch_generators.items():
            
            trg_encoded = self.clip_encode_image(model, trainable_img, preprocess)
            src_encoded = self.clip_encode_image(model, frozen_img, preprocess)
            trg_domain_emb = self.clip_encode_image(model, self.style_image_full_res, preprocess)
            src_domain_emb = self.clip_encode_image(model, self.style_image_inverted_A, preprocess)
            # src_domain_emb = self.reference_embeddings[visual_encoder_key][self.config.training.source_class]
            st_inverted_B_emb = self.clip_encode_image(model, style_image_inverted_B, preprocess)
            st_orig_emb = self.clip_encode_image(model, self.style_image_full_res, preprocess)
            
            clip_data[visual_encoder_key].update({
                'trg_encoded': trg_encoded,
                'src_encoded': src_encoded,
                'trg_domain_emb': trg_domain_emb,
                'src_domain_emb': src_domain_emb,
                'trg_trainable_emb': st_inverted_B_emb,
                'trg_emb': trg_domain_emb
            })

        rec_data = {
            'style_inverted_B_256x256': self.bicubic(style_image_inverted_B),
            'style_image_256x256': self.style_image_resized,
            'style_inverted_B_1024x1024': style_image_inverted_B,
            'style_image_1024x1024': self.style_image_full_res,
        }

        return {
            'clip_data': clip_data,
            'rec_data': rec_data,
            'offsets': offsets
        }

    @torch.no_grad()
    def log_images(self):
        self.trainable.eval()
        dict_to_log = {}

        for idx, z in enumerate(self.zs_for_logging):
            w_styles = self.source_generator.style(z)
            sampled_imgs, _ = self.forward_trainable(z, truncation=self.config.logging.truncation)
            n_lat = self.source_generator.generator.n_latent
            tmp_latents = w_styles[0].unsqueeze(1).repeat(1, n_lat, 1)
            gen_mean = self.source_generator.mean_latent.unsqueeze(1).repeat(1, n_lat, 1)
            style_mixing_latents = self.config.logging.truncation * (tmp_latents - gen_mean) + gen_mean
            style_mixing_latents[:, 7:, :] = self.style_image_latents[:, 7:, :]

            style_mixing_imgs, _ = self.forward_trainable(
                [style_mixing_latents], input_is_latent=True, truncation=1
            )

            sampled_imgs = construct_paper_image_grid(sampled_imgs)
            style_mixing_imgs = construct_paper_image_grid(style_mixing_imgs)

            dict_to_log.update({
                f"trg_domain_grids/{Path(self.config.training.target_class).stem}/{idx}": sampled_imgs,
                f"trg_domain_grids_sm/{Path(self.config.training.target_class).stem}/{idx}": style_mixing_imgs,

            })

        rec_img, _ = self.forward_trainable(
            [self.style_image_latents],
            input_is_latent=True,
        )
        rec_img = t2im(rec_img.squeeze())
        dict_to_log.update({"style_image/projected_B": rec_img})
        self.logger.log_images(self.current_step, dict_to_log)
        
    @torch.no_grad()
    def log_images_for_low_memory(self):
        self.trainable.eval()
        dict_to_log = {}

        for idx, z in enumerate(self.zs_for_logging):
            w_styles = self.source_generator.style(z)
            sampled_imgs, _ = self.forward_trainable(z, truncation=self.config.logging.truncation)
            n_lat = self.source_generator.generator.n_latent
            tmp_latents = w_styles[0].unsqueeze(1).repeat(1, n_lat, 1)
            gen_mean = self.source_generator.mean_latent.unsqueeze(1).repeat(1, n_lat, 1)
            style_mixing_latents = self.config.logging.truncation * (tmp_latents - gen_mean) + gen_mean
            style_mixing_latents[:, 7:, :] = self.style_image_latents[:, 7:, :]

            style_mixing_imgs, _ = self.forward_trainable(
                [style_mixing_latents], input_is_latent=True, truncation=1
            )

            sampled_imgs = torch.cat([sampled_imgs[0], sampled_imgs[1]], dim=2)
            sampled_imgs = t2im(sampled_imgs)
            style_mixing_imgs = torch.cat([style_mixing_imgs[0], style_mixing_imgs[1]], dim=2)
            style_mixing_imgs = t2im(style_mixing_imgs)

            dict_to_log.update({
                f"trg_domain_grids/{Path(self.config.training.target_class).stem}/{idx}": sampled_imgs,
                f"trg_domain_grids_sm/{Path(self.config.training.target_class).stem}/{idx}": style_mixing_imgs,

            })

        rec_img, _ = self.forward_trainable(
            [self.style_image_latents],
            input_is_latent=True,
        )
        rec_img = t2im(rec_img.squeeze())
        dict_to_log.update({"style_image/projected_B": rec_img})
        self.logger.log_images(self.current_step, dict_to_log)
        
        
@trainer_registry.add_to_registry("im2im_JoJo")
class JoJoSingleDomainAdaptationTrainer(Image2ImageSingleDomainAdaptationTrainer):
    def __init__(self, config):
        super().__init__(config)

        self.style_image_latents = None
        self.style_image_full_res = None
        self.style_image_resized = None
        self.style_image_inverted_A = None
    
    def setup(self):
        self._setup_base()
        self._setup_trainable()
        self._setup_optimizer()

        self._setup_style_image()
        self._log_target_images()
        self._setup_jojo()
        
    def _setup_jojo(self):
        self.a = 1 - self.config.training.alpha
        if self.config.training.mix_stylespace:
            with torch.no_grad():
                self.style_image_latents = self.source_generator.get_s_code(
                    [self.style_image_latents], input_is_latent=True
                )
        else:
            self.style_image_latents = [self.style_image_latents]
        
        
        if self.config.training.preserve_color:
            self.id_swap = [9, 11, 15, 16, 17]
        else:
            self.id_swap = list(range(7, self.source_generator.generator.n_latent))
        
        if self.config.training.mix_stylespace:
            new_id_swap = []
            for w_idx in self.id_swap:
                new_id_swap.append(w_idx_to_style_idx[w_idx])
            
            self.id_swap = new_id_swap
        
    def calc_batch(self, sample_z):
        clip_data = {k: {} for k in self.batch_generators}
        
        try:
            rep = self.config.training.batch_size // self.style_image_latents[0].size(0)
        except:
            rep = self.config.training.batch_size // self.style_image_latents[0].size(0)
        
        if self.config.training.mix_stylespace:
            in_latent = [t.clone().repeat(rep, 1) for t in self.style_image_latents]
            
            with torch.no_grad():
                mean_s = self._get_stylespace(sample_z)

            in_latent = [self.a * t + (1 - self.a) * mean_s[i] if i in self.id_swap else t for i, t in enumerate(in_latent)]

        else:
            w = self._get_w(sample_z)[0]
            in_latent = self.style_image_latents[0].clone().repeat(rep, 1, 1)
            in_latent[:, self.id_swap] = self.a * in_latent[:, self.id_swap] + (1 - self.a) * w[:, self.id_swap]
            in_latent = [in_latent]
        
        style_image_inverted_B, offsets = self.forward_trainable(
            in_latent, 
            input_is_latent=not self.config.training.mix_stylespace, 
            is_s_code=self.config.training.mix_stylespace
        )
        
        rec_data = {
            'style_inverted_B_1024x1024': style_image_inverted_B,
            'style_image_1024x1024': self.style_image_full_res,
            'style_inverted_B_256x256': self.bicubic(style_image_inverted_B),
            'style_image_256x256': self.style_image_resized,
        }

        return {
            'clip_data': clip_data,
            'rec_data': rec_data,
            'offsets': offsets
        }
    
    def _get_stylespace(self, latents, **kwargs):
        if self.config.training.patch_key == 'original':
            return self.trainable.get_s_code(latents, **kwargs)
        
        return self.source_generator.get_s_code(latents, **kwargs)
    
    def _get_w(self, latents, **kwargs):
        if self.config.training.patch_key == 'original':
            return self.trainable.style(latents, **kwargs)
        
        return self.source_generator.style(latents, **kwargs)
    
    @torch.no_grad()
    def log_images(self):
        self.trainable.eval()
        dict_to_log = {}

        for idx, z in enumerate(self.zs_for_logging):
            w_styles = self.source_generator.style(z)
            sampled_imgs, _ = self.forward_trainable(z, truncation=self.config.logging.truncation)
            sampled_imgs = construct_paper_image_grid(sampled_imgs)

            dict_to_log.update({
                f"trg_domain_grids/{Path(self.config.training.target_class).stem}/{idx}": sampled_imgs,

            })

        rec_img, _ = self.forward_trainable(
            self.style_image_latents,
            input_is_latent=not self.config.training.mix_stylespace, 
            is_s_code=self.config.training.mix_stylespace
        )
        rec_img = t2im(rec_img.squeeze())
        dict_to_log.update({"style_image/projected_B": rec_img})
        self.logger.log_images(self.current_step, dict_to_log)
        
    @torch.no_grad()
    def log_images_for_low_memory(self):
        self.trainable.eval()
        dict_to_log = {}

        for idx, z in enumerate(self.zs_for_logging):
            w_styles = self.source_generator.style(z)
            sampled_imgs, _ = self.forward_trainable(z, truncation=self.config.logging.truncation)
            sampled_imgs = torch.cat([sampled_imgs[0], sampled_imgs[1]], dim=2)
            sampled_imgs = t2im(sampled_imgs)

            dict_to_log.update({
                f"trg_domain_grids/{Path(self.config.training.target_class).stem}/{idx}": sampled_imgs,

            })

        rec_img, _ = self.forward_trainable(
            self.style_image_latents,
            input_is_latent=not self.config.training.mix_stylespace, 
            is_s_code=self.config.training.mix_stylespace
        )
        rec_img = t2im(rec_img.squeeze())
        dict_to_log.update({"style_image/projected_B": rec_img})
        self.logger.log_images(self.current_step, dict_to_log)
    
    def ckpt_info(self):
        return {
            'da_type': 'im2im_jojo'
        }


@trainer_registry.add_to_registry('im2im_difa')
class DiFATrainer(Image2ImageSingleDomainAdaptationTrainer):
    def __init__(self, config):
        super().__init__(config)

        self.style_image_latents = None
        self.style_image_full_res = None
        self.style_image_resized = None
        self.style_image_inverted_A = None
        
        self.setup_hook = False
        self.hook_cache = {}
    
    def setup(self):        
        self._setup_base()
        self._setup_trainable()
        self._setup_optimizer()
        
        self._setup_hooks()
        self._setup_style_image()
        self._log_target_images()
        self._setup_embedding()
        
        if 'difa_w' in self.config.optimization_setup.loss_funcs:
            self._setup_latent_encoder()
    
    def _setup_latent_encoder(self):
        print('Setting PSP encoder')
        from restyle_encoders.e4e import e4e
        from argparse import Namespace
        
        model_path = 'pretrained/e4e_ffhq_encode.pt'
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = model_path
        opts= Namespace(**opts)
        self.net = e4e(opts).eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize(256)
        ])
        
        self.loss_function.difa_w.iter = self.config.training.iter_num
        print(self.loss_function.difa_w.iter, 'loss difa_w iter')
        
    def get_image_latents(self, image, norm=False):
        img = self.transform(image)
        images, w_plus = self.net(img, randomize_noise=False, return_latents=True)
        
        w_plus = w_plus.reshape(img.size(0), -1)
        
        if norm:
            w_plus /= w_plus.clone().norm(dim=-1, keepdim=True)
        
        return w_plus
    
    def _setup_embedding(self):
        if self.has_clip_loss:
            self._process_target_embeddings()
            self._process_source_embeddings()
    
    @torch.no_grad()
    def _process_target_embeddings(self):
        self.trg_embeddings = {}
        self.trg_keys = {}
        self.trg_tokens = {}
        
        def _convert_image_to_rgb(image):
            return image.convert("RGB")

        from PIL import Image
        tr = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        
        for clip_key, (m, p) in self.batch_generators.items(): 
            im = tr(Image.open(self.config.training.target_class)).unsqueeze(0).to(self.device)
        
            encoding = m.encode_image(im)
            encoding /= encoding.clone().norm(dim=-1, keepdim=True)            
            
            self.trg_embeddings[clip_key] = encoding
            
            if self.setup_hook:
                trg_tokens = self.hook_cache[clip_key]['feat_tokens'][0]
                # trg_tokens /= trg_tokens.clone().norm(dim=-1, keepdim=True)
                
                trg_keys = self.hook_cache[clip_key]['feat_keys'][0]
                self.trg_keys[clip_key] = trg_keys
                self.trg_tokens[clip_key] = trg_tokens
            
            self._flush_hook_data()
    
    def _setup_hooks(self):
        if 'difa_local' in self.config.optimization_setup.loss_funcs:
            self.setup_hook = True
        
        if self.setup_hook:
            self.hook_cache = {k: {
                'feat_keys': [],
                'feat_tokens': [],
                'gen_attn_weights': [],
            } for k in self.batch_generators}
            self.hook_handlers = []
    
            self._register_hooks(layer_ids=[self.config.training.clip_layer], facet='key')
    
    def _get_hook(self, clip_key, facet):
        visual_model = self.batch_generators[clip_key][0]
        if facet in ['token']:
            def _hook(model, input, output):
                input = model.ln_1(input[0])
                attnmap = model.attn(input, input, input, need_weights=True, attn_mask=model.attn_mask)[1]
                self.hook_cache[clip_key]['feat_tokens'].append(output[1:].permute(1, 0, 2))
                self.hook_cache[clip_key]['gen_attn_weights'].append(attnmap)
            return _hook
        elif facet == 'feat':
            def _outer_hook(model, input, output):
                output = output[1:].permute(1, 0, 2)  # LxBxD -> BxLxD
                # TODO: Remember to add VisualTransformer ln_post, i.e. LayerNorm
                output = F.layer_norm(output, visual_model.ln_post.normalized_shape, \
                    visual_model.ln_post.weight.type(output.dtype), \
                    visual_model.ln_post.bias.type(output.dtype), \
                        visual_model.ln_post.eps)
                output = output @ visual_model.proj
                self.hook_cache[clip_key]['feat_tokens'].append(output)
            return _outer_hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            N, B, C = input.shape
            weight = module.in_proj_weight.detach()
            bias = module.in_proj_bias.detach()
            qkv = F.linear(input, weight, bias)[1:]  # remove cls key
            qkv = qkv.reshape(-1, B, 3, C).permute(2, 1, 0, 3)  # BxNxC
            self.hook_cache[clip_key]['feat_keys'].append(qkv[facet_idx])
        return _inner_hook
    
    def _register_hooks(self, layer_ids, facet='key'):
        for clip_name, (model, preprocess) in self.batch_generators.items():        
            for block_idx, block in enumerate(model.visual.transformer.resblocks):
                if block_idx in layer_ids:
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(clip_name, 'token')))
                    assert facet in ['key', 'query', 'value']
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(clip_name, facet)))
    
    def _unregister_hooks(self):
        for handle in self.hook_handlers:
            handle.remove()
        
        self.hook_handlers = []
    
    def _flush_hook_data(self):
        if self.setup_hook:
            self.hook_cache = {k: {
                'feat_keys': [],
                'feat_tokens': [],
                'gen_attn_weights': [],
            } for k in self.batch_generators}
        
    def calc_batch(self, sample_z):
        self._flush_hook_data()
        clip_data = {k: {} for k in self.batch_generators}
        
        frozen_img = self.forward_source(sample_z)
        trainable_img, offsets = self.forward_trainable(sample_z)

        
        if self.has_clip_loss:
            for visual_encoder_key, (model, preprocess) in self.batch_generators.items():

                src_encoded = self.clip_encode_image(model, frozen_img, preprocess)
                trg_encoded = self.clip_encode_image(model, trainable_img, preprocess)

                if self.setup_hook:
                    src_tokens = self.hook_cache[visual_encoder_key]['feat_tokens'][0]
                    src_tokens /= src_tokens.clone().norm(dim=-1, keepdim=True)

                    trg_tokens = self.hook_cache[visual_encoder_key]['feat_tokens'][1]
                    trg_tokens /= trg_tokens.clone().norm(dim=-1, keepdim=True)

                    clip_data[visual_encoder_key].update({
                        'trg_tokens': trg_tokens,
                        'src_tokens': src_tokens,
                        'trg_tokens_style': self.trg_tokens[visual_encoder_key]
                    })

                # trg_trainable_emb = self.clip_encode_image(model, style_image_inverted_B, preprocess)
                clip_data[visual_encoder_key].update({
                    'trg_encoded': trg_encoded,
                    'src_encoded': src_encoded,
                    'trg_domain_emb': self.trg_embeddings[visual_encoder_key],
                    'src_domain_emb': self.src_embeddings[visual_encoder_key],
                    # 'trg_trainable_emb': trg_trainable_emb,
                    # 'trg_emb': self.trg_embeddings[visual_encoder_key]
                })
        
            self._flush_hook_data()
        
        rec_data = {}
        
        if 'difa_w' in self.config.optimization_setup.loss_funcs:
            inv_data = {
                'src_latents': self.get_image_latents(frozen_img),
                'trg_latents': self.get_image_latents(trainable_img),
                'iters': self.current_step
            }
        else:
            inv_data = {}
        
        return {
            'clip_data': clip_data,
            'rec_data': rec_data,
            'offsets': offsets,
            'inv_data': inv_data
        }
    
    @torch.no_grad()
    def log_images(self):
        self.trainable.eval()
        dict_to_log = {}

        for idx, z in enumerate(self.zs_for_logging):
            w_styles = self.source_generator.style(z)
            sampled_imgs, _ = self.forward_trainable(z, truncation=self.config.logging.truncation)
            sampled_imgs = construct_paper_image_grid(sampled_imgs)

            dict_to_log.update({
                f"trg_domain_grids/{Path(self.config.training.target_class).stem}/{idx}": sampled_imgs,

            })

        rec_img, _ = self.forward_trainable(
            [self.style_image_latents],
            input_is_latent=True, 
        )
        rec_img = t2im(rec_img.squeeze())
        dict_to_log.update({"style_image/projected_B": rec_img})
        self.logger.log_images(self.current_step, dict_to_log)
        
    @torch.no_grad()
    def log_images_for_low_memory(self):
        self.trainable.eval()
        dict_to_log = {}

        for idx, z in enumerate(self.zs_for_logging):
            w_styles = self.source_generator.style(z)
            sampled_imgs, _ = self.forward_trainable(z, truncation=self.config.logging.truncation)
            sampled_imgs = torch.cat([sampled_imgs[0], sampled_imgs[1]], dim=2)
            sampled_imgs = t2im(sampled_imgs)

            dict_to_log.update({
                f"trg_domain_grids/{Path(self.config.training.target_class).stem}/{idx}": sampled_imgs,

            })

        rec_img, _ = self.forward_trainable(
            [self.style_image_latents],
            input_is_latent=True, 
        )
        rec_img = t2im(rec_img.squeeze())
        dict_to_log.update({"style_image/projected_B": rec_img})
        self.logger.log_images(self.current_step, dict_to_log)
    
    def ckpt_info(self):
        return {
            'da_type': 'im2im_difa'
        }
