import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp

from collections import defaultdict
from core import lpips

from core.utils.loss_utils import (
    cosine_loss,
    mse_loss,
    get_tril_elements_mask
)
from core.utils.class_registry import ClassRegistry
from gan_models.StyleGAN2.model import DiscriminatorJojo
from functools import lru_cache


clip_loss_registry = ClassRegistry()
rec_loss_registry = ClassRegistry()
reg_loss_registry = ClassRegistry()


class LossBuilder(torch.nn.Module):
    def __init__(self, opt):
        super(LossBuilder, self).__init__()

        self.opt = opt
        self.parsed_loss = [[opt.l2_lambda, "l2"], [opt.percept_lambda, "percep"]]
        self.l2 = torch.nn.MSELoss()
        if opt.device == "cuda":
            use_gpu = True
        else:
            use_gpu = False
        self.percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=use_gpu)
        self.percept.eval()

    def _loss_l2(self, gen_im, ref_im, **kwargs):
        return self.l2(gen_im, ref_im)

    def _loss_lpips(self, gen_im, ref_im, **kwargs):
        return self.percept(gen_im, ref_im).sum()

    def forward(self, ref_im_H, ref_im_L, gen_im_H, gen_im_L):
        loss = 0
        loss_fun_dict = {
            "l2": self._loss_l2,
            "percep": self._loss_lpips,
        }
        losses = {}
        for weight, loss_type in self.parsed_loss:
            if loss_type == "l2":
                var_dict = {
                    "gen_im": gen_im_H,
                    "ref_im": ref_im_H,
                }
            elif loss_type == "percep":
                var_dict = {
                    "gen_im": gen_im_L,
                    "ref_im": ref_im_L,
                }
            tmp_loss = loss_fun_dict[loss_type](**var_dict)
            losses[loss_type] = tmp_loss
            loss += weight * tmp_loss
        return loss, losses


@reg_loss_registry.add_to_registry("offsets_l2")
def l2_offsets(
    offsets: tp.Dict[str, tp.Dict[str, torch.Tensor]]
):
    loss = 0.
    for conv_key, conv_inputs in offsets.items():
        layer_deltas = sum([v for v in conv_inputs.values()])
        loss += torch.pow(layer_deltas, 2).sum() / torch.numel(layer_deltas)

    return loss


@reg_loss_registry.add_to_registry("affine_l2")
def cout_affine_l2_loss(
    offsets: tp.Dict[str, tp.Dict[str, torch.Tensor]]
):
    loss = 0.
    for conv_key, conv_inputs in offsets.items():
        val = (torch.pow(conv_inputs['gamma'] - 1, 2) + torch.pow(conv_inputs['beta'], 2)).sum()
        loss += val / torch.numel(conv_inputs['gamma'])
    return  loss


@reg_loss_registry.add_to_registry("offsets_l1")
def l1_offsets(
    offsets: tp.Dict[str, tp.Dict[str, torch.Tensor]]
):
    loss = 0.
    for conv_key, conv_inputs in offsets.items():
        layer_deltas = sum([v for v in conv_inputs.values()])
        loss += torch.abs(layer_deltas).sum() / torch.numel(layer_deltas)

    return loss


@clip_loss_registry.add_to_registry("global")
def global_loss(
    clip_batch
) -> torch.Tensor:
    trg_encoded, trg_domain_emb = clip_batch['trg_encoded'], clip_batch['trg_domain_emb']
    return cosine_loss(trg_encoded, trg_domain_emb).mean()

# batch /
#     clip_data /
#         ViT-B/32 /
#             src_encoded
#             (src_emb_domain..)


# @clip_loss_registry.add_to_registry("direction")
# def direction_loss(
#     trg_encoded: torch.Tensor, src_encoded: torch.Tensor,
#     trg_domain_emb: torch.Tensor, src_domain_emb: torch.Tensor
# ) -> torch.Tensor:

#     edit_im_direction = trg_encoded - src_encoded
#     edit_domain_direction = trg_domain_emb - src_domain_emb
        
#     if trg_domain_emb.ndim == 3:
#         edit_domain_direction = edit_domain_direction.mean(axis=1)
        
#     return cosine_loss(edit_im_direction, edit_domain_direction).mean()


@clip_loss_registry.add_to_registry("direction")
def direction_loss(
    clip_batch
) -> torch.Tensor:
    
    trg_encoded, src_encoded = clip_batch['trg_encoded'], clip_batch['src_encoded']
    trg_domain_emb, src_domain_emb = clip_batch['trg_domain_emb'], clip_batch['src_domain_emb']
    
    edit_im_direction = trg_encoded - src_encoded
    edit_domain_direction = trg_domain_emb - src_domain_emb
        
    if trg_domain_emb.ndim == 3:
        edit_domain_direction = edit_domain_direction.mean(axis=1)
        
    return cosine_loss(edit_im_direction, edit_domain_direction).mean()


@clip_loss_registry.add_to_registry("indomain")
def indomain_loss(
    clip_batch
) -> torch.Tensor:
    
    trg_encoded, src_encoded = clip_batch['trg_encoded'], clip_batch['src_encoded']
        
    src_cosines = src_encoded @ src_encoded.T
    trg_cosines = trg_encoded @ trg_encoded.T
    mask = torch.from_numpy(get_tril_elements_mask(src_encoded.size(0)))
    
    src_cosines = src_cosines[mask]
    trg_cosines = trg_cosines[mask]

    loss = torch.sum((src_cosines - trg_cosines) ** 2) / src_encoded.size(0) / (src_encoded.size(0) - 1) * 2

    return loss


@clip_loss_registry.add_to_registry("tt_direction")
def target_target_direction(
    clip_batch
) -> torch.Tensor:
    
    trg_encoded, src_encoded = clip_batch['trg_encoded'], clip_batch['src_encoded']
    trg_domain_emb, src_domain_emb = clip_batch['trg_domain_emb'], clip_batch['src_domain_emb']
    
    mask = torch.from_numpy(get_tril_elements_mask(trg_encoded.size(0)))
    
    deltas_text = (trg_domain_emb.unsqueeze(0) - trg_domain_emb.unsqueeze(1))[mask]
    deltas_img = (trg_encoded.unsqueeze(0) - trg_encoded.unsqueeze(1))[mask]
        
    if trg_domain_emb.ndim == 3:
        deltas_text = deltas_text.mean(dim=1)
    
    res_loss = cosine_loss(deltas_img.float(), deltas_text.float())
    
    return res_loss.mean()


@clip_loss_registry.add_to_registry('clip_within')
def clip_within(
    clip_batch
):
    trg_encoded, src_encoded = clip_batch['trg_encoded'], clip_batch['src_encoded']
    style_image_trg_encoded, style_image_src_encoded = clip_batch['trg_domain_emb'], clip_batch['src_domain_emb']
    
    trg_direction = trg_encoded - style_image_trg_encoded
    src_direction = src_encoded - style_image_src_encoded

    return cosine_loss(trg_direction, src_direction).mean()


@clip_loss_registry.add_to_registry('clip_ref')
def clip_ref(
    clip_batch: tp.Dict[str, tp.Dict]
):
    
    trg_trainable_emb = clip_batch['trg_trainable_emb']
    trg_emb = clip_batch['trg_emb']
    return cosine_loss(trg_trainable_emb, trg_emb).mean()
    

@clip_loss_registry.add_to_registry('difa_local')
def clip_difa_local(
    clip_batch: tp.Dict[str, tp.Dict]
):
    
    tgt_tokens, src_tokens = clip_batch['trg_tokens'], clip_batch['src_tokens']
    B, N, _ = tgt_tokens.shape
    style_tokens = clip_batch['trg_tokens_style'].repeat(B, 1, 1)

    tgt_tokens /= tgt_tokens.clone().norm(dim=-1, keepdim=True)
    style_tokens /= style_tokens.clone().norm(dim=-1, keepdim=True)

    attn_weights = torch.bmm(tgt_tokens, style_tokens.permute(0, 2, 1))

    cost_matrix = 1 - attn_weights
    B, N, M = cost_matrix.shape
    row_values, row_indices = cost_matrix.min(dim=2)
    col_values, col_indices = cost_matrix.min(dim=1)

    row_sum = row_values.mean(dim=1)
    col_sum = col_values.mean(dim=1)

    overall = torch.stack([row_sum, col_sum], dim=1)
    return overall.max(dim=1)[0].mean()


@rec_loss_registry.add_to_registry('l2_rec_resized')
def l2_rec(
    batch: tp.Dict[str, torch.Tensor]
):
    return mse_loss(
        batch['style_inverted_B_256x256'],
        batch['style_image_256x256']
    ).mean()


@rec_loss_registry.add_to_registry('l2_rec_fullres')
def l2_rec(
    batch: tp.Dict[str, torch.Tensor]
):
    return mse_loss(
        batch['style_inverted_B_1024x1024'],
        batch['style_image_1024x1024']
    ).mean()


@lru_cache(maxsize=1)
def get_discriminator():
    ckpt = torch.load('pretrained/stylegan2-ffhq-config-f.pt')
    disc = DiscriminatorJojo(1024, 2).eval().to('cuda:0')
    disc.load_state_dict(ckpt["d"], strict=False)
    return disc


@rec_loss_registry.add_to_registry('disc_feat_matching')
def batch(
    batch: tp.Dict[str, torch.Tensor]
):
    disc = get_discriminator()
    
    fake_feat = disc(batch['style_inverted_B_1024x1024'])
    real_feat = disc(batch['style_image_1024x1024'])
    rep = fake_feat[0].size(0) // real_feat[0].size(0)
    real_feat = [f.repeat(rep, 1, 1, 1) for f in real_feat]
    
    return sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)])/len(fake_feat)


@lru_cache(maxsize=1)
def get_model(use_gpu=True):
    return lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=use_gpu)


@rec_loss_registry.add_to_registry('lpips_rec')
def lpips_rec(
    batch: tp.Dict[str, torch.Tensor]
):
    model = get_model()
    
    return model(
        batch['style_inverted_B_256x256'],
        batch['style_image_256x256']
    ).mean()


class BaseLoss:
    loss_registry = []

    def __init__(self, loss_funcs, loss_coefs):
        self.funcs, self.coefs = [], []
        for func, coef in zip(loss_funcs, loss_coefs):
            if func not in self.loss_registry:
                continue
            
            self.funcs.append(func)
            self.coefs.append(coef)

    def __call__(self, batch):
        raise NotImplementedError()


class CLIPLoss(BaseLoss):
    loss_registry = clip_loss_registry

    def __call__(self, batch):
        losses = defaultdict(float)

        for loss, coef in zip(self.funcs, self.coefs):
            for visual_encoder_key, clip_batch in batch['clip_data'].items():
                log_vienc_key = visual_encoder_key.replace('/', '-')
                losses[f'{loss}_{log_vienc_key}'] = coef * self.loss_registry[loss](clip_batch)

        return losses


class RecLoss(BaseLoss):
    loss_registry = rec_loss_registry

    def __call__(self, batch):
        losses = defaultdict(float)

        for loss, coef in zip(self.funcs, self.coefs):
            losses[f'{loss}'] = coef * self.loss_registry[loss](batch['rec_data'])

        return losses


class RegLoss(BaseLoss):
    loss_registry = reg_loss_registry

    def __call__(self, batch):
        losses = defaultdict(float)
        
        for loss, coef in zip(self.funcs, self.coefs):
            losses[f'{loss}'] = coef * self.loss_registry[loss](batch['offsets'])

        return losses


class SCCloss:
    def __init__(self, encoder_type, weight, device='cuda:0'):
        self.num_keep_first = 7
        self.loss_type = 'dynamic'
        self.delta_w_type = 'mean'
        self.sliding_window_size = 50
        self.weight = weight
        self.psp_alpha = 0.6
        
        self.source_set = []
        self.target_set = []
        self.source_pos = 0
        self.target_pos = 0
        self.iter = 0
        
    def __call__(self, batch):
        target_encodings = batch['inv_data']['trg_latents']
        source_encodings = batch['inv_data']['src_latents']
        
        iters = batch['inv_data']['iters']

        if self.num_keep_first > 0:
            keep_num = self.num_keep_first * 512
            target_encodings = target_encodings[:, 0:keep_num]
            source_encodings = source_encodings[:, 0:keep_num]
        
        if self.loss_type == "multi_stage":
            ...
            # loss = self.multi_stage_loss(target_encodings, source_encodings)
        elif self.loss_type == "dynamic":
            delta_w = self.update_w(source_encodings, target_encodings)
            regular_weight = max(0, \
                    (iters - self.sliding_window_size) / (self.iter - self.sliding_window_size))
            loss = regular_weight * self.dynamic_loss(target_encodings, source_encodings, delta_w=delta_w)
        else:
            raise RuntimeError(f"No psp loss whose type is {self.psp_loss_type} !")
        
        return self.weight * loss
    
    def update_w(self, source_encodings, target_encodings):
        if self.delta_w_type == 'mean':
            self.update_queue(source_encodings, target_encodings)
            self.source_mean = torch.stack(self.source_set).mean(0, keepdim=True)
            self.target_mean = torch.stack(self.target_set).mean(0, keepdim=True)
            delta_w = self.target_mean - self.source_mean
            
        return delta_w
    
    def update_queue(self, src_vec, tgt_vec):
        if len(self.target_set) < self.sliding_window_size:
            self.source_set.append(src_vec.clone().mean(0).detach())
            self.target_set.append(tgt_vec.clone().mean(0).detach())
        else:
            self.source_set[self.source_pos] = src_vec.clone().mean(0).detach()
            self.source_pos = (self.source_pos + 1) % self.sliding_window_size
            self.target_set[self.target_pos] = tgt_vec.clone().mean(0).detach()
            self.target_pos = (self.target_pos + 1) % self.sliding_window_size
    
    def dynamic_loss(self, target_encodings, source_encodings, delta_w):
        delta_w = delta_w.flatten()
        num_channel = len(delta_w)
        order = delta_w.abs().argsort()
        chosen_order = order[0:int(self.psp_alpha * num_channel)]
        
        
        cond = torch.zeros(num_channel).to(target_encodings.device)
        cond[chosen_order] = 1
        cond = cond.unsqueeze(0)

        # Get masked encodings
        target_encodings = cond * target_encodings
        source_encodings = cond * source_encodings
        loss = F.l1_loss(target_encodings, source_encodings)
        return loss
    
    
class DirectLoss(nn.Module):
    def __init__(self, loss_config):
        super().__init__()
        self.config = loss_config
        self.loss_funcs = loss_config.loss_funcs
        self.loss_coefs = loss_config.loss_coefs

        for key, c in zip(
            ['clip', 'rec', 'reg'],
            [CLIPLoss, RecLoss, RegLoss]
        ):
            setattr(self, key, c(loss_config.loss_funcs, loss_config.loss_coefs))
        
        if 'difa_w' in loss_config.loss_funcs:
            self.difa_w = SCCloss('e4e', loss_config.loss_coefs[loss_config.loss_funcs.index('difa_w')])
        
    def forward(
        self, batch: tp.Dict[str, tp.Dict]
    ):
        clip_losses = self.clip(batch)
        image_rec_losses = self.rec(batch)
        reg_losses = self.reg(batch)
            
        losses = {**clip_losses, **image_rec_losses, **reg_losses}
        
        if hasattr(self, 'difa_w'):
            losses.update({
                'difa_psp_loss': self.difa_w(batch)
            })
        
        losses['total'] = sum(v for v in losses.values())

        return losses

