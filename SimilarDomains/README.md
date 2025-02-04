# Similar Domains technical details

## Table of Contents
  * [Getting Started](#getting-started)
  * [Model Training](#model-training)
    + [Launch Details](#launch-details)
    + [Config Setup](#config-setup)
    + [Additional notes](#additional-notes)
  * [Inference Notebooks](#inference-notebooks)
    + [Setup](#setup)
  * [Related Works](#related-works)
  * [Structure](#structure)
  

## Getting Started

Clone this repo and install dependencies via `pip install requirements.txt`.


### Pretrained models

run 

```bash
python download.py
```

Pretrained models

* StyleGAN2 (ffhq, afhqcat, afhqdog, cars, horses, churches)
* Finetuned StyleGAN2 models (ffhq -> afhqcat, ffhq -> afhqdog, ffhq -> metfaces, ffhq -> mega)
* Pretrained StyleSpace directions (StyleSpace directions based on DiFa and StyleGAN-NADA models)


## Model Training

### Launch Details

Training launches by following command

```
python main.py exp.config={config_name}
```

configs are stored in `configs` dir

4 model types are implemented in trainers.py, see the file to get details.

* Available methods:
    1. [MindTheGap](https://arxiv.org/abs/2110.08398) (configs/im2im_mtg.yaml)
    2. [DiFa](https://arxiv.org/abs/2207.08736) (configs/im2im_difa.yaml)
    3. [JoJoGAN](https://arxiv.org/abs/2112.11641) (configs/im2im_jojo.yaml)
    4. [StyleGAN-NADA](https://arxiv.org/abs/2108.00946) (configs/td_single_ffhq.yaml)

Ensure config settings to launch training with appropriate setup:

### Config setup

#### Exp:
  * `config_dir`: configs
  * `config`: config_name.yaml
  * `project`: `WandbProjectName`
  * `tags`:
    - tag1
    - tag2
  * `name`: `WandbRunName`
  * `seed`: 0
  * `root`: ./
  * `notes`: empty notes
  * `step_save`: 20 – *model dump frequency*
  * `trainer`: trainer_name
  * `models_root`: (used this folder as root for experiments when specified)
  
#### Training:
  * `iter_num`: 300 – *number of training iterations*
  * `batch_size`: 4
  * `device`: cuda:0
  * `generator`: stylegan2
  * `patch_key`: cin_mult (see __core/parametrizations.py__)
  * `phase`: mapping – **StyleGAN2 part which is fine-tuned, only used when `patch_key` = `original`**
  * `source_class`: Photo – *description of source domain*
  * `target_class`: 3D Render in the Style of Pixar – *description of target domain* (in case of im2im setup image path is required)
  * `auto_layer_k`: 16
  * `auto_layer_iters`: 0 – *number of iterations for adaptive corresponding stylegan2 layer freeze*
  * `auto_layer_batch`: 8
  * `mixing_noise`: 0.9

#### Optimization_setup:
  * `visual_encoders`: – *clip encoders that are used for clip based losses*
    - ViT-B/32
    - ViT-B/16
  * `loss_funcs`:
    - loss_name1
    - loss_name2
    - ...
  * `loss_coefs`:
    - loss_coef1
    - loss_coef2
    - ...
  * `g_reg_every`: 4 – *stylegan2 regularization coefficient (not recommended to change)* 
  * `optimizer`:
    * `weight_decay`: 0.0
    * `lr`: 0.02
    * `betas`:
      - 0.9
      - 0.999

#### Logging: (based on WanDB)
  * `log_every`: 10 – *loss logging step*
  * `log_images`: 20 – *images logging step*
  * `truncation`: 0.7 – *truncation during images logging*
  * `num_grid_outputs`: 1 – *number of logging grids*


### Additional notes

When training ends model checkpoints could be found in `local_logged_exps/`. Each `ckpt_name.pt` could be inferenced using a helper classes `Inferencer` in `core/utils/example_utils.py`.

## Inference Notebooks

<!-- * [Inference Notebooks](#inference-notebooks)
    + [Setup](#setup)
    + [Multuple Stylisation](#mutiple-stylisation)
    + [Combined Morphing](#combined-moprhing)
    + [Stylisation for transfered GAN](#stylisation-transfer)
    + [Editing](#editing) -->

### Setup

Pretrained models for various stylization are provided. 
Please refer to `download.py` and run `python download.py ckpt` from repository root.

Each notebook is depend on pretrained models from `download.py`

### Multiple Stylisation (paper Figures 6, 19)

* examples/multiple_morping.ipynb

### Combined Morphing (paper Figure 1)

* examples/combined_morphing.ipynb

### Stylisation Transfer (paper Figures 5, 15, 16)

* examples/adaptation_in_finetuned_gan.ipynb

### Editing (paper Figures 20, 21)

* examples/editing.ipynb

### Pruned offsets inference (paper Figures 10, 11, 12)

* examples/pruned_forward.ipynb

## Related Works

**StyleGAN2 implementation:**  
https://github.com/rosinality/stylegan2-pytorch
License (MIT) https://github.com/rosinality/stylegan2-pytorch/blob/master/LICENSE  

**Encoders implementation**
https://github.com/yuval-alaluf/restyle-encoder
License (MIT) https://github.com/yuval-alaluf/restyle-encoder/blob/main/LICENSE

**StyleFlow implementation**
https://github.com/RameenAbdal/StyleFlow/tree/master
License [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

**LPIPS implementation:**  
https://github.com/S-aiueo32/lpips-pytorch
License (BSD 2-Clause) https://github.com/S-aiueo32/lpips-pytorch/blob/master/LICENSE  




**Please Note**: The CUDA files under the [StyleGAN2 ops directory](https://github.com/eladrich/pixel2style2pixel/tree/master/models/stylegan2/op) are made available under the [Nvidia Source Code License-NC](https://nvlabs.github.io/stylegan2/license.html)


## Structure

| Path | Description <img width=200>
| :--- | :---
| SimilarDomains | Root for one-shot and zero-shot methods
| &boxvr;&nbsp; configs | Folder containing configs defining model training setup
| &boxvr;&nbsp; core | Folder containing implementation of ours method
| &boxvr;&nbsp; editing | Folder containing implementation of editing methods
| &boxvr;&nbsp; examples | Folder with jupyter notebooks containing inference playground
| &boxvr;&nbsp; gan_models | Folder containing implementation of common GAN models
| &boxvr;&nbsp; image_domains | Folder with several examples of image domains for one-shot adaptation
| &boxvr;&nbsp; restyle_encoders | Folder containing implementation of encoders
| &boxvr;&nbsp; main.py | Training launch script
| &boxvr;&nbsp; trainers.py | File with implementations of common domain adaptation methods
