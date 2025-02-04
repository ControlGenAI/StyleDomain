## Base repository
This code is mainly based on the [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) repository. We basically modify the training procedure and the StyleGAN2 architecture. 

## Getting started
This code has the same prerequisites and requirements as [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) repository.

Basic example of models loading and inference can be found in the notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/124GIlJ1KvKQ6Z1myZbKgk8u4YDy4An2e?usp=sharing)

List of pretrained models and checkpoint's download URLs can be found in the same notebook.

More complex usages of the models can be found under the section **Reproduction of paper results** down below.

## Preparing datasets

Generally, preprocessing should be performed exactly like in [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch).

We use three different resolutions for different experiments:
* We use a resolution of $256 \times 256$ for Few-Shot setup for fair comparison with existing baselines (AdAM and CDC)
* We use a resolution of $1024 \times 1024$ as it is required by semantic editing models (StyleFlow, StyleSpace)
* All other experiments were carried out in a resolution of $512 \times 512$
As a result, we provide exact instructions for dataset preprocessing in all these resolutions.

### [Metfaces](https://github.com/NVlabs/metfaces-dataset)
```bash
python dataset_tool.py --source ./metfaces/images --dest ~/datasets/metfaces.zip
python dataset_tool.py --source ./metfaces/images --dest ~/datasets/metfaces_512.zip --width=512 --height=512
```

### [Mega Cartoon Faces](https://github.com/justinpinkney/toonify)
```bash
python dataset_tool.py --source ./cartoon --dest ~/datasets/Mega.zip
python dataset_tool.py --source ./cartoon --dest ~/datasets/Mega_512.zip --width=512 --height=512
```

### [Ukiyoe](https://github.com/justinpinkney/toonify)
```bash
python dataset_tool.py --source ./ukiyoe --dest ~/datasets/ukiyoe.zip
python dataset_tool.py --source ./ukiyoe --dest ~/datasets/ukiyoe_512.zip  --width=512 --height=512
```

### [AFHQ Dog, Cat and Wild](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq)
```bash
python dataset_tool.py --source ./train/cat --dest ~/datasets/afhqcat_256.zip --width 256 --height 256
python dataset_tool.py --source ./train/dog --dest ~/datasets/afhqdog_256.zip --width 256 --height 256
python dataset_tool.py --source ./train/wild --dest ~/datasets/afhqwild_256.zip --width 256 --height 256

python dataset_tool.py --source ./train/cat --dest ~/datasets/afhqcat.zip --width 512 --height 512
python dataset_tool.py --source ./train/dog --dest ~/datasets/afhqdog.zip --width 512 --height 512
python dataset_tool.py --source ./train/wild --dest ~/datasets/afhqwild.zip --width 512 --height 512

python dataset_tool.py --source ./train/cat --dest ~/datasets/afhqcat_1024.zip --width 1024 --height 1024
python dataset_tool.py --source ./train/dog --dest ~/datasets/afhqdog_1024.zip --width 1024 --height 1024
python dataset_tool.py --source ./train/wild --dest ~/datasets/afhqwild_1024.zip --width 1024 --height 1024

# Validation parts of datasets were used in cross-domain image translation 
python dataset_tool.py --source ./val/cat --dest ~/datasets/afhqcat_val.zip --width 512 --height 512
python dataset_tool.py --source ./val/dog --dest ~/datasets/afhqdog_val.zip --width 512 --height 512
python dataset_tool.py --source ./val/wild --dest ~/datasets/afhqwild_val.zip --width 512 --height 512
```

### [Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
We used the following protocol to obtain training data:
* Download images and class labels
* Split all classes into two groups (train and validation classes)
* Take a subset of images that correspond to train classes (namely, we took classes $4, 6, 7, 8, 9, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 28, 29, 30, 31, 32, 35, 36, 39, 40, 44, 45, 46, 48, 49, 53, 55, 56, 57, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 74, 76, 79, 80, 82, 84, 87, 88, 90, 92, 97, 100, 101, 102$)
* Preprocess those images as usual:
```bash
python dataset_tool.py --source flowers-102/random_01/train --dest ~/datasets/flowers_102_train_random_01.zip --width 512 --height 512
```

### [LSUN Car](http://dl.yf.io/lsun/objects/) 
* Download `car.zip` and extract all files to `./car_lmdb`
* Note, some images have a width lower than $512$, so they will be skipped during preprocessing. So, we set `--max-images=62606` to take exactly $10000$ images from the training dataset.
```bash
python dataset_tool.py --source ./car_lmdb --dest ~/datasets/lsun_cars_512_10k.zip --width 512 --height 384 --max-images 62606 --transform center-crop-wide

# In order to get images for Few-Shot setup we downscale 512x512 images to 256x256 resolution:
unzip ~/datasets/lsun_cars_512_10k.zip -d ~/datasets/lsun_cars_512_10k
python dataset_tool.py --source ~/datasets/lsun_cars_512_10k --dest ~/datasets/lsun_cars_256_10k.zip --width 256 --height 256
```


### [LSUN Church](http://dl.yf.io/lsun/scenes/) 
* Download `church_outdoor_train_lmdb.zip` and extract all files to `./church_outdoor_train_lmdb`
```bash
python dataset_tool.py --source ./church_outdoor_train_lmdb --dest ~/datasets/lsun_church_outdoor_train_10k.zip --width 256 --height 256 --max-images 10000
python dataset_tool.py --source ./church_outdoor_train_lmdb --dest ~/datasets/lsun_church_outdoor_train_512_10k.zip --width 512 --height 512 --max-images 10000
```

## Training

In order to perform experiments, some additional options have been implemented that can be used through the CLI.
All new CLI arguments are optional. 

Here is a complete list of those options:
```bash
    # The exact CUDA device idx for single GPU training 
    --gpu # Optional[int]. Default: None
    # Override learning rate for both Generator and Discriminator
    --lrate # Optional[float]. Default: None
    # Override the learning rate for Generator (even if lrate is specified)
    --glrate # Optional[float]. Default: None
    # Override the learning rate for Discriminator (even if lrate is specified)
    --dlrate # Optional[float]. Default: None
    # A comma-separated list of Generator parameter groups that will be updated during training
    # This option does not register any parameters or buffers. It only specifies a list of parameters
    #   to require_grad right before forward and backward pass during training.
    # For more information, see training.training_loop.set_requires_grad and line 370.
    # * if an empty list is specified, the Generator will be frozen during training
    # The following options are allowed (any combination of them):
    # * all --- update all Generator weights
    # * mapping --- update Mapping Network
    # * synt_const --- update Const Input to Synthesis Network
    # The options down below can be altered by specifying an exact layer:
    # For instance, to train a convolutional block in the Synthesis Network 
    #   corresponding to resolution 256, one can specify synt_conv.b256
    # The allowed resolutions are: 4, 8, 16, 32, 64, 128, 256, 512, 1024
    # * (tRGB|synt)_affine --- update Affine Layers
    # * (tRGB|synt)_conv --- update Convolutional Layers
    # * (tRGB|synt)_offset --- update StyleDomain directions
    # * (tRGB|synt)_weights_offset --- update Weight Offsets
    # * (tRGB|synt)_affine_weights_offset --- update Weight Offsets in Affine layers
    --generator-requires-grad-parts # Optional[str]. Default: ''
    # Flag that specify whether to use StyleDomain directions, Weight Offsets or Affine Weight Offsets
    --use-domain-modulation # Optional[bool]: Default: False
    # A comma-separated list that specifies exact 
    #   parameterization of StyleDomain directions, Weight Offsets and Affine Weight Offsets
    # This list must contain at most one type of StyleDomain directions, 
    #   at most one type of Weight Offsets and at most one type of Affine Weight Offsets
    # The following types of StyleDomain directions are allowed:
    # * additive (the one that is used in StyleSpace parameter space)
    # * multiplicative
    # * additive_w_space
    # * multiplicative_w_space
    # The following types of Weight Offsets are allowed:
    # * in, in_additive, out, out_additive, spatial, spatial_additive
    # * in_spatial, in_spatial_additive, out_spatial, out_spatial_additive
    # * out_in, out_in_additive (the one that is used in StyleSpace+ and Affine+ parameter spaces)
    # * out+in, out+in_additive
    # * out_in_(1|5|10|20|50|100|256|512)
    # The following types of Affine Weight Offsets are allowed:
    # * affine_out_in_([0-9]+)_([0-9]+), affine_out_in_([0-9]+)_([0-9]+)_additive (the one that is used in AffineLight+)
    # For more information, see training.networks.register_*_modulation and
    #   training.networks.weight_to_weight
    --domain-modulation-parametrization # Optional[str]. Default: 'multiplicative'
```

Examples:
```bash
# Define data-specific parameters:
# Fow 256x256 resolution (Few-shot setup):
cfg=paper256
resume=ffhq256
dataset_path=~/datasets/lsun_cars_256_10k.zip

# For 512x512 resolution:
cfg=stylegan2
resume=ffhq512
dataset_path=~/datasets/metfaces_512.zip

# For 1024x1024 resolution:
cfg=stylegan2
resume=ffhq1024
dataset_path=~/datasets/afhqdog_1024.zip


# 01. Train Full parametrization:
python train.py --outdir=~/training-runs \
  --cfg ${cfg} --resume ${resume} --data "${dataset_path}" \
  --gpu=0 --kimg 241 --snap 5 --metrics fid5k \
  --generator-requires-grad-parts all

# 02. Train Mapping parametrization:
python train.py --outdir=~/training-runs \
  --cfg ${cfg} --resume ${resume} --data "${dataset_path}" \
  --gpu=0 --kimg 241 --snap 5 --metrics fid5k \
  --generator-requires-grad-parts mapping

# 03. Train SyntConv parametrization:
python train.py --outdir=~/training-runs \
  --cfg ${cfg} --resume ${resume} --data "${dataset_path}" \
  --gpu=0 --kimg 241 --snap 5 --metrics fid5k \
  --generator-requires-grad-parts synt_conv,tRGB_conv

# 04. Train Affine parametrization:
python train.py --outdir=~/training-runs \
  --cfg ${cfg} --resume ${resume} --data "${dataset_path}" \
  --gpu=0 --kimg 241 --snap 5 --metrics fid5k \
  --generator-requires-grad-parts synt_affine,tRGB_affine

# 05. Train Affine+ (Affine+64) with Generator learning rate 0.02:
python train.py --outdir=~/training-runs \
  --cfg ${cfg} --resume ${resume} --data "${dataset_path}" \
  --gpu=0 --kimg 241 --snap 5 --metrics fid5k --glrate 0.02 \
  --generator-requires-grad-parts synt_affine,tRGB_affine,synt_weights_offset.b64,tRGB_weights_offset.b64 \
  --use-domain-modulation --domain-modulation-parametrization out_in_additive
  
# 06. Train AffineLight+ parametrization with Generator learning rate 0.02:
python train.py --outdir=~/training-runs \
  --cfg ${cfg} --resume ${resume} --data "${dataset_path}" \
  --gpu=0 --kimg 241 --snap 5 --metrics fid5k --glrate 0.02 \
  --generator-requires-grad-parts synt_affine_weights_offset,tRGB_affine_weights_offset,synt_weights_offset.b64,tRGB_weights_offset.b64 \
  --use-domain-modulation --domain-modulation-parametrization out_in_additive,affine_out_in_5_1_additive
  
# 07. Train StyleSpace parametrization with Generator learning rate 0.02:
python train.py --outdir=~/training-runs \
  --cfg ${cfg} --resume ${resume} --data "${dataset_path}" \
  --gpu=0 --kimg 241 --snap 5 --metrics fid5k --glrate 0.02 \
  --generator-requires-grad-parts synt_offset,tRGB_offset \
  --use-domain-modulation --domain-modulation-parametrization additive
```

## Evaluation
We compute metrics for Moderately Similar and Dissimilar Domains using `calc_metrics.py`.
For instance:
```bash
dataset_path=~/datasets/afhqdog.zip
network_chkpt=~/training-runs/00000-afhqdog-stylegan2-kimg241-resumeffhq512/network-snapshot-000241.pkl
# The new CLI option `--gpu` allows metrics to be computed on a specific device. Works only when `--gpus=1`
python calc_metrics.py --data "${dataset_path}" --network "${network_chkpt}" --gpus=1 --gpu=0 --metrics fid50k,kid50k
```

See [stylegan2-ada documentation](https://github.com/NVlabs/stylegan2-ada-pytorch#quality-metrics) for more info.

## Reproduction of paper results

We provide a set of Jupyter Notebooks that reproduce most of the results presented on Moderately Similar and Dissimilar Domains.

* [A.3.3, A.3.4, A.4.2 Quantitative and Qualitative Results](examples/General%20Results%20ICCV.ipynb)
* [A.5.4 Semantic Editing for Moderately Similar and Dissimilar Domains](examples/Semantic%20Editing%20ICCV.ipynb)
* [A.8 Cross-domain image translation](examples/I2I%20ICCV.ipynb)

## Citation
```latex
@misc{...,
  doi = {...},
  url = {...},
  author = {...},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {StyleDomain: Analysis of StyleSpace for Domain Adaptation of StyleGAN},
  publisher = {...},
  year = {2022},
  copyright = {...}
}
```
