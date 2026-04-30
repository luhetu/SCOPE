# Experiment Recipes

This file records the known-good recipes and checkpoint choices so downstream
segmentation/detection runs do not accidentally use weak pretraining.

## Log Locations

- Classification logs: `/home3/dnrx52/Size/logs`
- Segmentation and detection logs: `/home3/dnrx52/SCOPE/slurm/logs`

## ImageNet Classification Pretraining

### ViT-CoPE Small, known good

- Checkpoint: `checkpoint/pretrained/COPE/vitcope_cls_size224_patch16_dim384_depth12_best.pth`
- Duplicate checkpoint: `checkpoint/vitcope_cls_size224_patch16_dim384_depth12_best.pth`
- Accuracy: `best_acc=73.20`, `epoch=288`
- Log: `/home3/dnrx52/Size/logs/cope_small886444.out`
- Config path in log: `configs/vitcope_small.yaml`

Recipe from the successful run:

```yaml
task: cls
model: vitcope
dataset: imagenet
data_dir: /tmp/vitcope
bs: 768
size: 224
n_epochs: 300
patch: 16
dim: 384
depth: 12
heads: 6
mlp_dim: 1536
dim_head: 64
use_cls_token: false
pool: mean
lr: 7.5e-4
min_lr: 0
warmup_epochs: 5
weight_decay: 0.05
betas: [0.9, 0.999]
lrschedule: one_cycle_cosine
opt: adamw
amp: true
aug: true
label_smoothing: true
nowandb: false
```

Important: that successful run was from the older classification recipe. The log
does not show the newer `Mixup/CutMix active`, `RandomErasing`, or `DropPath`
diagnostics. Treat the recipe as:

- `dropout=0.1`, `emb_dropout=0.1` for `vitcope` model construction.
- Label smoothing through cross entropy.
- No effective Mixup/CutMix.
- No effective stochastic depth / DropPath.
- No effective random erasing.

The current stronger DeiT-style recipe can be used for new experiments, but it
is not the same recipe that produced `73.20`.

### ViT-CoPE Base

Good root checkpoints:

- `checkpoint/vitcope_cls_size224_patch16_dim768_depth12_best.pth`
  - `best_acc=73.394`, `epoch=273`
- `checkpoint/vitcope_cls_size224_patch16_dim768_depth12_0422_111221_best.pth`
  - `best_acc=73.284`, `epoch=288`

Bad/weak pretrained checkpoint:

- `checkpoint/pretrained/COPE/vitcope_cls_size224_patch16_dim768_depth12_best.pth`
  - `best_acc=32.266`, `epoch=14`

Do not use the weak `pretrained/COPE` base checkpoint for downstream runs.
Point base CoPE downstream configs to one of the root `73.x` checkpoints instead.

### ViT-CoPE Tiny

Known root checkpoints:

- `checkpoint/vitcope_cls_size224_patch16_dim192_depth12_0421_235647_best.pth`
  - `best_acc=68.126`, `epoch=294`
- `checkpoint/vitcope_cls_size224_patch16_dim192_depth12_0428_015558_best.pth`
  - `best_acc=66.136`, `epoch=295`
- `checkpoint/vitcope_cls_size224_patch16_dim192_depth12_best.pth`
  - `best_acc=64.594`, `epoch=295`

For tiny downstream, prefer the `68.126` checkpoint unless a newer stronger
checkpoint is produced.

Weak generic checkpoint:

- `checkpoint/pretrained/vitcope_pretrained.pth`
  - `best_acc=61.314`, `epoch=99`

Avoid this for final downstream comparisons.

## ADE20K Segmentation Recipe

Use the paper-style 40K UPerNet setting unless explicitly running 160K:

```yaml
task: seg
bs: 16
size: 224
img_scale: [2048, 512]
test_img_scale: [2048, 512]
crop_size: 512
max_iters: 40000
n_epochs: 32  # approximate reference only
lr: 6e-5
min_lr: 0
warmup_iters: 1500
weight_decay: 0.01
opt: adamw
amp: true
drop_path_rate: 0.1
```

Implementation details:

- Backbone outputs layers `(2, 5, 8, 11)`.
- Segmentation adapter uses a MultiLevelNeck-style resize pyramid with scales
  `[4, 2, 1, 0.5]`.
- UPerNet decode head channels:
  - Tiny: `192`
  - Small: `384`
  - Base: `512`
- Auxiliary head:
  - `in_index=3`
  - `channels=256`

Recommended CoPE pretrained paths:

- Tiny: `checkpoint/vitcope_cls_size224_patch16_dim192_depth12_0421_235647_best.pth`
- Small: `checkpoint/pretrained/COPE/vitcope_cls_size224_patch16_dim384_depth12_best.pth`
- Base: `checkpoint/vitcope_cls_size224_patch16_dim768_depth12_best.pth`

## COCO Detection Recipe

Current implementation is Mask R-CNN + FPN with global-attention ViT-style
backbones. This is not fully ViTDet-equivalent because ViTDet uses window
attention in most blocks and only a few global-attention blocks.

Current safe batch sizes for `img_scale: [1333, 800]`:

- Tiny: `bs=8`, `lr=5e-5`
- Small: `bs=4`, `lr=2.5e-5`
- Base: `bs=1`, `lr=6.25e-6`

Implementation details:

- Detection adapter uses a ViTDet SimpleFeaturePyramid-style deconv/maxpool
  pyramid with scale factors `[4, 2, 1, 0.5]`.
- FPN output channels: `256`
- Anchor strides: `[4, 8, 16, 32, 64]`
- ROI featmap strides: `[4, 8, 16, 32]`

Known risk:

- Global attention at COCO resolution is memory-heavy. If `bs=4` small still
  OOMs, the next fix should be activation checkpointing or window attention,
  not just lowering batch repeatedly.
