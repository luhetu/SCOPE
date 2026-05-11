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

Known tiny recipe notes:

- The `68.126` run is `/home3/dnrx52/Size/logs/cope_tiny904820.out`.
- It was run before the newer classification diagnostics and should be treated
  like the legacy CoPE recipe: label smoothing on, but no effective Mixup/CutMix,
  random erasing, repeated augmentation, or DropPath.
- The `65.998` run `/home3/dnrx52/Size/logs/cope_tiny910847.out` used the newer
  DeiT-style recipe and actually ran `drop_path_rate=0.1`; it is not evidence
  for the current `drop_path_rate=0.0` setting.

Weak generic checkpoint:

- `checkpoint/pretrained/vitcope_pretrained.pth`
  - `best_acc=61.314`, `epoch=99`

Avoid this for final downstream comparisons.

## ADE20K Segmentation Recipe

Use the paper-style 40K UPerNet setting unless explicitly running 160K:

```yaml
task: seg
bs: 16
size: 224  # ImageNet pretraining grid; ADE20K crop is still 512
img_scale: [2048, 512]
test_img_scale: [2048, 512]
crop_size: 512
max_iters: 40000
n_epochs: 32  # approximate reference only
lr: 6e-5
min_lr: 0
warmup_iters: 1500
weight_decay: 0.01
layer_decay_rate: 0.75
opt: adamw
amp: true
drop_path_rate: 0.1
```

For `seg_vitscope_tiny.yaml`, use `drop_path_rate: 0.0` as the current
SCoPE-tiny strong recipe. The 40K logs show SCoPE learns faster early, but the
`0.1` drop path run was slightly overtaken late by baseline ViT, so tiny SCoPE
should preserve the pretrained SCoPE path more aggressively.

For tiny ViT/CoPE/SCoPE, use `seg_neck_style: internal_resize` while diagnosing
the mIoU gap. This restores the older high-performing path that reached
`31.21` mIoU on `seg_vit_tiny_911559`, where the backbone emits resized pyramid
features directly instead of going through the newer external `MultiLevelNeck`.
Use `layer_decay_rate: 0.75` for ViT-family segmentation fine-tuning so shallow
pretrained layers are not updated as aggressively as the decoder/head.

Implementation details:

- Backbone outputs layers `(2, 5, 8, 11)`.
- Tiny configs currently set `seg_neck_style: internal_resize` to recover the
  older high-performing segmentation path before re-testing the official neck.
- Segmentation uses the official MMSeg ViT+UPerNet topology:
  backbone outputs four same-resolution features, then `MultiLevelNeck` resizes
  them with scales `[4, 2, 1, 0.5]`.
- Keep segmentation `size=224` so the backbone is initialized on the ImageNet
  pretraining grid and pretrained position tables load exactly. ADE20K input
  resolution is controlled by `crop_size=512` and `img_scale=[2048, 512]`.
- ViT `pos_embedding` and CoPE/SCoPE `cope.pos_emb` are resized dynamically in
  the backbone forward pass when the 512 crop produces more patch tokens.
- `pos_embedding`, `pos_emb`, `cls_token`, `hk_gate`, `lam`, and normalization
  parameters use `decay_mult=0.0` during downstream fine-tuning.
- Classification final `norm.weight/bias` are remapped to the last segmentation
  backbone output norm (`backbone.norms[-1]`) instead of being silently skipped.
- Backbone-internal FPN adapters are disabled for segmentation
  (`fpn_adapter_style="identity"`). Detection still uses the ViTDet-style simple
  feature pyramid separately.
- `MultiLevelNeck.out_channels` defaults to the backbone dim:
  - Tiny: `192`
  - Small: `384`
  - Base: `768`
- UPerNet decode head channels: `512`
- Auxiliary head:
  - `in_index=3`
  - `channels=256`

Recommended CoPE pretrained paths:

- Tiny: `checkpoint/vitcope_cls_size224_patch16_dim192_depth12_0421_235647_best.pth`
- Small: `checkpoint/pretrained/COPE/vitcope_cls_size224_patch16_dim384_depth12_best.pth`
- Base: `checkpoint/vitcope_cls_size224_patch16_dim768_depth12_best.pth`

Recommended SCoPE pretrained paths:

- Tiny: `checkpoint/pretrained/SCOPE/scopetiny.pth`
  - `best_acc=70.946`, same as `checkpoint/vitscope_nocls_cls_size224_patch16_dim192_depth12_0422_111127_best.pth`
- Small: `checkpoint/pretrained/SCOPE/scopesmall.pth`
  - `best_acc=76.414`
- Base: `checkpoint/pretrained/SCOPE/scopebasedp2.pth`
  - `best_acc=78.072`

Recent tiny ADE20K observations:

- `seg_vit_tiny_912581.err`: best logged `mIoU=28.02`, `mAcc=37.04`, `aAcc=75.26` at iter `38019`.
- `seg_vitscope_tiny_912582.err`: best logged `mIoU=27.53`, `mAcc=36.42`, `aAcc=74.60` at iter `38019`.
- `seg_vitscope_tiny_910848.err`: best logged `mIoU=28.38`, `mAcc=37.82`, `aAcc=74.60` at iter `38019`, before the latest official-neck cleanup.
- `seg_vitscope_tiny_914727.err`: `seg_neck_style=internal_resize`,
  `size=224`, `crop_size=512`, `drop_path_rate=0.0`; best logged
  `mIoU=29.40`, `mAcc=38.62`, `aAcc=74.74` at iter `38019`.

## COCO Detection Recipe

Current implementation is Mask R-CNN + FPN with global-attention ViT-style
backbones. This is not fully ViTDet-equivalent because ViTDet uses window
attention in most blocks and only a few global-attention blocks.

The detector neck should follow the official ViTDet-style topology:

- Backbone outputs four same-resolution block features `(2, 5, 8, 11)`.
- Backbone-internal adapters are disabled (`fpn_adapter_style="identity"`).
- `SimpleFeaturePyramid` is the explicit MMDetection neck.
- Scale factors: `[4, 2, 1, 0.5]`.
- Neck output channels: `256`.
- A final max-pooled level provides the fifth RPN level.

Current safe batch sizes for `img_scale: [1333, 800]`:

- Tiny: `bs=8`, `lr=5e-5`
- Small: `bs=4`, `lr=2.5e-5`
- Base: `bs=1`, `lr=6.25e-6`

Implementation details:

- FPN output channels: `256`
- Anchor strides: `[4, 8, 16, 32, 64]`
- ROI featmap strides: `[4, 8, 16, 32]`
- Old MMCV's `LN` in `ConvModule` is not safe for NCHW feature maps; use GN in
  `SimpleFeaturePyramid` to avoid the `normalized_shape=[256]` crash.

Recent COCO detection observations:

- `det_vitscope_tiny914745.err`: failed before validation on `gpu13` with
  `RuntimeError: CUDA error: no kernel image is available for execution on the device`.
  No AP result was produced; this is an environment/GPU compatibility issue, not
  a model-quality result.
- `det_vitscope_small912401.err`: finished 12 epochs with
  `configs/detection_vitscope_small.yaml`, `bs=4`, `lr=2.5e-5`, pretrained
  `checkpoint/pretrained/SCOPE/scopesmall.pth` (`best_acc=76.414`).
  Best `bbox_mAP=37.1` at epoch 10; final epoch 12 also reports
  `bbox_mAP=37.1`, `bbox_mAP_50=60.0`, `bbox_mAP_75=39.2`,
  `segm_mAP=34.1`, `segm_mAP_50=56.6`, `segm_mAP_75=35.3`.

Known risk:

- Global attention at COCO resolution is memory-heavy. If `bs=4` small still
  OOMs, the next fix should be activation checkpointing or window attention,
  not just lowering batch repeatedly.
