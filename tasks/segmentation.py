# -*- coding: utf-8 -*-
"""
Semantic Segmentation Task
MMSegmentation + ViT / ViT-CoPE / ViT-SCoPE backbone.

This version:
1. Explicitly registers ViTBackbone / ViTCoPEBackbone / ViTSCoPEBackbone.
2. Aligns downstream backbone config with classification backbone naming.
3. Loads classification checkpoint into backbone with simple key prefix mapping.
4. Avoids EvalHook and TextLoggerHook collision that causes KeyError: 'data_time'.
"""

import os
import time
import torch

from mmcv import Config
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.models.builder import BACKBONES as MMSEG_BACKBONES

from models.vit_backbone import (
    ViTBackbone,
    ViTCoPEBackbone,
    ViTSCoPEBackbone,
)


def _register_backbone_once(name, module):
    if name in MMSEG_BACKBONES.module_dict:
        print(f"✅ [MMSEG Registry] {name} already registered")
        return

    MMSEG_BACKBONES.register_module(name=name, module=module)
    print(f"✅ [MMSEG Registry] registered {name}")


_register_backbone_once("ViTBackbone", ViTBackbone)
_register_backbone_once("ViTCoPEBackbone", ViTCoPEBackbone)
_register_backbone_once("ViTSCoPEBackbone", ViTSCoPEBackbone)


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _as_betas(value, default=(0.9, 0.999)):
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return tuple(float(v) for v in value)
    return default


class SegmentationTask:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.run_name = self._build_run_name()

        print("\n🔧 Segmentation Task Init")
        print(f"   model: {args.model}")
        print(f"   pretrained: {getattr(args, 'pretrained', 'NOT SET')}")

        self.use_wandb = WANDB_AVAILABLE and not getattr(args, "nowandb", False)

        self.cfg = self._build_mmseg_config()

        if hasattr(args, "seed") and args.seed is not None:
            set_random_seed(args.seed, deterministic=True)

        self.model = build_segmentor(
            self.cfg.model,
            train_cfg=self.cfg.get("train_cfg"),
            test_cfg=self.cfg.get("test_cfg"),
        )

        if hasattr(args, "pretrained") and args.pretrained:
            print(f"\n🔧 Loading pretrained weights: {args.pretrained}")
            self._load_pretrained_backbone(args.pretrained)
        else:
            print("⚠️  Training from scratch: no pretrained weights provided.")

        self.datasets = [build_dataset(self.cfg.data.train)]
        self.model.CLASSES = self.datasets[0].CLASSES

        if self.use_wandb:
            project_name = "ade20k-experiments"
            watermark = (
                f"{self.run_name}_size{args.size}_patch{args.patch}_"
                f"dim{getattr(args, 'dim', getattr(args, 'embed_dim', 'na'))}_"
                f"bs{args.bs}_lr{args.lr}"
            )
            wandb.init(project=project_name, name=watermark)
            wandb.config.update(vars(args))
        elif not WANDB_AVAILABLE and not getattr(args, "nowandb", False):
            print("WARNING: WandB not installed, skipping WandB logging.")

    # ------------------------------------------------------- #
    def _build_run_name(self):
        args = self.args
        cfg_path = getattr(args, "cfg", "")
        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0] if cfg_path else ""
        if not cfg_name:
            dim = getattr(args, "dim", getattr(args, "embed_dim", "na"))
            cfg_name = f"{args.model}_seg_dim{dim}"

        run_tag = getattr(args, "run_tag", None)
        return f"{cfg_name}_{run_tag}" if run_tag else cfg_name

    # ------------------------------------------------------- #
    def _build_mmseg_config(self):
        args = self.args
        cfg = Config()

        cfg.model = self._get_upernet_config()
        cfg.data = self._get_data_config()

        cfg.optimizer = dict(
            type="AdamW",
            lr=args.lr,
            betas=_as_betas(getattr(args, "betas", None)),
            weight_decay=getattr(args, "weight_decay", 0.01),
            paramwise_cfg=dict(
                custom_keys={
                    "pos_embedding": dict(decay_mult=0.0),
                    "cls_token": dict(decay_mult=0.0),
                    "norm": dict(decay_mult=0.0),
                }
            ),
        )

        cfg.optimizer_config = dict(grad_clip=None)

        if bool(getattr(args, "amp", False)):
            cfg.fp16 = dict(loss_scale="dynamic")
            print("✅ MMSeg fp16 enabled: cfg.fp16 = dynamic loss scale")

        max_iters = getattr(args, "max_iters", None)
        if max_iters is None:
            max_iters = int(getattr(args, "n_epochs", 32) * 1000)

        warmup_iters = getattr(args, "warmup_iters", None)
        if warmup_iters is None:
            warmup_iters = int(getattr(args, "warmup_epochs", 0) * 1000)

        cfg.runner = dict(
            type="IterBasedRunner",
            max_iters=max_iters,
        )

        cfg.lr_config = dict(
            policy="poly",
            warmup="linear" if warmup_iters > 0 else None,
            warmup_iters=warmup_iters,
            warmup_ratio=1e-6,
            power=1.0,
            min_lr=getattr(args, "min_lr", 0.0),
            by_epoch=False,
        )

        model_name = self.run_name
        task_name = args.task or "seg"

        cfg.checkpoint_config = dict(
            by_epoch=False,
            interval=int(getattr(args, "checkpoint_interval", 5000)),
            filename_tmpl=f"{model_name}_{task_name}_iter_{{}}.pth",
            max_keep_ckpts=3,
        )

        # --------------------------------------------------- #
        # IMPORTANT FIX:
        # Avoid EvalHook and TextLoggerHook triggering at same iter.
        # Old mmseg/mmcv can throw KeyError: 'data_time'.
        # --------------------------------------------------- #
        log_interval = int(getattr(args, "log_interval", 100))
        eval_interval = int(getattr(args, "eval_interval", 2001))

        if eval_interval % log_interval == 0:
            eval_interval += 1

        cfg.evaluation = dict(
            interval=eval_interval,
            metric="mIoU",
            pre_eval=True,
            save_best="mIoU",
            classwise=False,
        )

        cfg.log_config = dict(
            interval=log_interval,
            hooks=[
                dict(type="TextLoggerHook", by_epoch=False),
            ],
        )

        cfg.custom_hooks = []

        cfg.dist_params = dict(backend="nccl")
        cfg.log_level = "INFO"
        cfg.work_dir = f"./work_dirs/{self.run_name}_upernet"
        cfg.load_from = None
        cfg.resume_from = None
        cfg.workflow = [("train", 1)]
        cfg.gpu_ids = [0]
        cfg.cudnn_benchmark = True
        cfg.seed = getattr(args, "seed", None)

        print("\n✅ MMSeg config summary")
        print(f"   max_iters: {max_iters}")
        print(f"   warmup_iters: {warmup_iters}")
        print(f"   log_interval: {log_interval}")
        print(f"   eval_interval: {eval_interval}")
        print(f"   lr: {args.lr}")
        print(f"   weight_decay: {getattr(args, 'weight_decay', 0.01)}")
        print(f"   crop_size: {getattr(args, 'crop_size', 512)}")
        print(f"   img_scale: {getattr(args, 'img_scale', [2048, 512])}")
        print(f"   test_img_scale: {getattr(args, 'test_img_scale', [2048, 512])}")
        default_head_channels = min(self._get_backbone_out_channels()[0], 512)
        print(f"   seg_head_dim: {getattr(args, 'seg_head_dim', default_head_channels)}")
        print(f"   seg_aux_dim: {getattr(args, 'seg_aux_dim', 256)}")
        print(f"   work_dir: {cfg.work_dir}")

        return cfg

    # ------------------------------------------------------- #
    def _get_upernet_config(self):
        backbone_cfg = self._get_backbone_config()
        in_channels = self._get_backbone_out_channels()
        default_head_channels = min(int(in_channels[0]), 512)
        head_channels = int(getattr(self.args, "seg_head_dim", default_head_channels))

        decode_head_cfg = dict(
            type="UPerHead",
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=head_channels,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                loss_weight=1.0,
            ),
        )

        auxiliary_head_cfg = dict(
            type="FCNHead",
            in_channels=in_channels[3],
            in_index=3,
            channels=int(getattr(self.args, "seg_aux_dim", 256)),
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                loss_weight=0.4,
            ),
        )

        model = dict(
            type="EncoderDecoder",
            pretrained=None,
            backbone=backbone_cfg,
            decode_head=decode_head_cfg,
            auxiliary_head=auxiliary_head_cfg,
            train_cfg=dict(),
            test_cfg=dict(mode="whole"),
        )

        return model

    # ------------------------------------------------------- #
    def _get_backbone_config(self):
        args = self.args

        common = dict(
            image_size=args.size,
            patch_size=args.patch,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_dim=args.mlp_dim,
            dim_head=getattr(args, "dim_head", 64),
            drop_path_rate=float(getattr(args, "drop_path_rate", 0.0)),
            out_indices=tuple(getattr(args, "out_indices", (2, 5, 8, 11))),
            fpn_adapter_style="resize",
        )

        if args.model == "swin":
            return dict(
                type="SwinTransformer",
                embed_dim=args.embed_dim,
                depths=args.depths,
                num_heads=args.num_heads,
                window_size=args.window_size,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=float(getattr(args, "drop_path_rate", 0.0)),
                ape=False,
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                use_checkpoint=False,
            )

        if args.model == "vit":
            return dict(
                type="ViTBackbone",
                **common,
            )

        if args.model == "vitcope":
            return dict(
                type="ViTCoPEBackbone",
                use_cls_token=bool(getattr(args, "use_cls_token", False)),
                **common,
            )

        if args.model == "vitscope":
            return dict(
                type="ViTSCoPEBackbone",
                **common,
            )

        raise ValueError(f"Unknown segmentation backbone model: {args.model}")

    # ------------------------------------------------------- #
    def _get_backbone_out_channels(self):
        args = self.args

        if args.model == "swin":
            base_dim = args.embed_dim
            return [base_dim * (2 ** i) for i in range(4)]

        return [args.dim] * 4

    # ------------------------------------------------------- #
    def _get_data_config(self):
        args = self.args

        def _pair_from_arg(name, default):
            value = getattr(args, name, default)
            if isinstance(value, (tuple, list)):
                if len(value) != 2:
                    raise ValueError(f"{name} must have two values, got {value}")
                return (int(value[0]), int(value[1]))
            return (int(value), int(value))

        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
        )

        # Scratching follows XCiT/Swin ADE20K UPerNet protocol:
        # 512x512 crop with long-side img_scale (2048, 512), while args.size
        # still describes the ImageNet pretraining/backbone setup.
        crop_size = _pair_from_arg("crop_size", 512)
        img_scale = _pair_from_arg("img_scale", (2048, 512))
        test_img_scale = _pair_from_arg("test_img_scale", img_scale)

        train_pipeline = [
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", reduce_zero_label=True),
            dict(
                type="Resize",
                img_scale=img_scale,
                ratio_range=(0.5, 2.0),
            ),
            dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
            dict(type="RandomFlip", prob=0.5),
            dict(type="PhotoMetricDistortion"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_semantic_seg"]),
        ]

        test_pipeline = [
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=test_img_scale,
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="Pad", size_divisor=int(getattr(args, "patch", 16))),
                    dict(type="Normalize", **img_norm_cfg),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ]

        workers_per_gpu = getattr(args, "workers_per_gpu", None)
        if workers_per_gpu is None:
            workers_per_gpu = 4

        data = dict(
            samples_per_gpu=args.bs,
            workers_per_gpu=int(workers_per_gpu),
            train=dict(
                type="ADE20KDataset",
                data_root=args.data_dir,
                img_dir="images/training",
                ann_dir="annotations/training",
                pipeline=train_pipeline,
            ),
            val=dict(
                type="ADE20KDataset",
                data_root=args.data_dir,
                img_dir="images/validation",
                ann_dir="annotations/validation",
                pipeline=test_pipeline,
            ),
            test=dict(
                type="ADE20KDataset",
                data_root=args.data_dir,
                img_dir="images/validation",
                ann_dir="annotations/validation",
                pipeline=test_pipeline,
            ),
        )

        return data

    # ------------------------------------------------------- #
    def _extract_state_dict(self, checkpoint):
        if "model" in checkpoint:
            print("   Using checkpoint['model']")
            return checkpoint["model"]

        if "state_dict" in checkpoint:
            print("   Using checkpoint['state_dict']")
            return checkpoint["state_dict"]

        print("   Using checkpoint directly")
        return checkpoint

    # ------------------------------------------------------- #
    def _load_pretrained_backbone(self, pretrained_path):
        print(f"\n{'=' * 60}")
        print("🔧 PRETRAINED LOADING DEBUG")
        print(f"{'=' * 60}")
        print(f"📦 Pretrained path: {pretrained_path}")
        print(f"   File exists: {os.path.exists(pretrained_path)}")

        if not os.path.exists(pretrained_path):
            print("⚠️  Pretrained weights not found. Training from scratch.")
            print(f"{'=' * 60}\n")
            return

        checkpoint = torch.load(pretrained_path, map_location="cpu")
        print("✅ Checkpoint loaded")
        print(f"   Checkpoint keys: {list(checkpoint.keys())[:5]}")

        pretrained_dict = self._extract_state_dict(checkpoint)

        print(f"   Total pretrained keys: {len(pretrained_dict)}")
        print(f"   Sample keys: {list(pretrained_dict.keys())[:5]}")

        backbone_dict = {}
        skipped = []

        for k, v in pretrained_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("net."):
                k = k[len("net."):]
            if k.startswith("model."):
                k = k[len("model."):]

            if (
                k.startswith("mlp_head.")
                or k.startswith("head.")
                or k.startswith("fc.")
                or "classifier" in k
            ):
                skipped.append(k)
                continue

            new_key = k if k.startswith("backbone.") else f"backbone.{k}"
            backbone_dict[new_key] = v

        model_dict = self.model.state_dict()

        matched = {}
        unmatched = []

        for k, v in backbone_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                matched[k] = v
            else:
                if k in model_dict:
                    unmatched.append(
                        f"{k} shape mismatch: ckpt={tuple(v.shape)} model={tuple(model_dict[k].shape)}"
                    )
                else:
                    unmatched.append(f"{k} not in model")

        print("\n📊 Matching results")
        print(f"   After mapping: {len(backbone_dict)} keys")
        print(f"   Matched: {len(matched)} keys")
        print(f"   Unmatched: {len(unmatched)} keys")
        print(f"   Skipped classifier keys: {len(skipped)}")

        if len(matched) > 0:
            print("\n✅ Sample matched keys:")
            for k in list(matched.keys())[:5]:
                print(f"     {k}")

        if len(unmatched) > 0:
            print("\n⚠️ Sample unmatched keys:")
            for k in unmatched[:10]:
                print(f"     {k}")

        model_dict.update(matched)
        self.model.load_state_dict(model_dict, strict=False)

        match_rate = 100.0 * len(matched) / max(1, len(backbone_dict))
        print(f"\n✅ FINAL: Loaded {len(matched)}/{len(backbone_dict)} backbone keys ({match_rate:.1f}%)")

        min_match_rate = float(getattr(self.args, "min_pretrained_match_rate", 80.0))
        if match_rate < min_match_rate:
            print("\n❌ ERROR: Low match rate. Check whether classification and backbone structures are aligned.")
            model_backbone_keys = [k for k in model_dict.keys() if k.startswith("backbone.")]
            print(f"   Sample ckpt mapped key: {list(backbone_dict.keys())[0] if backbone_dict else 'NONE'}")
            print(f"   Sample model key: {model_backbone_keys[0] if model_backbone_keys else 'NONE'}")
            raise RuntimeError(
                f"Pretrained backbone match rate {match_rate:.1f}% is below "
                f"required {min_match_rate:.1f}% for {pretrained_path}"
            )

        print(f"{'=' * 60}\n")

    # ------------------------------------------------------- #
    def train(self):
        print(f"🚀 Start training {self.args.model} + UPerNet on ADE20K\n")

        train_segmentor(
            self.model,
            self.datasets,
            self.cfg,
            distributed=False,
            validate=True,
            timestamp=time.strftime("%Y%m%d_%H%M%S", time.localtime()),
            meta=dict(),
        )

        print("\n✅ Segmentation training finished!\n")

        if self.use_wandb:
            wandb.finish()