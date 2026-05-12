# -*- coding: utf-8 -*-
"""
Semantic Segmentation Task
MMSegmentation + ViT / ViT-CoPE / ViT-SCoPE backbone.

Fixes in this version:
1. Registers ViT / ViT-CoPE / ViT-SCoPE backbones explicitly.
2. Uses crop_size/backbone_size for dense-prediction backbone initialization.
3. Loads ImageNet classification checkpoints into the segmentation backbone.
4. Remaps final norm from both norm.* and mlp_head.0.* to backbone.norms[-1].
5. Keeps CoPE / ViT position tables by interpolation instead of dropping them.
6. Pads ADE20K validation images to 32 for stable multi-level ViT features.
"""

import os
import time
import torch
import torch.nn.functional as F

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmseg.apis import set_random_seed, single_gpu_test, train_segmentor
from mmseg.datasets import build_dataloader, build_dataset
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


def _as_pair(value, default):
    if value is None:
        value = default
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"pair value must have two elements, got {value}")
        return (int(value[0]), int(value[1]))
    return (int(value), int(value))


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
            paramwise_cfg=self._get_paramwise_cfg(),
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

        cfg.runner = dict(type="IterBasedRunner", max_iters=max_iters)
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

        # Avoid old mmcv EvalHook/TextLoggerHook same-iter data_time KeyError.
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
            hooks=[dict(type="TextLoggerHook", by_epoch=False)],
        )

        cfg.custom_hooks = []
        cfg.final_eval = bool(getattr(args, "final_eval", True))
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
        print(f"   layer_decay_rate: {getattr(args, 'layer_decay_rate', 1.0)}")
        print(f"   crop_size: {getattr(args, 'crop_size', 512)}")
        print(f"   img_scale: {getattr(args, 'img_scale', [2048, 512])}")
        print(f"   test_img_scale: {getattr(args, 'test_img_scale', [2048, 512])}")
        print(f"   backbone_image_size: {self._get_backbone_image_size()}")
        print(f"   seg_head_dim: {getattr(args, 'seg_head_dim', self._get_default_seg_head_dim())}")
        print(f"   seg_aux_dim: {getattr(args, 'seg_aux_dim', 256)}")
        print(f"   seg_aux_in_index: {getattr(args, 'seg_aux_in_index', 2)}")
        print(f"   seg_norm_type: {getattr(args, 'seg_norm_type', 'BN')}")
        print(f"   seg_neck_dim: {getattr(args, 'seg_neck_dim', self._get_backbone_out_channels()[0])}")
        print(f"   final_eval: {cfg.final_eval}")
        print(f"   work_dir: {cfg.work_dir}")

        return cfg

    # ------------------------------------------------------- #
    def _get_paramwise_cfg(self):
        args = self.args
        layer_decay_rate = float(getattr(args, "layer_decay_rate", 1.0))

        custom_keys = {
            "backbone.pos_embedding": dict(decay_mult=0.0),
            "cope.pos_emb": dict(decay_mult=0.0),
            "backbone.cls_token": dict(decay_mult=0.0),
            "backbone.hk_gate": dict(decay_mult=0.0),
            ".fn.lam": dict(decay_mult=0.0),
        }

        if args.model != "swin" and layer_decay_rate < 1.0:
            depth = int(getattr(args, "depth", 12))
            embed_lr_mult = layer_decay_rate ** depth
            custom_keys.update({
                "backbone.to_patch_embedding": dict(lr_mult=embed_lr_mult),
                "backbone.to_patch": dict(lr_mult=embed_lr_mult),
                "backbone.pos_embedding": dict(lr_mult=embed_lr_mult, decay_mult=0.0),
                "backbone.cls_token": dict(lr_mult=embed_lr_mult, decay_mult=0.0),
                "backbone.hk_gate": dict(lr_mult=embed_lr_mult, decay_mult=0.0),
            })
            for layer_idx in range(depth):
                lr_mult = layer_decay_rate ** (depth - layer_idx - 1)
                layer_prefix = f"backbone.transformer.layers.{layer_idx}"
                custom_keys[f"{layer_prefix}.0.fn"] = dict(lr_mult=lr_mult)
                custom_keys[f"{layer_prefix}.1.fn"] = dict(lr_mult=lr_mult)
                custom_keys[f"{layer_prefix}.0.fn.cope.pos_emb"] = dict(lr_mult=lr_mult, decay_mult=0.0)
                custom_keys[f"{layer_prefix}.0.fn.lam"] = dict(lr_mult=lr_mult, decay_mult=0.0)
                custom_keys[f"{layer_prefix}.0.norm"] = dict(lr_mult=lr_mult, decay_mult=0.0)
                custom_keys[f"{layer_prefix}.1.norm"] = dict(lr_mult=lr_mult, decay_mult=0.0)
            custom_keys["backbone.norms"] = dict(lr_mult=1.0, decay_mult=0.0)

        return dict(custom_keys=custom_keys, norm_decay_mult=0.0)

    # ------------------------------------------------------- #
    def _get_upernet_config(self):
        backbone_cfg = self._get_backbone_config()
        in_channels = self._get_backbone_out_channels()
        default_head_channels = self._get_default_seg_head_dim()
        head_channels = int(getattr(self.args, "seg_head_dim", default_head_channels))
        norm_cfg = self._get_seg_norm_cfg()
        neck_cfg = None
        seg_neck_style = str(getattr(self.args, "seg_neck_style", "none")).lower()

        if self.args.model != "swin" and seg_neck_style in ("multilevel", "external"):
            neck_out_channels = int(getattr(self.args, "seg_neck_dim", in_channels[0]))
            neck_cfg = dict(
                type="MultiLevelNeck",
                in_channels=in_channels,
                out_channels=neck_out_channels,
                scales=[4, 2, 1, 0.5],
            )
            in_channels = [neck_out_channels] * 4

        decode_head_cfg = dict(
            type="UPerHead",
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=head_channels,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        )

        aux_in_index = int(getattr(self.args, "seg_aux_in_index", 2))
        if aux_in_index < 0:
            aux_in_index += len(in_channels)
        if aux_in_index < 0 or aux_in_index >= len(in_channels):
            raise ValueError(
                f"seg_aux_in_index={aux_in_index} is out of range for {len(in_channels)} feature maps"
            )

        auxiliary_head_cfg = dict(
            type="FCNHead",
            in_channels=in_channels[aux_in_index],
            in_index=aux_in_index,
            channels=int(getattr(self.args, "seg_aux_dim", 256)),
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
        )

        return dict(
            type="EncoderDecoder",
            pretrained=None,
            backbone=backbone_cfg,
            neck=neck_cfg,
            decode_head=decode_head_cfg,
            auxiliary_head=auxiliary_head_cfg,
            train_cfg=dict(),
            test_cfg=dict(mode="whole"),
        )

    # ------------------------------------------------------- #
    def _get_default_seg_head_dim(self):
        return 512

    # ------------------------------------------------------- #
    def _get_seg_norm_cfg(self):
        norm_type = str(getattr(self.args, "seg_norm_type", "BN"))
        if norm_type.upper() == "GN":
            return dict(type="GN", num_groups=32, requires_grad=True)
        return dict(type=norm_type, requires_grad=True)

    # ------------------------------------------------------- #
    def _get_backbone_image_size(self):
        args = self.args
        # args.size stays as ImageNet pretraining size. Dense prediction backbone
        # tables should be initialized at crop_size, usually 512, then checkpoints
        # are interpolated once at load time.
        value = getattr(args, "backbone_size", getattr(args, "crop_size", args.size))
        if isinstance(value, (tuple, list)):
            return tuple(int(v) for v in value)
        return int(value)

    # ------------------------------------------------------- #
    def _get_backbone_config(self):
        args = self.args

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

        common = dict(
            image_size=self._get_backbone_image_size(),
            patch_size=args.patch,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_dim=args.mlp_dim,
            dim_head=getattr(args, "dim_head", 64),
            drop_path_rate=float(getattr(args, "drop_path_rate", 0.0)),
            out_indices=tuple(getattr(args, "out_indices", (3, 5, 7, 11))),
            fpn_adapter_style=self._get_vit_fpn_adapter_style(),
        )

        if args.model == "vit":
            return dict(type="ViTBackbone", **common)

        if args.model == "vitcope":
            return dict(
                type="ViTCoPEBackbone",
                use_cls_token=bool(getattr(args, "use_cls_token", False)),
                **common,
            )

        if args.model == "vitscope":
            return dict(
                type="ViTSCoPEBackbone",
                use_cls_token=bool(getattr(args, "use_cls_token", True)),
                **common,
            )

        raise ValueError(f"Unknown segmentation backbone model: {args.model}")

    # ------------------------------------------------------- #
    def _get_vit_fpn_adapter_style(self):
        seg_neck_style = str(getattr(self.args, "seg_neck_style", "xcit_fpn")).lower()
        if seg_neck_style in ("internal_resize", "resize"):
            return "resize"
        if seg_neck_style in ("xcit_fpn", "simple_fpn", "official", "official_xcit"):
            return "simple_fpn"
        return "identity"

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

        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
        )

        crop_size = _as_pair(getattr(args, "crop_size", None), 512)
        img_scale = _as_pair(getattr(args, "img_scale", None), (2048, 512))
        test_img_scale = _as_pair(getattr(args, "test_img_scale", None), img_scale)

        train_pipeline = [
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", reduce_zero_label=True),
            dict(type="Resize", img_scale=img_scale, ratio_range=(0.5, 2.0)),
            dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
            dict(type="RandomFlip", prob=0.5),
            dict(type="PhotoMetricDistortion"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_semantic_seg"]),
        ]

        # simple_fpn has a stride-32 output. Pad validation images to 32 so the
        # patch grid is even before the final 0.5-scale feature.
        test_pipeline = [
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=test_img_scale,
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(type="Pad", size_divisor=32),
                    dict(type="Normalize", **img_norm_cfg),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ]

        return dict(
            samples_per_gpu=args.bs,
            workers_per_gpu=int(getattr(args, "workers_per_gpu", 4)),
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
    def _map_pretrained_key(self, key):
        raw_key = key[len("backbone."):] if key.startswith("backbone.") else key
        last_norm_idx = len(getattr(self.args, "out_indices", (3, 5, 7, 11))) - 1

        # ViT / CoPE final norm lives in mlp_head.0.*.
        if raw_key in ("mlp_head.0.weight", "mlp_head.0.bias"):
            suffix = raw_key.rsplit(".", 1)[1]
            return f"backbone.norms.{last_norm_idx}.{suffix}", "final_norm"

        # SCoPE final norm lives in norm.*.
        if raw_key in ("norm.weight", "norm.bias"):
            suffix = raw_key.split(".", 1)[1]
            return f"backbone.norms.{last_norm_idx}.{suffix}", "final_norm"

        if (
            raw_key.startswith("mlp_head.")
            or raw_key.startswith("head.")
            or raw_key.startswith("fc.")
            or "classifier" in raw_key
        ):
            return None, "skip_head"

        return (key if key.startswith("backbone.") else f"backbone.{key}"), "backbone"

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
        remapped_final_norm = []

        for k, v in pretrained_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("net."):
                k = k[len("net."):]
            if k.startswith("model."):
                k = k[len("model."):]

            new_key, kind = self._map_pretrained_key(k)
            if kind == "skip_head":
                skipped.append(k)
                continue
            if kind == "final_norm":
                remapped_final_norm.append(f"{k} -> {new_key}")
            backbone_dict[new_key] = v

        model_dict = self.model.state_dict()
        matched = {}
        unmatched = []
        interpolated = []

        for k, v in backbone_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                matched[k] = v
            elif (
                k in model_dict
                and k.endswith("pos_embedding")
                and v.ndim == 3
                and model_dict[k].ndim == 3
                and v.shape[-1] == model_dict[k].shape[-1]
            ):
                matched[k] = self._resize_token_position_embedding(v, model_dict[k])
                interpolated.append(f"{k}: {tuple(v.shape)} -> {tuple(model_dict[k].shape)}")
            elif (
                k in model_dict
                and k.endswith("cope.pos_emb")
                and v.ndim == 3
                and model_dict[k].ndim == 3
                and v.shape[:2] == model_dict[k].shape[:2]
            ):
                matched[k] = F.interpolate(
                    v.float(),
                    size=model_dict[k].shape[-1],
                    mode="linear",
                    align_corners=False,
                ).to(dtype=model_dict[k].dtype)
                interpolated.append(f"{k}: {tuple(v.shape)} -> {tuple(model_dict[k].shape)}")
            else:
                if k in model_dict:
                    unmatched.append(f"{k} shape mismatch: ckpt={tuple(v.shape)} model={tuple(model_dict[k].shape)}")
                else:
                    unmatched.append(f"{k} not in model")

        print("\n📊 Matching results")
        print(f"   After mapping: {len(backbone_dict)} keys")
        print(f"   Matched: {len(matched)} keys")
        print(f"   Interpolated position tables: {len(interpolated)}")
        print(f"   Unmatched: {len(unmatched)} keys")
        print(f"   Skipped classifier keys: {len(skipped)}")
        print(f"   Remapped final norm keys: {len(remapped_final_norm)}")

        if matched:
            print("\n✅ Sample matched keys:")
            for k in list(matched.keys())[:5]:
                print(f"     {k}")
        if unmatched:
            print("\n⚠️ Sample unmatched keys:")
            for k in unmatched[:10]:
                print(f"     {k}")
        if remapped_final_norm:
            print("\n🔁 Remapped classification final norm:")
            for k in remapped_final_norm[:6]:
                print(f"     {k}")
        if interpolated:
            print("\n🔁 Sample interpolated position tables:")
            for k in interpolated[:6]:
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
                f"Pretrained backbone match rate {match_rate:.1f}% is below required {min_match_rate:.1f}% "
                f"for {pretrained_path}"
            )

        print(f"{'=' * 60}\n")

    # ------------------------------------------------------- #
    def _resize_token_position_embedding(self, source, target):
        source_tokens = source.shape[1]
        target_tokens = target.shape[1]
        source_patch_with_cls = source_tokens - 1
        target_patch_with_cls = target_tokens - 1
        has_cls = (
            int(source_patch_with_cls ** 0.5) ** 2 == source_patch_with_cls
            and int(target_patch_with_cls ** 0.5) ** 2 == target_patch_with_cls
        )

        if has_cls:
            source_cls = source[:, :1]
            source_patch = source[:, 1:]
            target_patch_tokens = target_tokens - 1
        else:
            source_cls = None
            source_patch = source
            target_patch_tokens = target_tokens

        old_size = int(source_patch.shape[1] ** 0.5)
        new_size = int(target_patch_tokens ** 0.5)
        if old_size * old_size != source_patch.shape[1] or new_size * new_size != target_patch_tokens:
            return source

        source_patch = source_patch.reshape(1, old_size, old_size, source.shape[-1]).permute(0, 3, 1, 2)
        source_patch = F.interpolate(
            source_patch.float(),
            size=(new_size, new_size),
            mode="bicubic",
            align_corners=False,
        )
        source_patch = source_patch.permute(0, 2, 3, 1).reshape(1, target_patch_tokens, source.shape[-1])

        if source_cls is not None:
            resized = torch.cat([source_cls.float(), source_patch], dim=1)
        else:
            resized = source_patch
        return resized.to(dtype=target.dtype)

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

        if self.cfg.get("final_eval", True):
            self._run_final_eval()

        print("\n✅ Segmentation training finished!\n")
        if self.use_wandb:
            wandb.finish()

    # ------------------------------------------------------- #
    def _run_final_eval(self):
        print("\n🚀 Running final ADE20K evaluation after max_iters")
        print("   This evaluates the in-memory final model weights.")

        val_dataset = build_dataset(self.cfg.data.val, dict(test_mode=True))
        val_loader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
        )

        eval_model = MMDataParallel(
            self.model.cuda(self.cfg.gpu_ids[0]),
            device_ids=self.cfg.gpu_ids,
        )
        eval_model.CLASSES = val_dataset.CLASSES
        if hasattr(val_dataset, "PALETTE"):
            eval_model.PALETTE = val_dataset.PALETTE

        results = single_gpu_test(eval_model, val_loader, show=False)
        eval_results = val_dataset.evaluate(
            results,
            metric=self.cfg.evaluation.get("metric", "mIoU"),
            logger=None,
        )

        print("\n" + "=" * 60)
        print("FINAL SEGMENTATION EVALUATION RESULTS")
        print("=" * 60)
        for key, value in eval_results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("=" * 60 + "\n")

        if self.use_wandb:
            wandb.log({f"final_{k}": v for k, v in eval_results.items() if isinstance(v, (int, float))})
