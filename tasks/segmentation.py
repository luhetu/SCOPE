# -*- coding: utf-8 -*-
"""Semantic segmentation task: MMSegmentation + ViT / ViT-CoPE / ViT-SCoPE."""

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

from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone


def _register_backbone_once(name, module):
    if name not in MMSEG_BACKBONES.module_dict:
        MMSEG_BACKBONES.register_module(name=name, module=module)
        print(f"✅ [MMSEG Registry] registered {name}")
    else:
        print(f"✅ [MMSEG Registry] {name} already registered")


_register_backbone_once("ViTBackbone", ViTBackbone)
_register_backbone_once("ViTCoPEBackbone", ViTCoPEBackbone)
_register_backbone_once("ViTSCoPEBackbone", ViTSCoPEBackbone)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _as_betas(value, default=(0.9, 0.999)):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return tuple(float(v) for v in value)
    return default


def _as_pair(value, default):
    if value is None:
        value = default
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Expected pair value, got {value}")
        return (int(value[0]), int(value[1]))
    return (int(value), int(value))


def _arg(args, name, default):
    value = getattr(args, name, default)
    return default if value is None else value


class SegmentationTask:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.run_name = self._build_run_name()

        print("\n🔧 Segmentation Task Init")
        print(f"   model: {args.model}")
        print(f"   pretrained: {getattr(args, 'pretrained', 'NOT SET')}")

        self.use_wandb = WANDB_AVAILABLE and not _arg(args, "nowandb", False)
        self.cfg = self._build_mmseg_config()

        if getattr(args, "seed", None) is not None:
            set_random_seed(args.seed, deterministic=True)

        self.model = build_segmentor(
            self.cfg.model,
            train_cfg=self.cfg.get("train_cfg"),
            test_cfg=self.cfg.get("test_cfg"),
        )

        if getattr(args, "pretrained", None):
            print(f"\n🔧 Loading pretrained weights: {args.pretrained}")
            self._load_pretrained_backbone(args.pretrained)
        else:
            print("⚠️  Training from scratch: no pretrained weights provided.")

        self.datasets = [build_dataset(self.cfg.data.train)]
        self.model.CLASSES = self.datasets[0].CLASSES

        if self.use_wandb:
            name = f"{self.run_name}_size{args.size}_patch{args.patch}_dim{getattr(args, 'dim', 'na')}_bs{args.bs}_lr{args.lr}"
            wandb.init(project="ade20k-experiments", name=name)
            wandb.config.update(vars(args))
        elif not WANDB_AVAILABLE and not _arg(args, "nowandb", False):
            print("WARNING: WandB not installed, skipping WandB logging.")

    def _build_run_name(self):
        cfg_path = getattr(self.args, "cfg", "")
        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0] if cfg_path else ""
        if not cfg_name:
            cfg_name = f"{self.args.model}_seg_dim{getattr(self.args, 'dim', 'na')}"
        run_tag = getattr(self.args, "run_tag", None)
        return f"{cfg_name}_{run_tag}" if run_tag else cfg_name

    def _build_mmseg_config(self):
        args = self.args
        cfg = Config()
        cfg.model = self._get_upernet_config()
        cfg.data = self._get_data_config()

        cfg.optimizer = dict(
            type="AdamW",
            lr=args.lr,
            betas=_as_betas(getattr(args, "betas", None)),
            weight_decay=_arg(args, "weight_decay", 0.01),
            paramwise_cfg=self._get_paramwise_cfg(),
        )
        cfg.optimizer_config = dict(grad_clip=None)

        if bool(_arg(args, "amp", False)):
            cfg.fp16 = dict(loss_scale="dynamic")
            print("✅ MMSeg fp16 enabled: cfg.fp16 = dynamic loss scale")

        max_iters = getattr(args, "max_iters", None)
        if max_iters is None:
            max_iters = int(_arg(args, "n_epochs", 32) * 1000)
        warmup_iters = getattr(args, "warmup_iters", None)
        if warmup_iters is None:
            warmup_iters = int(_arg(args, "warmup_epochs", 0) * 1000)

        cfg.runner = dict(type="IterBasedRunner", max_iters=max_iters)
        cfg.lr_config = dict(
            policy="poly",
            warmup="linear" if warmup_iters > 0 else None,
            warmup_iters=warmup_iters,
            warmup_ratio=1e-6,
            power=1.0,
            min_lr=_arg(args, "min_lr", 0.0),
            by_epoch=False,
        )
        cfg.checkpoint_config = dict(
            by_epoch=False,
            interval=int(_arg(args, "checkpoint_interval", 5000)),
            filename_tmpl=f"{self.run_name}_{args.task or 'seg'}_iter_{{}}.pth",
            max_keep_ckpts=3,
        )

        log_interval = int(_arg(args, "log_interval", 100))
        eval_interval = int(_arg(args, "eval_interval", 2001))
        if eval_interval % log_interval == 0:
            eval_interval += 1
        cfg.evaluation = dict(interval=eval_interval, metric="mIoU", pre_eval=True, save_best="mIoU", classwise=False)
        cfg.log_config = dict(interval=log_interval, hooks=[dict(type="TextLoggerHook", by_epoch=False)])

        cfg.custom_hooks = []
        cfg.final_eval = bool(_arg(args, "final_eval", True))
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
        print(f"   lr: {args.lr}")
        print(f"   weight_decay: {_arg(args, 'weight_decay', 0.01)}")
        print(f"   layer_decay_rate: {_arg(args, 'layer_decay_rate', 1.0)}")
        print(f"   crop_size: {_arg(args, 'crop_size', 512)}")
        print(f"   backbone_image_size: {self._get_backbone_image_size()}")
        print(f"   seg_head_dim: {_arg(args, 'seg_head_dim', self._get_default_seg_head_dim())}")
        print(f"   seg_aux_dim: {_arg(args, 'seg_aux_dim', self._get_default_seg_head_dim())}")
        print(f"   seg_norm_type: {_arg(args, 'seg_norm_type', 'SyncBN')}")
        print(f"   work_dir: {cfg.work_dir}")
        return cfg

    def _get_paramwise_cfg(self):
        args = self.args
        layer_decay_rate = float(_arg(args, "layer_decay_rate", 1.0))
        custom_keys = {
            "backbone.pos_embedding": dict(decay_mult=0.0),
            "cope.pos_emb": dict(decay_mult=0.0),
            "backbone.cls_token": dict(decay_mult=0.0),
            "backbone.hk_gate": dict(decay_mult=0.0),
            ".fn.lam": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
        if args.model != "swin" and layer_decay_rate < 1.0:
            depth = int(_arg(args, "depth", 12))
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
                prefix = f"backbone.transformer.layers.{layer_idx}"
                custom_keys[f"{prefix}.0.fn"] = dict(lr_mult=lr_mult)
                custom_keys[f"{prefix}.1.fn"] = dict(lr_mult=lr_mult)
                custom_keys[f"{prefix}.0.norm"] = dict(lr_mult=lr_mult, decay_mult=0.0)
                custom_keys[f"{prefix}.1.norm"] = dict(lr_mult=lr_mult, decay_mult=0.0)
        return dict(custom_keys=custom_keys, norm_decay_mult=0.0)

    def _get_upernet_config(self):
        in_channels = self._get_backbone_out_channels()
        head_dim = int(_arg(self.args, "seg_head_dim", self._get_default_seg_head_dim()))
        aux_dim = int(_arg(self.args, "seg_aux_dim", self._get_default_seg_head_dim()))
        aux_idx = int(_arg(self.args, "seg_aux_in_index", 2))
        norm_cfg = self._get_seg_norm_cfg()

        neck_cfg = None
        seg_neck_style = str(_arg(self.args, "seg_neck_style", "xcit_fpn")).lower()
        if self.args.model != "swin" and seg_neck_style in ("multilevel", "external"):
            neck_dim = int(_arg(self.args, "seg_neck_dim", in_channels[0]))
            neck_cfg = dict(type="MultiLevelNeck", in_channels=in_channels, out_channels=neck_dim, scales=[4, 2, 1, 0.5])
            in_channels = [neck_dim] * 4

        return dict(
            type="EncoderDecoder",
            pretrained=None,
            backbone=self._get_backbone_config(),
            neck=neck_cfg,
            decode_head=dict(
                type="UPerHead",
                in_channels=in_channels,
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                channels=head_dim,
                dropout_ratio=0.1,
                num_classes=150,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            ),
            auxiliary_head=dict(
                type="FCNHead",
                in_channels=in_channels[aux_idx],
                in_index=aux_idx,
                channels=aux_dim,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=150,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
            ),
            train_cfg=dict(),
            test_cfg=dict(mode="whole"),
        )

    def _get_default_seg_head_dim(self):
        # Scratching ViT protocol: Ti/S/B use head dims 192/384/512 respectively.
        return int(_arg(self.args, "dim", 512))

    def _get_seg_norm_cfg(self):
        norm_type = str(_arg(self.args, "seg_norm_type", "SyncBN"))
        if norm_type.upper() == "GN":
            return dict(type="GN", num_groups=32, requires_grad=True)
        return dict(type=norm_type, requires_grad=True)

    def _get_backbone_image_size(self):
        value = _arg(self.args, "backbone_size", _arg(self.args, "crop_size", self.args.size))
        return tuple(int(v) for v in value) if isinstance(value, (tuple, list)) else int(value)

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
                drop_path_rate=float(_arg(args, "drop_path_rate", 0.0)),
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
            dim_head=_arg(args, "dim_head", 64),
            drop_path_rate=float(_arg(args, "drop_path_rate", 0.0)),
            out_indices=tuple(_arg(args, "out_indices", (3, 5, 7, 11))),
            fpn_adapter_style=self._get_vit_fpn_adapter_style(),
        )
        if args.model == "vit":
            return dict(type="ViTBackbone", **common)
        if args.model == "vitcope":
            return dict(type="ViTCoPEBackbone", use_cls_token=bool(_arg(args, "use_cls_token", False)), **common)
        if args.model == "vitscope":
            return dict(type="ViTSCoPEBackbone", use_cls_token=bool(_arg(args, "use_cls_token", True)), **common)
        raise ValueError(f"Unknown segmentation backbone model: {args.model}")

    def _get_vit_fpn_adapter_style(self):
        style = str(_arg(self.args, "seg_neck_style", "xcit_fpn")).lower()
        if style in ("internal_resize", "resize"):
            return "resize"
        if style in ("xcit_fpn", "simple_fpn", "official", "official_xcit"):
            return "simple_fpn"
        return "identity"

    def _get_backbone_out_channels(self):
        if self.args.model == "swin":
            return [self.args.embed_dim * (2 ** i) for i in range(4)]
        return [self.args.dim] * 4

    def _get_data_config(self):
        args = self.args
        img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        crop_size = _as_pair(_arg(args, "crop_size", None), 512)
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
        # Official ADE20K/XCiT-style validation pipeline: no extra Pad before Normalize.
        test_pipeline = [
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=test_img_scale,
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(type="Normalize", **img_norm_cfg),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ]
        return dict(
            samples_per_gpu=args.bs,
            workers_per_gpu=int(_arg(args, "workers_per_gpu", 4)),
            train=dict(type="ADE20KDataset", data_root=args.data_dir, img_dir="images/training", ann_dir="annotations/training", pipeline=train_pipeline),
            val=dict(type="ADE20KDataset", data_root=args.data_dir, img_dir="images/validation", ann_dir="annotations/validation", pipeline=test_pipeline),
            test=dict(type="ADE20KDataset", data_root=args.data_dir, img_dir="images/validation", ann_dir="annotations/validation", pipeline=test_pipeline),
        )

    def _extract_state_dict(self, checkpoint):
        if "model" in checkpoint:
            print("   Using checkpoint['model']")
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            print("   Using checkpoint['state_dict']")
            return checkpoint["state_dict"]
        print("   Using checkpoint directly")
        return checkpoint

    def _map_pretrained_key(self, key):
        raw_key = key[len("backbone."):] if key.startswith("backbone.") else key
        last_norm_idx = len(getattr(self.args, "out_indices", (3, 5, 7, 11))) - 1
        if raw_key in ("mlp_head.0.weight", "mlp_head.0.bias"):
            return f"backbone.norms.{last_norm_idx}.{raw_key.rsplit('.', 1)[1]}", "final_norm"
        if raw_key in ("norm.weight", "norm.bias"):
            return f"backbone.norms.{last_norm_idx}.{raw_key.split('.', 1)[1]}", "final_norm"
        if raw_key.startswith(("mlp_head.", "head.", "fc.")) or "classifier" in raw_key:
            return None, "skip_head"
        return (key if key.startswith("backbone.") else f"backbone.{key}"), "backbone"

    def _load_pretrained_backbone(self, pretrained_path):
        print(f"\n{'=' * 60}\n🔧 PRETRAINED LOADING DEBUG\n{'=' * 60}")
        print(f"📦 Pretrained path: {pretrained_path}")
        print(f"   File exists: {os.path.exists(pretrained_path)}")
        if not os.path.exists(pretrained_path):
            print("⚠️  Pretrained weights not found. Training from scratch.")
            return
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        pretrained_dict = self._extract_state_dict(checkpoint)
        backbone_dict, skipped, remapped = {}, [], []
        for k, v in pretrained_dict.items():
            for prefix in ("module.", "net.", "model."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
            new_key, kind = self._map_pretrained_key(k)
            if kind == "skip_head":
                skipped.append(k)
                continue
            if kind == "final_norm":
                remapped.append(f"{k} -> {new_key}")
            backbone_dict[new_key] = v

        model_dict = self.model.state_dict()
        matched, unmatched, interpolated = {}, [], []
        for k, v in backbone_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                matched[k] = v
            elif k in model_dict and k.endswith("pos_embedding") and v.ndim == 3 and model_dict[k].ndim == 3 and v.shape[-1] == model_dict[k].shape[-1]:
                matched[k] = self._resize_token_position_embedding(v, model_dict[k])
                interpolated.append(f"{k}: {tuple(v.shape)} -> {tuple(model_dict[k].shape)}")
            elif k in model_dict and k.endswith("cope.pos_emb") and v.ndim == 3 and model_dict[k].ndim == 3 and v.shape[:2] == model_dict[k].shape[:2]:
                matched[k] = F.interpolate(v.float(), size=model_dict[k].shape[-1], mode="linear", align_corners=False).to(model_dict[k].dtype)
                interpolated.append(f"{k}: {tuple(v.shape)} -> {tuple(model_dict[k].shape)}")
            else:
                unmatched.append(f"{k} mismatch or not in model")

        print("\n📊 Matching results")
        print(f"   After mapping: {len(backbone_dict)} keys")
        print(f"   Matched: {len(matched)} keys")
        print(f"   Interpolated position tables: {len(interpolated)}")
        print(f"   Unmatched: {len(unmatched)} keys")
        print(f"   Skipped classifier keys: {len(skipped)}")
        print(f"   Remapped final norm keys: {len(remapped)}")
        if unmatched:
            for item in unmatched[:10]:
                print(f"     {item}")
        if interpolated:
            for item in interpolated[:6]:
                print(f"     {item}")
        model_dict.update(matched)
        self.model.load_state_dict(model_dict, strict=False)
        rate = 100.0 * len(matched) / max(1, len(backbone_dict))
        print(f"\n✅ FINAL: Loaded {len(matched)}/{len(backbone_dict)} backbone keys ({rate:.1f}%)")
        if rate < float(getattr(self.args, "min_pretrained_match_rate", 80.0)):
            raise RuntimeError(f"Pretrained backbone match rate {rate:.1f}% is too low")
        print(f"{'=' * 60}\n")

    def _resize_token_position_embedding(self, source, target):
        source_tokens, target_tokens = source.shape[1], target.shape[1]
        has_cls = int((source_tokens - 1) ** 0.5) ** 2 == source_tokens - 1 and int((target_tokens - 1) ** 0.5) ** 2 == target_tokens - 1
        if has_cls:
            source_cls, source_patch, target_patch_tokens = source[:, :1], source[:, 1:], target_tokens - 1
        else:
            source_cls, source_patch, target_patch_tokens = None, source, target_tokens
        old_size, new_size = int(source_patch.shape[1] ** 0.5), int(target_patch_tokens ** 0.5)
        if old_size * old_size != source_patch.shape[1] or new_size * new_size != target_patch_tokens:
            return source
        source_patch = source_patch.reshape(1, old_size, old_size, source.shape[-1]).permute(0, 3, 1, 2)
        source_patch = F.interpolate(source_patch.float(), size=(new_size, new_size), mode="bicubic", align_corners=False)
        source_patch = source_patch.permute(0, 2, 3, 1).reshape(1, target_patch_tokens, source.shape[-1])
        resized = torch.cat([source_cls.float(), source_patch], dim=1) if source_cls is not None else source_patch
        return resized.to(dtype=target.dtype)

    def train(self):
        print(f"🚀 Start training {self.args.model} + UPerNet on ADE20K\n")
        train_segmentor(self.model, self.datasets, self.cfg, distributed=False, validate=True, timestamp=time.strftime("%Y%m%d_%H%M%S", time.localtime()), meta=dict())
        if self.cfg.get("final_eval", True):
            self._run_final_eval()
        print("\n✅ Segmentation training finished!\n")
        if self.use_wandb:
            wandb.finish()

    def _run_final_eval(self):
        print("\n🚀 Running final ADE20K evaluation after max_iters")
        val_dataset = build_dataset(self.cfg.data.val, dict(test_mode=True))
        val_loader = build_dataloader(val_dataset, samples_per_gpu=1, workers_per_gpu=self.cfg.data.workers_per_gpu, dist=False, shuffle=False)
        eval_model = MMDataParallel(self.model.cuda(self.cfg.gpu_ids[0]), device_ids=self.cfg.gpu_ids)
        eval_model.CLASSES = val_dataset.CLASSES
        if hasattr(val_dataset, "PALETTE"):
            eval_model.PALETTE = val_dataset.PALETTE
        results = single_gpu_test(eval_model, val_loader, show=False)
        eval_results = val_dataset.evaluate(results, metric=self.cfg.evaluation.get("metric", "mIoU"), logger=None)
        print("\n" + "=" * 60)
        print("FINAL SEGMENTATION EVALUATION RESULTS")
        print("=" * 60)
        for key, value in eval_results.items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        print("=" * 60 + "\n")
        if self.use_wandb:
            wandb.log({f"final_{k}": v for k, v in eval_results.items() if isinstance(v, (int, float))})
