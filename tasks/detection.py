# -*- coding: utf-8 -*-
"""
Object Detection Task
MMDetection + ViT / ViT-CoPE / ViT-SCoPE backbone.
"""

import os
import time
import torch

from mmcv import Config
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.models.builder import BACKBONES as MMDET_BACKBONES

from models.vit_backbone import (
    ViTBackbone,
    ViTCoPEBackbone,
    ViTSCoPEBackbone,
)


def _register_backbone_once(name, module):
    if name in MMDET_BACKBONES.module_dict:
        print(f"✅ [MMDET Registry] {name} already registered")
        return

    MMDET_BACKBONES.register_module(name=name, module=module)
    print(f"✅ [MMDET Registry] registered {name}")


_register_backbone_once("ViTBackbone", ViTBackbone)
_register_backbone_once("ViTCoPEBackbone", ViTCoPEBackbone)
_register_backbone_once("ViTSCoPEBackbone", ViTSCoPEBackbone)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class DetectionTask:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.run_name = self._build_run_name()

        print("\n🔧 Detection Task Init")
        print(f"   model: {args.model}")
        print(f"   pretrained: {getattr(args, 'pretrained', 'NOT SET')}")

        self.cfg = self._build_mmdet_config()

        if hasattr(args, "seed") and args.seed is not None:
            set_random_seed(args.seed, deterministic=True)

        self.model = build_detector(
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

        if len(self.cfg.workflow) == 2:
            import copy
            val_dataset = copy.deepcopy(self.cfg.data.val)
            val_dataset.pipeline = self.cfg.data.train.pipeline
            self.datasets.append(build_dataset(val_dataset))

        self.use_wandb = WANDB_AVAILABLE and not getattr(args, "nowandb", False)

        if self.use_wandb:
            project_name = "coco-experiments"
            watermark = (
                f"{self.run_name}_size{args.size}_patch{args.patch}_"
                f"dim{getattr(args, 'dim', getattr(args, 'embed_dim', 'na'))}_"
                f"bs{args.bs}_lr{args.lr}"
            )
            wandb.init(project=project_name, name=watermark)
            wandb.config.update(vars(args))
        elif not WANDB_AVAILABLE and not getattr(args, "nowandb", False):
            print("WARNING: WandB not installed, skipping WandB logging.")

        self.model.CLASSES = self.datasets[0].CLASSES

    # ------------------------------------------------------- #
    def _build_run_name(self):
        args = self.args
        cfg_path = getattr(args, "cfg", "")
        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0] if cfg_path else ""
        if not cfg_name:
            dim = getattr(args, "dim", getattr(args, "embed_dim", "na"))
            cfg_name = f"{args.model}_det_dim{dim}"

        run_tag = getattr(args, "run_tag", None)
        return f"{cfg_name}_{run_tag}" if run_tag else cfg_name

    # ------------------------------------------------------- #
    def _build_mmdet_config(self):
        args = self.args
        cfg = Config()

        cfg.model = self._get_mask_rcnn_config()
        cfg.data = self._get_data_config()

        cfg.optimizer = dict(
            type="AdamW",
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=getattr(args, "weight_decay", 0.05),
            paramwise_cfg=dict(
                custom_keys={
                    "pos_embedding": dict(decay_mult=0.0),
                    "cls_token": dict(decay_mult=0.0),
                    "absolute_pos_embed": dict(decay_mult=0.0),
                    "relative_position_bias_table": dict(decay_mult=0.0),
                    "norm": dict(decay_mult=0.0),
                }
            ),
        )

        cfg.optimizer_config = dict(
            grad_clip=dict(max_norm=35, norm_type=2)
        )

        if bool(getattr(args, "amp", False)):
            cfg.fp16 = dict(loss_scale="dynamic")
            print("✅ MMDet fp16 enabled: cfg.fp16 = dynamic loss scale")

        warmup_iters = getattr(args, "warmup_iters", None)
        if warmup_iters is None:
            warmup_iters = int(getattr(args, "warmup_epochs", 0) * 1000)

        cfg.lr_config = dict(
            policy="step",
            warmup="linear" if warmup_iters > 0 else None,
            warmup_iters=warmup_iters,
            warmup_ratio=0.001,
            step=[
                args.n_epochs * 2 // 3,
                args.n_epochs * 8 // 9,
            ],
        )

        cfg.runner = dict(
            type="EpochBasedRunner",
            max_epochs=args.n_epochs,
        )

        model_name = self.run_name
        task_name = args.task or "det"

        cfg.checkpoint_config = dict(
            interval=1,
            filename_tmpl=f"{model_name}_{task_name}_epoch_{{}}.pth",
            max_keep_ckpts=3,
        )

        cfg.evaluation = dict(
            interval=1,
            metric=["bbox", "segm"],
            save_best="bbox_mAP",
            classwise=False,
        )

        cfg.log_config = dict(
            interval=int(getattr(args, "log_interval", 50)),
            hooks=[
                dict(type="TextLoggerHook"),
            ],
        )

        # 只保留 mmdet 内置 hook，避免未注册自定义 hook 报错
        cfg.custom_hooks = [
            dict(type="NumClassCheckHook"),
        ]

        cfg.dist_params = dict(backend="nccl")
        cfg.log_level = "INFO"
        cfg.work_dir = f"./work_dirs/{self.run_name}_maskrcnn"
        cfg.load_from = None
        cfg.resume_from = None
        cfg.workflow = [("train", 1)]
        cfg.gpu_ids = range(1)
        cfg.seed = getattr(args, "seed", None)

        print("\n✅ MMDet config summary")
        print(f"   epochs: {args.n_epochs}")
        print(f"   lr: {args.lr}")
        print(f"   weight_decay: {getattr(args, 'weight_decay', 0.05)}")
        print(f"   warmup_iters: {warmup_iters}")
        print(f"   img_scale: {getattr(args, 'img_scale', [1333, 800])}")
        print(f"   work_dir: {cfg.work_dir}")

        return cfg

    # ------------------------------------------------------- #
    def _get_mask_rcnn_config(self):
        backbone_cfg = self._get_backbone_config()

        neck_cfg = dict(
            type="FPN",
            in_channels=self._get_backbone_out_channels(),
            out_channels=256,
            num_outs=5,
        )

        model = dict(
            type="MaskRCNN",
            backbone=backbone_cfg,
            neck=neck_cfg,
            rpn_head=dict(
                type="RPNHead",
                in_channels=256,
                feat_channels=256,
                anchor_generator=dict(
                    type="AnchorGenerator",
                    scales=[8],
                    ratios=[0.5, 1.0, 2.0],
                    strides=[4, 8, 16, 32, 64],
                ),
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[1.0, 1.0, 1.0, 1.0],
                ),
                loss_cls=dict(
                    type="CrossEntropyLoss",
                    use_sigmoid=True,
                    loss_weight=1.0,
                ),
                loss_bbox=dict(
                    type="L1Loss",
                    loss_weight=1.0,
                ),
            ),
            roi_head=dict(
                type="StandardRoIHead",
                bbox_roi_extractor=dict(
                    type="SingleRoIExtractor",
                    roi_layer=dict(
                        type="RoIAlign",
                        output_size=7,
                        sampling_ratio=0,
                    ),
                    out_channels=256,
                    featmap_strides=[4, 8, 16, 32],
                ),
                bbox_head=dict(
                    type="Shared2FCBBoxHead",
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=80,
                    bbox_coder=dict(
                        type="DeltaXYWHBBoxCoder",
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[0.1, 0.1, 0.2, 0.2],
                    ),
                    reg_class_agnostic=False,
                    loss_cls=dict(
                        type="CrossEntropyLoss",
                        use_sigmoid=False,
                        loss_weight=1.0,
                    ),
                    loss_bbox=dict(
                        type="L1Loss",
                        loss_weight=1.0,
                    ),
                ),
                mask_roi_extractor=dict(
                    type="SingleRoIExtractor",
                    roi_layer=dict(
                        type="RoIAlign",
                        output_size=14,
                        sampling_ratio=0,
                    ),
                    out_channels=256,
                    featmap_strides=[4, 8, 16, 32],
                ),
                mask_head=dict(
                    type="FCNMaskHead",
                    num_convs=4,
                    in_channels=256,
                    conv_out_channels=256,
                    num_classes=80,
                    loss_mask=dict(
                        type="CrossEntropyLoss",
                        use_mask=True,
                        loss_weight=1.0,
                    ),
                ),
            ),
            train_cfg=dict(
                rpn=dict(
                    assigner=dict(
                        type="MaxIoUAssigner",
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.3,
                        min_pos_iou=0.3,
                        match_low_quality=True,
                        ignore_iof_thr=-1,
                    ),
                    sampler=dict(
                        type="RandomSampler",
                        num=256,
                        pos_fraction=0.5,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=False,
                    ),
                    allowed_border=-1,
                    pos_weight=-1,
                    debug=False,
                ),
                rpn_proposal=dict(
                    nms_pre=2000,
                    max_per_img=1000,
                    nms=dict(type="nms", iou_threshold=0.7),
                    min_bbox_size=0,
                ),
                rcnn=dict(
                    assigner=dict(
                        type="MaxIoUAssigner",
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        match_low_quality=True,
                        ignore_iof_thr=-1,
                    ),
                    sampler=dict(
                        type="RandomSampler",
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True,
                    ),
                    mask_size=28,
                    pos_weight=-1,
                    debug=False,
                ),
            ),
            test_cfg=dict(
                rpn=dict(
                    nms_pre=1000,
                    max_per_img=1000,
                    nms=dict(type="nms", iou_threshold=0.7),
                    min_bbox_size=0,
                ),
                rcnn=dict(
                    score_thr=0.05,
                    nms=dict(type="nms", iou_threshold=0.5),
                    max_per_img=100,
                    mask_thr_binary=0.5,
                ),
            ),
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
            fpn_adapter_style="simple_fpn",
        )

        if args.model == "swin":
            return dict(
                type="SwinTransformer",
                embed_dim=args.embed_dim,
                depths=args.depths,
                num_heads=args.num_heads,
                window_size=args.window_size,
                mlp_ratio=4.0,
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

        raise ValueError(f"Unknown detection backbone model: {args.model}")

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

        if hasattr(args, "img_scale"):
            if isinstance(args.img_scale, (tuple, list)):
                img_scale = tuple(args.img_scale)
            else:
                img_scale = (args.img_scale, args.img_scale)
        else:
            img_scale = (1333, 800)

        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
        )

        train_pipeline = [
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
            dict(type="Resize", img_scale=img_scale, keep_ratio=True),
            dict(type="RandomFlip", flip_ratio=0.5),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="DefaultFormatBundle"),
            dict(
                type="Collect",
                keys=["img", "gt_bboxes", "gt_labels", "gt_masks"],
            ),
        ]

        test_pipeline = [
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=img_scale,
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="Normalize", **img_norm_cfg),
                    dict(type="Pad", size_divisor=32),
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
                type="CocoDataset",
                ann_file=f"{args.data_dir}/annotations/instances_train2017.json",
                img_prefix=f"{args.data_dir}/train2017/",
                pipeline=train_pipeline,
            ),
            val=dict(
                type="CocoDataset",
                ann_file=f"{args.data_dir}/annotations/instances_val2017.json",
                img_prefix=f"{args.data_dir}/val2017/",
                pipeline=test_pipeline,
            ),
            test=dict(
                type="CocoDataset",
                ann_file=f"{args.data_dir}/annotations/instances_val2017.json",
                img_prefix=f"{args.data_dir}/val2017/",
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

            # skip classifier heads
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
        print(f"🚀 Start training {self.args.model} + Mask R-CNN on COCO\n")

        train_detector(
            self.model,
            self.datasets,
            self.cfg,
            distributed=False,
            validate=True,
            timestamp=time.strftime("%Y%m%d_%H%M%S", time.localtime()),
            meta=dict(),
        )

        print("\n✅ Detection training finished!\n")

        if self.use_wandb:
            wandb.finish()