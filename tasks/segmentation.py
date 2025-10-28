# -*- coding: utf-8 -*-
"""
语义分割任务
基于 MMSegmentation 框架，支持 Swin/ViT/CoPE/SCoPE 作为 Backbone
"""
import os
import time
import torch
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import get_root_logger

# WandB 可选依赖
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SegmentationTask:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 构建 MMSegmentation 配置
        self.cfg = self._build_mmseg_config()
        
        # 设置随机种子
        if hasattr(args, 'seed'):
            set_random_seed(args.seed, deterministic=True)
        
        # 构建模型
        self.model = build_segmentor(
            self.cfg.model,
            train_cfg=self.cfg.get('train_cfg'),
            test_cfg=self.cfg.get('test_cfg')
        )
        
        # 构建数据集
        self.datasets = [build_dataset(self.cfg.data.train)]
        
        # WandB
        self.use_wandb = WANDB_AVAILABLE and not args.nowandb
        if self.use_wandb:
            # 统一项目名为数据集名-experiments
            project_name = "ade20k-experiments"
            
            # 运行名称包含模型、任务和关键参数
            watermark = f"{args.model}_upernet_size{args.size}_bs{args.bs}_lr{args.lr}"
            
            wandb.init(project=project_name, name=watermark)
            wandb.config.update(vars(args))
        elif not WANDB_AVAILABLE and not args.nowandb:
            print("WARNING: WandB not installed, skipping logging")
        
        # 设置模型类别
        self.model.CLASSES = self.datasets[0].CLASSES
        
    def _build_mmseg_config(self):
        """从 args 构建 MMSegmentation 配置"""
        args = self.args
        
        # 基础配置
        cfg = Config()
        
        # ==================== 模型配置 ==================== #
        cfg.model = self._get_upernet_config()
        
        # ==================== 数据集配置 ==================== #
        cfg.data = self._get_data_config()
        
        # ==================== 优化器配置 ==================== #
        cfg.optimizer = dict(
            type='AdamW',
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )
        cfg.optimizer_config = dict(grad_clip=None)
        
        # ==================== 学习率策略 ==================== #
        cfg.lr_config = dict(
            policy='poly',
            warmup='linear',
            warmup_iters=args.warmup_epochs * 1000,  # 假设每个epoch 1000 iters
            warmup_ratio=1e-6,
            power=1.0,
            min_lr=args.min_lr,
            by_epoch=False
        )
        
        # ==================== Runner 配置（单卡训练）==================== #
        cfg.runner = dict(type='IterBasedRunner', max_iters=args.n_epochs * 1000)
        
        # ==================== Checkpoint 配置 ==================== #
        cfg.checkpoint_config = dict(by_epoch=False, interval=5000)
        
        # ==================== 日志配置（仅在 epoch 结束时打印）==================== #
        cfg.log_config = dict(
            interval=9999999,  # 设置很大的值，避免频繁打印
            hooks=[
                dict(type='TextLoggerHook', by_epoch=False),
            ]
        )
        
        # ==================== 进度条配置 ==================== #
        cfg.custom_hooks = [
            dict(type='SegProgressBarHook')  # 添加进度条（分割专用）
        ]
        
        # ==================== 其他配置 ==================== #
        cfg.dist_params = dict(backend='nccl')
        cfg.log_level = 'INFO'
        cfg.work_dir = f'./work_dirs/{args.model}_upernet'
        cfg.load_from = None
        cfg.gpu_ids = [0]  # 单GPU训练
        cfg.resume_from = None
        cfg.workflow = [('train', 1)]
        cfg.cudnn_benchmark = True
        cfg.seed = getattr(args, 'seed', None)
        
        return cfg
    
    def _get_upernet_config(self):
        """构建 UPerNet 模型配置（经典的语义分割架构）"""
        args = self.args
        
        # Backbone 配置
        backbone_cfg = self._get_backbone_config()
        
        # Decode Head (UPerNet)
        in_channels = self._get_backbone_out_channels()
        decode_head_cfg = dict(
            type='UPerHead',
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=150,  # ADE20K has 150 classes
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),  # 使用GroupNorm（支持bs=1）
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        )
        
        # Auxiliary Head
        auxiliary_head_cfg = dict(
            type='FCNHead',
            in_channels=in_channels[2],  # 使用第3层特征
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),  # 使用GroupNorm（支持bs=1）
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
        )
        
        # Training and Testing settings
        train_cfg = dict()
        test_cfg = dict(mode='whole')
        
        model = dict(
            type='EncoderDecoder',
            pretrained=None,
            backbone=backbone_cfg,
            decode_head=decode_head_cfg,
            auxiliary_head=auxiliary_head_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg
        )
        
        return model
    
    def _get_backbone_config(self):
        """根据模型类型构建 Backbone 配置"""
        args = self.args
        
        if args.model == 'swin':
            return dict(
                type='SwinTransformer',
                embed_dim=args.embed_dim,
                depths=args.depths,
                num_heads=args.num_heads,
                window_size=args.window_size,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=args.drop_path_rate,
                ape=False,
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                use_checkpoint=False
            )
        elif args.model == 'vit':
            return dict(
                type='ViTBackbone',
                image_size=args.size,
                patch_size=args.patch,
                dim=args.dim,
                depth=args.depth,
                heads=args.heads,
                mlp_dim=args.mlp_dim,
                dim_head=getattr(args, 'dim_head', 64),
                out_indices=(2, 5, 8, 11)
            )
        elif args.model == 'vitcope':
            return dict(
                type='ViTCoPEBackbone',
                image_size=args.size,
                patch_size=args.patch,
                dim=args.dim,
                depth=args.depth,
                heads=args.heads,
                mlp_dim=args.mlp_dim,
                dim_head=getattr(args, 'dim_head', 64),
                out_indices=(2, 5, 8, 11)
            )
        elif args.model == 'vitscope':
            return dict(
                type='ViTSCoPEBackbone',
                image_size=args.size,
                patch_size=args.patch,
                dim=args.dim,
                depth=args.depth,
                heads=args.heads,
                mlp_dim=args.mlp_dim,
                dim_head=getattr(args, 'dim_head', 64),
                out_indices=(2, 5, 8, 11)
            )
        else:
            raise ValueError(f"Unknown model: {args.model}")
    
    def _get_backbone_out_channels(self):
        """获取 Backbone 输出通道数"""
        args = self.args
        
        if args.model == 'swin':
            # Swin Transformer: [96, 192, 384, 768] for Tiny
            base_dim = args.embed_dim
            return [base_dim * (2 ** i) for i in range(4)]
        else:
            # ViT 系列: 所有层都是相同维度
            return [args.dim] * 4
    
    def _get_data_config(self):
        """构建数据配置"""
        args = self.args
        
        # 图像归一化参数
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True
        )
        
        # 数据增强pipeline
        crop_size = (args.size, args.size)
        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg']),
        ]
        
        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ])
        ]
        
        data = dict(
            samples_per_gpu=args.bs,
            workers_per_gpu=4,
            train=dict(
                type='ADE20KDataset',
                data_root=args.data_dir,
                img_dir='images/training',
                ann_dir='annotations/training',
                pipeline=train_pipeline
            ),
            val=dict(
                type='ADE20KDataset',
                data_root=args.data_dir,
                img_dir='images/validation',
                ann_dir='annotations/validation',
                pipeline=test_pipeline
            ),
            test=dict(
                type='ADE20KDataset',
                data_root=args.data_dir,
                img_dir='images/validation',
                ann_dir='annotations/validation',
                pipeline=test_pipeline
            )
        )
        
        return data
    
    def train(self):
        """开始训练"""
        print(f"🚀 Start training {self.args.model} + UPerNet on ADE20K\n")
        
        # 调用 MMSegmentation 的训练 API
        train_segmentor(
            self.model,
            self.datasets,
            self.cfg,
            distributed=False,
            validate=True,
            timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            meta=dict()
        )
        
        print(f"\n✅ Training finished!\n")
        
        if self.use_wandb:
            wandb.finish()

