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
        
        print(f"\n🔧 Segmentation Task Init")
        print(f"   pretrained: {getattr(args, 'pretrained', 'NOT SET')}")
        
        # WandB（需要在构建配置前设置）
        self.use_wandb = WANDB_AVAILABLE and not args.nowandb
        
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
        
        # 加载预训练权重（如果提供）
        if hasattr(args, 'pretrained') and args.pretrained:
            print(f"\n🔧 Loading pretrained weights: {args.pretrained}")
            self._load_pretrained_backbone(args.pretrained)
        else:
            print(f"⚠️  Training from scratch (no pretrained weights)")
        
        # 构建数据集
        self.datasets = [build_dataset(self.cfg.data.train)]
        
        # WandB 初始化
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
        # 保存格式: {model}_{task}_iter_{iter}.pth
        model_name = args.model
        task_name = args.task or 'seg'
        cfg.checkpoint_config = dict(
            by_epoch=False,
            interval=5000,
            filename_tmpl=f'{model_name}_{task_name}_iter_{{}}.pth'
        )
        
        # ==================== 评估配置 ==================== #
        cfg.evaluation = dict(
            interval=2000,  # 每2000 iter评估一次（避免与日志冲突）
            metric='mIoU',
            pre_eval=True,
            save_best='mIoU',
            classwise=False  # 禁用每类详细输出
        )
        
        # ==================== 日志配置 ==================== #
        cfg.log_config = dict(
            interval=100,  # 每100 iter记录一次
            hooks=[
                dict(type='TextLoggerHook', by_epoch=False),
            ]
        )
        
        # 修复评估间隔，避免与日志冲突
        # 确保评估不在日志记录时进行
        cfg.evaluation.interval = 2001  # 避免与 100 的倍数冲突
        
        # ==================== 进度条和 WandB 配置 ==================== #
        cfg.custom_hooks = [
            dict(type='SegProgressBarHook'),  # 添加进度条（分割专用）
            dict(type='SimpleWandBHook', use_wandb=self.use_wandb, log_interval=100),  # WandB 记录训练和评估指标
        ]
        
        # ==================== 其他配置 ==================== #
        cfg.dist_params = dict(backend='nccl')
        cfg.log_level = 'INFO'
        cfg.work_dir = f'./work_dirs/{args.model}_upernet'
        cfg.load_from = None  # 不使用 MMSeg 自动加载
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
            dict(type='Resize', img_scale=(args.size * 2, args.size), ratio_range=(0.5, 2.0)),  # 基于 args.size
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
                img_scale=(args.size, args.size),  # 使用固定尺寸，避免 ViT patch 尺寸不匹配
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),  # 不保持比例，确保尺寸正确
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
    
    def _load_pretrained_backbone(self, pretrained_path):
        """加载分类任务预训练的backbone权重"""
        import os
        print(f"\n{'='*60}")
        print(f"🔧 PRETRAINED LOADING DEBUG")
        print(f"{'='*60}")
        print(f"📦 Pretrained path: {pretrained_path}")
        print(f"   File exists: {os.path.exists(pretrained_path)}")
        
        if not os.path.exists(pretrained_path):
            print(f"⚠️  Pretrained weights not found!")
            print(f"   Training from scratch...")
            print(f"{'='*60}\n")
            return
        
        # 加载分类模型的checkpoint
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        print(f"✅ Checkpoint loaded")
        print(f"   Checkpoint keys: {list(checkpoint.keys())[:5]}")
        
        # 获取模型权重（支持不同的保存格式）
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
            print(f"   Using checkpoint['model']")
        elif 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
            print(f"   Using checkpoint['state_dict']")
        else:
            pretrained_dict = checkpoint
            print(f"   Using checkpoint directly")
        
        print(f"   Total pretrained keys: {len(pretrained_dict)}")
        print(f"   Sample keys: {list(pretrained_dict.keys())[:3]}")
        
        # 过滤并重命名权重，只加载backbone部分
        backbone_dict = {}
        skipped_keys = []
        for k, v in pretrained_dict.items():
            # 跳过分类头和 CLS token
            if 'mlp_head' in k or 'head' in k or 'fc' in k or 'cls_token' in k:
                skipped_keys.append(k)
                continue
            
            # 键名映射：分类模型 -> 分割模型
            new_k = k
            
            # 1. transformer.layers.X.Y -> transformer_blocks.X.Y
            new_k = new_k.replace('transformer.layers', 'transformer_blocks')
            
            # 2. blocks.X -> transformer_blocks.X
            if new_k.startswith('blocks.'):
                new_k = new_k.replace('blocks.', 'transformer_blocks.', 1)
            
            # 3. to_patch.1 -> to_patch_embedding.1
            if new_k.startswith('to_patch.'):
                new_k = new_k.replace('to_patch.', 'to_patch_embedding.', 1)
            
            # 4. fn.scope.pos_emb -> fn.cope.pos_emb (SCoPE模型)
            if 'fn.scope.pos_emb' in new_k:
                new_k = new_k.replace('fn.scope.pos_emb', 'fn.cope.pos_emb')
            
            # 5. cope_emb 跳过（全局CoPE vs 每层CoPE）
            if 'cope_emb' in new_k:
                skipped_keys.append(k)
                continue
            
            # 重命名为分割模型的backbone前缀
            new_key = f'backbone.{new_k}'
            backbone_dict[new_key] = v
        
        # 加载到分割模型
        model_dict = self.model.state_dict()
        
        # 检查哪些键匹配
        matched_keys = []
        unmatched_keys = []
        for k, v in backbone_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    matched_keys.append(k)
                else:
                    unmatched_keys.append(f"{k} (shape mismatch: {v.shape} vs {model_dict[k].shape})")
            else:
                unmatched_keys.append(f"{k} (not in model)")
        
        print(f"\n📊 Matching results:")
        print(f"   After mapping: {len(backbone_dict)} keys")
        print(f"   Matched: {len(matched_keys)} keys")
        print(f"   Unmatched: {len(unmatched_keys)} keys")
        
        if len(matched_keys) > 0:
            print(f"\n✅ Sample matched keys (first 3):")
            for k in list(matched_keys)[:3]:
                print(f"     {k}")
        
        if len(unmatched_keys) > 0 and len(unmatched_keys) <= 10:
            print(f"\n⚠️  Unmatched keys:")
            for uk in unmatched_keys[:10]:
                print(f"     {uk}")
        
        # 只更新存在且形状匹配的键
        pretrained_dict_filtered = {k: v for k, v in backbone_dict.items() if k in matched_keys}
        model_dict.update(pretrained_dict_filtered)
        self.model.load_state_dict(model_dict, strict=False)
        
        match_rate = 100*len(pretrained_dict_filtered)/max(1,len(backbone_dict))
        print(f"\n✅ FINAL: Loaded {len(pretrained_dict_filtered)}/{len(backbone_dict)} layers ({match_rate:.1f}%)")
        
        if match_rate < 50:
            print(f"\n⚠️  WARNING: Low match rate! Checking key format mismatch...")
            print(f"   Sample backbone_dict key: {list(backbone_dict.keys())[0] if backbone_dict else 'NONE'}")
            print(f"   Sample model_dict key: {[k for k in model_dict.keys() if 'backbone' in k][0] if any('backbone' in k for k in model_dict.keys()) else 'NONE'}")
        
        print(f"{'='*60}\n")

