# -*- coding: utf-8 -*-
"""
目标检测任务
基于 MMDetection 框架，支持 Swin/ViT/CoPE/SCoPE 作为 Backbone
"""
import os
import time
import torch
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import get_root_logger

# WandB 可选依赖
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class DetectionTask:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 构建 MMDetection 配置
        self.cfg = self._build_mmdet_config()
        
        # 设置随机种子
        if hasattr(args, 'seed'):
            set_random_seed(args.seed, deterministic=True)
        
        # 构建模型
        self.model = build_detector(
            self.cfg.model,
            train_cfg=self.cfg.get('train_cfg'),
            test_cfg=self.cfg.get('test_cfg')
        )
        
        # 加载预训练权重（如果提供）
        if hasattr(args, 'pretrained') and args.pretrained:
            self._load_pretrained_backbone(args.pretrained)
        
        # 构建数据集
        self.datasets = [build_dataset(self.cfg.data.train)]
        if len(self.cfg.workflow) == 2:
            import copy
            val_dataset = copy.deepcopy(self.cfg.data.val)
            val_dataset.pipeline = self.cfg.data.train.pipeline
            self.datasets.append(build_dataset(val_dataset))
        
        # WandB
        self.use_wandb = WANDB_AVAILABLE and not args.nowandb
        if self.use_wandb:
            # 统一项目名为数据集名-experiments
            project_name = "coco-experiments"
            
            # 运行名称包含模型、任务和关键参数
            watermark = f"{args.model}_maskrcnn_size{args.size}_bs{args.bs}_lr{args.lr}"
            
            wandb.init(project=project_name, name=watermark)
            wandb.config.update(vars(args))
        elif not WANDB_AVAILABLE and not args.nowandb:
            print("WARNING: WandB not installed, skipping logging")
        
        # 设置模型类别
        self.model.CLASSES = self.datasets[0].CLASSES
        
    def _build_mmdet_config(self):
        """从 args 构建 MMDetection 配置"""
        args = self.args
        
        # 基础配置
        cfg = Config()
        
        # ==================== 模型配置 ==================== #
        cfg.model = self._get_mask_rcnn_config()
        
        # ==================== 数据配置 ==================== #
        cfg.data = self._get_data_config()
        
        # ==================== 优化器配置 ==================== #
        cfg.optimizer = dict(
            type='AdamW',
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
            paramwise_cfg=dict(
                custom_keys={
                    'absolute_pos_embed': dict(decay_mult=0.),
                    'relative_position_bias_table': dict(decay_mult=0.),
                    'norm': dict(decay_mult=0.)
                }
            )
        )
        
        # ==================== 学习率调度 ==================== #
        cfg.lr_config = dict(
            policy='step',
            warmup='linear',
            warmup_iters=args.warmup_epochs * 1000,  # 假设每个epoch 1000 iters
            warmup_ratio=0.001,
            step=[args.n_epochs * 2 // 3, args.n_epochs * 8 // 9]
        )
        
        # ==================== Runner 配置（单卡训练）==================== #
        cfg.runner = dict(type='EpochBasedRunner', max_epochs=args.n_epochs)
        cfg.optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
        
        # ==================== Checkpoint 配置 ==================== #
        cfg.checkpoint_config = dict(interval=1)
        
        # ==================== 日志配置（仅在 epoch 结束时打印）==================== #
        cfg.log_config = dict(
            interval=9999999,  # 设置很大的值，避免频繁打印
            hooks=[
                dict(type='TextLoggerHook'),
            ]
        )
        
        # ==================== 进度条配置 ==================== #
        # 检查是否使用 WandB
        use_wandb = WANDB_AVAILABLE and not args.nowandb
        
        cfg.custom_hooks = [
            dict(type='NumClassCheckHook'),
            dict(type='ProgressBarHook'),  # 添加进度条
            dict(type='WandBLoggerHook', interval=1, use_wandb=use_wandb)  # WandB 指标记录
        ]
        
        # ==================== 评估配置 ==================== #
        cfg.evaluation = dict(
            interval=1,  # 每个 epoch 评估一次
            metric=['bbox', 'segm'],  # 评估 bbox 和 segm
            save_best='bbox_mAP',  # 根据 bbox mAP 保存最佳模型
            classwise=False  # 不需要每个类别的详细指标
        )
        
        # ==================== 其他配置 ==================== #
        cfg.dist_params = dict(backend='nccl')
        cfg.log_level = 'INFO'
        cfg.work_dir = f'./work_dirs/{args.model}_maskrcnn'
        cfg.load_from = None
        cfg.resume_from = None
        cfg.workflow = [('train', 1)]
        cfg.gpu_ids = range(1)
        cfg.seed = getattr(args, 'seed', None)
        
        return cfg
    
    def _get_mask_rcnn_config(self):
        """构建 Mask R-CNN 配置"""
        args = self.args
        
        # Backbone 配置
        backbone_cfg = self._get_backbone_config()
        
        # Neck (FPN)
        neck_cfg = dict(
            type='FPN',
            in_channels=self._get_backbone_out_channels(),
            out_channels=256,
            num_outs=5
        )
        
        model = dict(
            type='MaskRCNN',
            backbone=backbone_cfg,
            neck=neck_cfg,
            rpn_head=dict(
                type='RPNHead',
                in_channels=256,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[8],
                    ratios=[0.5, 1.0, 2.0],
                    strides=[4, 8, 16, 32, 64]
                ),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[.0, .0, .0, .0],
                    target_stds=[1.0, 1.0, 1.0, 1.0]
                ),
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)
            ),
            roi_head=dict(
                type='StandardRoIHead',
                bbox_roi_extractor=dict(
                    type='SingleRoIExtractor',
                    roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                    out_channels=256,
                    featmap_strides=[4, 8, 16, 32]
                ),
                bbox_head=dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=80,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.1, 0.1, 0.2, 0.2]
                    ),
                    reg_class_agnostic=False,
                    loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                    loss_bbox=dict(type='L1Loss', loss_weight=1.0)
                ),
                mask_roi_extractor=dict(
                    type='SingleRoIExtractor',
                    roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
                    out_channels=256,
                    featmap_strides=[4, 8, 16, 32]
                ),
                mask_head=dict(
                    type='FCNMaskHead',
                    num_convs=4,
                    in_channels=256,
                    conv_out_channels=256,
                    num_classes=80,
                    loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
                )
            ),
            train_cfg=dict(
                rpn=dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.3,
                        min_pos_iou=0.3,
                        match_low_quality=True,
                        ignore_iof_thr=-1
                    ),
                    sampler=dict(
                        type='RandomSampler',
                        num=256,
                        pos_fraction=0.5,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=False
                    ),
                    allowed_border=-1,
                    pos_weight=-1,
                    debug=False
                ),
                rpn_proposal=dict(
                    nms_pre=2000,
                    max_per_img=1000,
                    nms=dict(type='nms', iou_threshold=0.7),
                    min_bbox_size=0
                ),
                rcnn=dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        match_low_quality=True,
                        ignore_iof_thr=-1
                    ),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True
                    ),
                    mask_size=28,
                    pos_weight=-1,
                    debug=False
                )
            ),
            test_cfg=dict(
                rpn=dict(
                    nms_pre=1000,
                    max_per_img=1000,
                    nms=dict(type='nms', iou_threshold=0.7),
                    min_bbox_size=0
                ),
                rcnn=dict(
                    score_thr=0.05,
                    nms=dict(type='nms', iou_threshold=0.5),
                    max_per_img=100,
                    mask_thr_binary=0.5
                )
            )
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
                mlp_ratio=4.,
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
                out_indices=(2, 5, 8, 11)  # 选择4个中间层作为特征输出
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
        
        # 图像归一化
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True
        )
        
        # 训练Pipeline
        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
        ]
        
        # 测试Pipeline
        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ]
            )
        ]
        
        data = dict(
            samples_per_gpu=args.bs,
            workers_per_gpu=4,
            train=dict(
                type='CocoDataset',
                ann_file=f'{args.data_dir}/annotations/instances_train2017.json',
                img_prefix=f'{args.data_dir}/train2017/',
                pipeline=train_pipeline
            ),
            val=dict(
                type='CocoDataset',
                ann_file=f'{args.data_dir}/annotations/instances_val2017.json',
                img_prefix=f'{args.data_dir}/val2017/',
                pipeline=test_pipeline
            ),
            test=dict(
                type='CocoDataset',
                ann_file=f'{args.data_dir}/annotations/instances_val2017.json',
                img_prefix=f'{args.data_dir}/val2017/',
                pipeline=test_pipeline
            )
        )
        
        return data
    
    def _load_pretrained_backbone(self, pretrained_path):
        """加载分类任务预训练的backbone权重"""
        import os
        if not os.path.exists(pretrained_path):
            print(f"⚠️  Pretrained weights not found: {pretrained_path}")
            print("   Skipping pretrained loading, training from scratch...")
            return
        
        print(f"📦 Loading pretrained backbone from: {pretrained_path}")
        
        # 加载分类模型的checkpoint
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 获取模型权重（支持不同的保存格式）
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        else:
            pretrained_dict = checkpoint
        
        # 过滤并重命名权重，只加载backbone部分
        backbone_dict = {}
        for k, v in pretrained_dict.items():
            # 跳过分类头
            if 'mlp_head' in k or 'head' in k or 'fc' in k:
                continue
            # 重命名为检测模型的backbone前缀
            new_key = f'backbone.{k}'
            backbone_dict[new_key] = v
        
        # 加载到检测模型
        model_dict = self.model.state_dict()
        # 只更新存在的键
        pretrained_dict_filtered = {k: v for k, v in backbone_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict_filtered)
        self.model.load_state_dict(model_dict, strict=False)
        
        print(f"✅ Loaded {len(pretrained_dict_filtered)} pretrained layers")
        print(f"   Total model layers: {len(model_dict)}")
    
    def train(self):
        """开始训练"""
        print(f"🚀 Start training {self.args.model} + Mask R-CNN on COCO\n")
        
        # 调用 MMDetection 的训练 API
        train_detector(
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

