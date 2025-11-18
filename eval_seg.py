#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate segmentation model
"""
import os
import torch
import argparse
import yaml
from mmcv import Config
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
    parser.add_argument('--config', type=str, default='configs/seg_vit.yaml',
                        help='config file')
    parser.add_argument('--checkpoint', type=str, 
                        default='work_dirs/vit_upernet/latest.pth',
                        help='checkpoint file')
    return parser.parse_args()


def build_mmseg_config(args_dict):
    """Build MMSegmentation configuration"""
    cfg = Config()
    
    # Model configuration
    cfg.model = dict(
        type='EncoderDecoder',
        backbone=dict(
            type='ViT',
            img_size=512,
            patch_size=16,
            in_channels=3,
            embed_dims=192,
            num_layers=12,
            num_heads=3,
            mlp_ratio=4,
            out_indices=(2, 5, 8, 11),
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            with_cls_token=True,
            interpolate_mode='bicubic',
        ),
        decode_head=dict(
            type='UPerHead',
            in_channels=[192, 192, 192, 192],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        ),
        auxiliary_head=dict(
            type='FCNHead',
            in_channels=192,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
        ),
        train_cfg=dict(),
        test_cfg=dict(mode='whole')
    )
    
    # Dataset configuration
    data_dir = args_dict.get('data_dir', './datasets/ADE20K/ADEChallengeData2016')
    
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    crop_size = (512, 512)
    
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
    
    cfg.data = dict(
        samples_per_gpu=1,
        workers_per_gpu=4,
        test=dict(
            type='ADE20KDataset',
            data_root=data_dir,
            img_dir='images/validation',
            ann_dir='annotations/validation',
            pipeline=test_pipeline
        )
    )
    
    return cfg


def main():
    args = parse_args()
    
    print(f"📂 Loading config from: {args.config}")
    with open(args.config) as f:
        args_dict = yaml.safe_load(f)
    
    print(f"🔧 Building model...")
    cfg = build_mmseg_config(args_dict)
    
    # Build model
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )
    
    # Load checkpoint
    print(f"📥 Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda()
    model.eval()
    
    # Build validation dataset
    print(f"📊 Building validation dataset...")
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        dist=False,
        shuffle=False
    )
    
    # Run evaluation
    print(f"🚀 Running evaluation...")
    results = single_gpu_test(model, data_loader, show=False)
    
    # Compute metrics
    print(f"📈 Computing metrics...")
    eval_results = dataset.evaluate(results, metric='mIoU', logger='silent')
    
    # Print results
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    for key, val in eval_results.items():
        if isinstance(val, float):
            print(f"{key:20s}: {val*100:.2f}%")
        else:
            print(f"{key:20s}: {val}")
    print("="*60)
    
    return eval_results


if __name__ == '__main__':
    main()









