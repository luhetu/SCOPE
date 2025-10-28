#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新增的 CoPE 变体模型
"""

import torch
import sys

def test_model(model_name, model_class, model_kwargs):
    """测试单个模型"""
    print(f"\n{'='*60}")
    print(f"🧪 测试 {model_name}")
    print(f"{'='*60}")
    
    try:
        # 创建模型
        model = model_class(**model_kwargs)
        print(f"✅ 模型创建成功")
        
        # 统计参数量
        params = sum(p.numel() for p in model.parameters())
        print(f"📊 参数量: {params/1e6:.2f}M")
        
        # 测试前向传播（不带梯度）
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            y_test = model(x)
        print(f"✅ 前向传播成功: {x.shape} -> {y_test.shape}")
        
        # 测试梯度反向传播（需要重新计算带梯度）
        x_grad = torch.randn(2, 3, 224, 224)
        y = model(x_grad)
        loss = y.sum()
        loss.backward()
        print(f"✅ 反向传播成功")
        
        print(f"🎉 {model_name} 测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ {model_name} 测试失败:")
        print(f"   错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("🚀 测试所有 CoPE 变体模型")
    print("="*60)
    
    results = {}
    
    # ==================== 1. ViTCoPE (Embedding层) ==================== #
    try:
        from models.vitcope_embed import ViTcope
        results['vitcope'] = test_model(
            'ViTCoPE (Embedding层)',
            ViTcope,
            {
                'image_size': 224,
                'patch_size': 32,
                'num_classes': 1000,
                'dim': 384,
                'depth': 12,
                'heads': 6,
                'mlp_dim': 1536,
                'dim_head': 64,
                'pool': 'mean',
            }
        )
    except ImportError as e:
        print(f"\n❌ vitcope_embed.py 导入失败: {e}")
        results['vitcope'] = False
    
    # ==================== 2. ViTSCoPE (Embedding层) ==================== #
    try:
        from models.vitscope_embed import ViTScope
        results['vitscope'] = test_model(
            'ViTSCoPE (Embedding层)',
            ViTScope,
            {
                'image_size': 224,
                'patch_size': 32,
                'num_classes': 1000,
                'dim': 384,
                'depth': 12,
                'heads': 6,
                'mlp_dim': 1536,
                'dim_head': 64,
                'pool': 'mean',
            }
        )
    except ImportError as e:
        print(f"\n❌ vitscope_embed.py 导入失败: {e}")
        results['vitscope'] = False
    
    # ==================== 总结 ==================== #
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name:20s} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 所有模型测试通过！")
        print("\n下一步:")
        print("  python train.py --cfg configs/vit.yaml      # ViT baseline")
        print("  python train.py --cfg configs/vitcope.yaml  # CoPE (Embedding)")
        print("  python train.py --cfg configs/vitscope.yaml # SCoPE (Embedding)")
    else:
        print("⚠️  部分模型测试失败，请检查上面的错误信息")
        sys.exit(1)
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

