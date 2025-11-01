#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提取 checkpoint 中的模型权重，移除 optimizer、scheduler 等，节省空间
用法: python scripts/extract_model_weights.py <checkpoint_path> [output_path]
"""
import torch
import sys
import os

def extract_model_weights(input_path, output_path=None):
    """
    从 checkpoint 中提取模型权重
    
    Args:
        input_path: 输入 checkpoint 路径
        output_path: 输出路径（如果不指定，会在同目录下创建 _model_only.pth）
    """
    if not os.path.exists(input_path):
        print(f"❌ 文件不存在: {input_path}")
        return
    
    print(f"📂 加载 checkpoint: {input_path}")
    ckpt = torch.load(input_path, map_location='cpu')
    
    # 获取文件大小
    input_size = os.path.getsize(input_path) / (1024*1024)  # MB
    
    # 检查内容
    original_keys = list(ckpt.keys())
    print(f"   原始内容: {original_keys}")
    
    # 创建新的 checkpoint（只包含模型）
    new_ckpt = {
        'model': ckpt['model']
    }
    
    # 保留训练信息（如果有用）
    if 'epoch' in ckpt:
        new_ckpt['epoch'] = ckpt['epoch']
    if 'acc' in ckpt:
        new_ckpt['acc'] = ckpt['acc']
    if 'best_acc' in ckpt:
        new_ckpt['best_acc'] = ckpt['best_acc']
    
    # 确定输出路径
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = base_name + '_model_only.pth'
    
    # 保存
    print(f"💾 保存模型权重到: {output_path}")
    torch.save(new_ckpt, output_path)
    
    # 计算节省的空间
    output_size = os.path.getsize(output_path) / (1024*1024)  # MB
    saved_size = input_size - output_size
    
    print(f"\n📊 大小对比:")
    print(f"   原始文件: {input_size:.1f} MB")
    print(f"   模型权重: {output_size:.1f} MB")
    print(f"   节省空间: {saved_size:.1f} MB ({saved_size/input_size*100:.1f}%)")
    
    # 计算模型参数量
    model_params = sum(p.numel() for p in ckpt['model'].values())
    print(f"\n🔧 模型信息:")
    print(f"   参数量: {model_params:,} ({model_params/1e6:.2f}M)")
    
    if 'optimizer' in ckpt:
        opt = ckpt['optimizer']
        if 'state' in opt:
            opt_params = sum(p.numel() for state in opt['state'].values() 
                            for p in state.values() if isinstance(p, torch.Tensor))
            print(f"   Optimizer 参数: {opt_params:,} ({opt_params/1e6:.2f}M) - 已移除 ✅")
    
    print(f"\n✅ 完成！")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python scripts/extract_model_weights.py <checkpoint_path> [output_path]")
        print("\n示例:")
        print("  python scripts/extract_model_weights.py checkpoint/vitscope_best.pth")
        print("  python scripts/extract_model_weights.py checkpoint/vitscope_best.pth checkpoint/vitscope_best_model.pth")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    extract_model_weights(input_path, output_path)

