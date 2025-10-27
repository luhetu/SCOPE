# --------------------------------------------------------
# Swin Transformer V2 - 工具函数
# 包含权重加载、模型构建等辅助函数
# --------------------------------------------------------

import torch
import torch.nn as nn
from collections import OrderedDict
import os


def load_pretrained_weights(model, pretrained_path, strict=False, prefix=''):
    """
    加载预训练权重到模型
    
    Args:
        model: PyTorch模型
        pretrained_path (str): 预训练权重文件路径
        strict (bool): 是否严格匹配所有参数。默认False（允许部分加载）
        prefix (str): 权重键的前缀（用于处理不同的保存格式）
    
    Returns:
        model: 加载权重后的模型
    """
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"权重文件不存在: {pretrained_path}")
    
    print(f"从 {pretrained_path} 加载预训练权重...")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # 处理不同的checkpoint格式
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 处理DDP/DP包装的模型
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # 移除'module.'前缀
        else:
            name = k
        
        if prefix and not name.startswith(prefix):
            name = prefix + name
        
        new_state_dict[name] = v
    
    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=strict)
    
    if missing_keys:
        print(f"缺失的键 ({len(missing_keys)}): {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"未预期的键 ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
    
    print("预训练权重加载完成！")
    return model


def build_model_from_config(config, num_classes=1000):
    """
    根据配置字典构建模型
    
    Args:
        config (dict): 模型配置字典
        num_classes (int): 分类类别数
    
    Returns:
        model: 构建的模型
    """
    from .swin_v2_model import SwinTransformerV2
    
    # 更新类别数
    model_config = config.copy()
    model_config['num_classes'] = num_classes
    
    # 构建模型
    model = SwinTransformerV2(**model_config)
    
    return model


def download_pretrained_weights(url, save_path):
    """
    从URL下载预训练权重
    
    Args:
        url (str): 下载URL
        save_path (str): 保存路径
    """
    import urllib.request
    
    if os.path.exists(save_path):
        print(f"权重文件已存在: {save_path}")
        return
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"从 {url} 下载预训练权重...")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"下载完成，保存至: {save_path}")
    except Exception as e:
        print(f"下载失败: {e}")
        raise


def create_model(model_name='tiny', num_classes=1000, pretrained=False, 
                 pretrained_path=None, img_size=None):
    """
    创建Swin V2模型的便捷函数
    
    Args:
        model_name (str): 模型名称 ('tiny', 'small', 'base', 'large')
        num_classes (int): 分类类别数
        pretrained (bool): 是否加载预训练权重
        pretrained_path (str): 预训练权重路径（如果提供）
        img_size (int): 输入图像尺寸（如果提供，会覆盖默认配置）
    
    Returns:
        model: 创建的模型
    
    示例:
        >>> # 创建不带预训练权重的模型
        >>> model = create_model('tiny', num_classes=100)
        
        >>> # 创建带预训练权重的模型
        >>> model = create_model('base', pretrained=True, pretrained_path='path/to/weights.pth')
        
        >>> # 自定义图像尺寸
        >>> model = create_model('small', num_classes=10, img_size=384)
    """
    from .model_configs import get_config
    
    # 获取配置
    config = get_config(model_name)
    
    # 覆盖图像尺寸
    if img_size is not None:
        config['img_size'] = img_size
    
    # 构建模型
    model = build_model_from_config(config, num_classes=num_classes)
    
    # 加载预训练权重
    if pretrained and pretrained_path:
        model = load_pretrained_weights(model, pretrained_path, strict=False)
    
    return model


def get_model_info(model):
    """
    获取模型信息（参数量、FLOPs等）
    
    Args:
        model: PyTorch模型
    
    Returns:
        dict: 包含模型信息的字典
    """
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_M': total_params / 1e6,
        'trainable_params_M': trainable_params / 1e6,
    }
    
    # 尝试获取FLOPs（如果模型有flops方法）
    if hasattr(model, 'flops'):
        try:
            flops = model.flops()
            info['flops'] = flops
            info['flops_G'] = flops / 1e9
        except:
            pass
    
    return info


def print_model_info(model):
    """
    打印模型信息
    
    Args:
        model: PyTorch模型
    """
    info = get_model_info(model)
    
    print("=" * 60)
    print("模型信息:")
    print("-" * 60)
    print(f"总参数量: {info['total_params']:,} ({info['total_params_M']:.2f}M)")
    print(f"可训练参数: {info['trainable_params']:,} ({info['trainable_params_M']:.2f}M)")
    
    if 'flops_G' in info:
        print(f"FLOPs: {info['flops']:,} ({info['flops_G']:.2f}G)")
    
    print("=" * 60)


def convert_to_backbone(model, output_indices=[0, 1, 2, 3]):
    """
    将分类模型转换为backbone（用于检测/分割任务）
    
    Args:
        model: Swin V2分类模型
        output_indices (list): 需要输出的stage索引
    
    Returns:
        backbone: 修改后的backbone模型
    
    注意: 这会移除分类头，并修改forward方法以输出多尺度特征
    """
    # 移除分类头
    model.head = nn.Identity()
    
    # 保存原始的forward方法
    original_forward = model.forward_features
    
    def forward_backbone(self, x):
        """提取多尺度特征的forward方法"""
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in output_indices:
                # 重塑为 (B, C, H, W) 格式
                B, L, C = x.shape
                H = W = int(L ** 0.5)
                feat = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                features.append(feat)
        
        return features
    
    # 替换forward方法
    import types
    model.forward = types.MethodType(forward_backbone, model)
    
    return model

