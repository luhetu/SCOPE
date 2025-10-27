# 导入 backbones
import sys
sys.path.insert(0, '/home/hetu/MY project/SCOPE')

# MMDetection 内部依赖（必须保留）
from .resnet import ResNet, ResNetV1d

# 我们使用的 backbones
from .swin_transformer import SwinTransformer
from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

__all__ = [
    # MMDetection 内部依赖
    'ResNet', 'ResNetV1d',
    # 我们的模型
    'SwinTransformer',
    'ViTBackbone', 'ViTCoPEBackbone', 'ViTSCoPEBackbone'
]
