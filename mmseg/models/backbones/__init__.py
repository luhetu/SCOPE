# 添加项目根目录到路径，以便导入自定义模型
import sys
sys.path.insert(0, '/home/hetu/MY project/SCOPE')

from .cgnet import CGNet
from .fast_scnn import FastSCNN
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnext import ResNeXt
from .unet import UNet
from .swin_transformer import SwinTransformer

# 导入自定义 ViT 系列 Backbone
from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

__all__ = [
    'ResNet', 'ResNetV1c', 'ResNetV1d', 'ResNeXt', 'HRNet', 'FastSCNN',
    'ResNeSt', 'MobileNetV2', 'UNet', 'CGNet', 'MobileNetV3', 'SwinTransformer',
    'ViTBackbone', 'ViTCoPEBackbone', 'ViTSCoPEBackbone'
]
