# --------------------------------------------------------
# Swin Transformer V2 - Utility Functions
# Contains helper functions for weight loading, model building, etc.
# --------------------------------------------------------

import torch
import torch.nn as nn
from collections import OrderedDict
import os


def load_pretrained_weights(model, pretrained_path, strict=False, prefix=''):
    """
    Load pretrained weights into a model.
    
    Args:
        model: PyTorch model.
        pretrained_path (str): Path to the pretrained weights file.
        strict (bool): Whether to strictly enforce that the keys in state_dict
                       match the keys returned by model.state_dict(). Default False.
        prefix (str): Optional prefix to add to state_dict keys (for handling
                      different saving conventions).
    
    Returns:
        model: The model with loaded weights.
    """
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Weight file not found: {pretrained_path}")
    
    print(f"Loading pretrained weights from {pretrained_path} ...")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DDP/DP-wrapped models
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # remove 'module.' prefix
        else:
            name = k
        
        if prefix and not name.startswith(prefix):
            name = prefix + name
        
        new_state_dict[name] = v
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=strict)
    
    if missing_keys:
        print(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]} ...")
    if unexpected_keys:
        print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]} ...")
    
    print("Pretrained weights loaded successfully!")
    return model


def build_model_from_config(config, num_classes=1000):
    """
    Build a SwinTransformerV2 model from a configuration dictionary.
    
    Args:
        config (dict): Model configuration dictionary.
        num_classes (int): Number of classification categories.
    
    Returns:
        model: The built model instance.
    """
    from .swin_v2_model import SwinTransformerV2
    
    # Update num_classes
    model_config = config.copy()
    model_config['num_classes'] = num_classes
    
    # Build model
    model = SwinTransformerV2(**model_config)
    
    return model


def download_pretrained_weights(url, save_path):
    """
    Download pretrained weights from a URL.
    
    Args:
        url (str): Download URL.
        save_path (str): Path to save the downloaded file.
    """
    import urllib.request
    
    if os.path.exists(save_path):
        print(f"Weight file already exists: {save_path}")
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Downloading pretrained weights from {url} ...")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Download completed and saved to: {save_path}")
    except Exception as e:
        print(f"Download failed: {e}")
        raise


def create_model(model_name='tiny', num_classes=1000, pretrained=False, 
                 pretrained_path=None, img_size=None):
    """
    Convenience function to create a Swin V2 model.
    
    Args:
        model_name (str): Model size ('tiny', 'small', 'base', 'large').
        num_classes (int): Number of classification categories.
        pretrained (bool): Whether to load pretrained weights.
        pretrained_path (str): Path to pretrained weights (if provided).
        img_size (int): Input image size; if provided, overrides config default.
    
    Returns:
        model: The created model.
    
    Example:
        >>> # Create a model without pretrained weights
        >>> model = create_model('tiny', num_classes=100)
        
        >>> # Create a model with pretrained weights
        >>> model = create_model('base', pretrained=True, pretrained_path='path/to/weights.pth')
        
        >>> # Custom image size
        >>> model = create_model('small', num_classes=10, img_size=384)
    """
    from .model_configs import get_config
    
    # Get base configuration
    config = get_config(model_name)
    
    # Override image size if specified
    if img_size is not None:
        config['img_size'] = img_size
    
    # Build model
    model = build_model_from_config(config, num_classes=num_classes)
    
    # Load pretrained weights if requested
    if pretrained and pretrained_path:
        model = load_pretrained_weights(model, pretrained_path, strict=False)
    
    return model


def get_model_info(model):
    """
    Get basic model information (parameter count, FLOPs, etc.).
    
    Args:
        model: PyTorch model.
    
    Returns:
        dict: Dictionary containing model information.
    """
    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_M': total_params / 1e6,
        'trainable_params_M': trainable_params / 1e6,
    }
    
    # Try to get FLOPs if the model defines a flops() method
    if hasattr(model, 'flops'):
        try:
            flops = model.flops()
            info['flops'] = flops
            info['flops_G'] = flops / 1e9
        except Exception:
            pass
    
    return info


def print_model_info(model):
    """
    Print model information.
    
    Args:
        model: PyTorch model.
    """
    info = get_model_info(model)
    
    print("=" * 60)
    print("Model Info:")
    print("-" * 60)
    print(f"Total params: {info['total_params']:,} ({info['total_params_M']:.2f} M)")
    print(f"Trainable params: {info['trainable_params']:,} ({info['trainable_params_M']:.2f} M)")
    
    if 'flops_G' in info:
        print(f"FLOPs: {info['flops']:,} ({info['flops_G']:.2f} G)")
    
    print("=" * 60)


def convert_to_backbone(model, output_indices=[0, 1, 2, 3]):
    """
    Convert a classification Swin V2 model into a backbone (for detection/segmentation).
    
    Args:
        model: Swin V2 classification model.
        output_indices (list): Indices of stages whose features will be returned.
    
    Returns:
        backbone: Modified backbone model.
    
    Note:
        This removes the classification head and modifies the forward method
        to output multi-scale feature maps.
    """
    # Remove classification head
    model.head = nn.Identity()
    
    # Save original forward_features (not used directly after replacement,
    # but kept in case you want to restore or inspect)
    original_forward = model.forward_features  # noqa: F841 (kept intentionally)
    
    def forward_backbone(self, x):
        """Forward method for extracting multi-scale feature maps."""
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in output_indices:
                # Reshape to (B, C, H, W)
                B, L, C = x.shape
                H = W = int(L ** 0.5)
                feat = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                features.append(feat)
        
        return features
    
    # Replace forward with backbone-style forward
    import types
    model.forward = types.MethodType(forward_backbone, model)
    
    return model
