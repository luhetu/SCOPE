import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16

from ..builder import NECKS


@NECKS.register_module()
class SimpleFeaturePyramid(nn.Module):
    """ViTDet-style simple feature pyramid for plain ViT backbones.

    The input features are same-resolution ViT block outputs. Each output level
    is projected to `out_channels` while applying scale factors [4, 2, 1, 0.5].
    A final max-pooled level provides P6 for RPN anchors.
    """

    def __init__(
        self,
        in_channels,
        out_channels=256,
        scale_factors=(4, 2, 1, 0.5),
        num_outs=5,
        norm_cfg=dict(type="LN", requires_grad=True),
        act_cfg=dict(type="GELU"),
    ):
        super().__init__()
        assert isinstance(in_channels, list)
        assert len(in_channels) == len(scale_factors)
        assert num_outs in (len(scale_factors), len(scale_factors) + 1)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factors = list(scale_factors)
        self.num_outs = num_outs
        self.fp16_enabled = False

        self.stages = nn.ModuleList([
            self._make_stage(in_channel, out_channels, scale, norm_cfg, act_cfg)
            for in_channel, scale in zip(in_channels, scale_factors)
        ])

        self.extra_pool = nn.MaxPool2d(kernel_size=1, stride=2)

    def _make_stage(self, in_channels, out_channels, scale, norm_cfg, act_cfg):
        layers = []

        if scale == 4:
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.GELU(),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            ])
        elif scale == 2:
            layers.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            )
        elif scale == 1:
            if in_channels != out_channels:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            else:
                layers.append(nn.Identity())
        elif scale == 0.5:
            layers.extend([
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            ])
        else:
            raise ValueError(f"Unsupported SimpleFeaturePyramid scale: {scale}")

        layers.append(
            ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
        )

        return nn.Sequential(*layers)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                xavier_init(module, distribution="uniform")

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        outs = [
            stage(inputs[i])
            for i, stage in enumerate(self.stages)
        ]

        if self.num_outs == len(outs) + 1:
            outs.append(self.extra_pool(outs[-1]))

        return tuple(outs)
