#!/usr/bin/env python3
"""Lightweight validation for segmentation and detection task wiring.

This check intentionally avoids importing the compiled MMCV/MMDetection/
MMSegmentation runtime. It stubs the small pieces needed by the local task
builders, validates every seg/det YAML config, and smoke-tests the ViT-family
backbones with real PyTorch tensors.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import os
import sys
import types
from types import SimpleNamespace

import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _wrap_config_value(value):
    if isinstance(value, dict):
        return ConfigDict(value)
    if isinstance(value, list):
        return [_wrap_config_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_wrap_config_value(item) for item in value)
    return value


class ConfigDict(dict):
    """Tiny subset of mmcv.ConfigDict used by the task builders."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(dict(*args, **kwargs))

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __setitem__(self, key, value):
        super().__setitem__(key, _wrap_config_value(value))

    def update(self, other=None, **kwargs):
        if other:
            for key, value in dict(other).items():
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value


class Config(ConfigDict):
    """Tiny subset of mmcv.Config used in tasks/*.py."""


class Registry:
    """Minimal registry with the register_module API used by OpenMMLab."""

    def __init__(self):
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False, **kwargs):
        def _register(cls):
            key = name or cls.__name__
            if not force and key in self.module_dict:
                raise KeyError(key)
            self.module_dict[key] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


def install_openmmlab_stubs():
    """Install small in-memory stubs before importing task modules."""

    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv.__version__ = "1.3.17"
    mmcv._ext = object()
    sys.modules["mmcv"] = mmcv

    for root in ("mmdet", "mmseg"):
        pkg = types.ModuleType(root)
        pkg.__path__ = []
        sys.modules[root] = pkg

        apis = types.ModuleType(f"{root}.apis")
        apis.set_random_seed = lambda *args, **kwargs: None
        if root == "mmdet":
            apis.train_detector = lambda *args, **kwargs: None
        else:
            apis.train_segmentor = lambda *args, **kwargs: None
        sys.modules[f"{root}.apis"] = apis

        datasets = types.ModuleType(f"{root}.datasets")
        datasets.build_dataset = lambda cfg: cfg
        sys.modules[f"{root}.datasets"] = datasets

        models = types.ModuleType(f"{root}.models")
        models.__path__ = []
        if root == "mmdet":
            models.build_detector = lambda *args, **kwargs: None
        else:
            models.build_segmentor = lambda *args, **kwargs: None
        sys.modules[f"{root}.models"] = models

        builder = types.ModuleType(f"{root}.models.builder")
        builder.BACKBONES = Registry()
        sys.modules[f"{root}.models.builder"] = builder


def load_args(path):
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    defaults = dict(
        cfg=path,
        run_tag=None,
        seed=1,
        workers_per_gpu=0,
        checkpoint_interval=100,
        eval_interval=101,
        log_interval=10,
        warmup_epochs=0,
        warmup_iters=None,
        min_lr=0.0,
        opt="adamw",
        amp=False,
        nowandb=True,
        weight_decay=0.01,
        drop_path_rate=0.0,
        use_cls_token=False,
        dim_head=64,
    )
    defaults.update(data)
    return SimpleNamespace(**defaults)


def validate_segmentation_configs(config_dir):
    segmentation = importlib.import_module("tasks.segmentation")
    paths = sorted(glob.glob(os.path.join(config_dir, "seg_*.yaml")))
    if not paths:
        raise AssertionError(f"No segmentation configs found in {config_dir}")

    for path in paths:
        args = load_args(path)
        task = segmentation.SegmentationTask.__new__(segmentation.SegmentationTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmseg_config()

        assert cfg.model.type == "EncoderDecoder"
        assert len(cfg.model.decode_head.in_channels) == 4
        assert cfg.data.train.type == "ADE20KDataset"
        assert cfg.model.backbone.type in {
            "SwinTransformer",
            "ViTBackbone",
            "ViTCoPEBackbone",
            "ViTSCoPEBackbone",
        }
        print(f"OK seg {os.path.relpath(path, REPO_ROOT)} -> {cfg.model.backbone.type}")


def validate_detection_configs(config_dir):
    detection = importlib.import_module("tasks.detection")
    paths = sorted(glob.glob(os.path.join(config_dir, "detection_*.yaml")))
    if not paths:
        raise AssertionError(f"No detection configs found in {config_dir}")

    for path in paths:
        args = load_args(path)
        task = detection.DetectionTask.__new__(detection.DetectionTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmdet_config()

        assert cfg.model.type == "MaskRCNN"
        assert len(cfg.model.neck.in_channels) == 4
        assert cfg.data.train.type == "CocoDataset"
        assert cfg.model.backbone.type in {
            "SwinTransformer",
            "ViTBackbone",
            "ViTCoPEBackbone",
            "ViTSCoPEBackbone",
        }
        print(f"OK det {os.path.relpath(path, REPO_ROOT)} -> {cfg.model.backbone.type}")


def validate_backbone_forward():
    try:
        import torch
        from models.vit_backbone import (
            ViTBackbone,
            ViTCoPEBackbone,
            ViTSCoPEBackbone,
        )
    except ImportError as exc:
        raise SystemExit(
            "Backbone smoke test requires torch, timm, and einops. "
            "Install the project runtime dependencies first."
        ) from exc

    torch.set_num_threads(1)

    cases = [
        ("vit", ViTBackbone, {}),
        ("vitcope_no_cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("vitcope_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("vitscope", ViTSCoPEBackbone, {}),
    ]
    expected_shapes = [
        (2, 24, 8, 8),
        (2, 24, 4, 4),
        (2, 24, 2, 2),
        (2, 24, 1, 1),
    ]

    for adapter_style in ("resize", "simple_fpn"):
        for name, cls, extra in cases:
            model = cls(
                image_size=32,
                patch_size=16,
                dim=24,
                depth=4,
                heads=3,
                mlp_dim=48,
                dim_head=8,
                out_indices=(0, 1, 2, 3),
                fpn_adapter_style=adapter_style,
                **extra,
            )
            model.eval()
            with torch.no_grad():
                outputs = model(torch.randn(2, 3, 32, 32))

            shapes = [tuple(output.shape) for output in outputs]
            assert shapes == expected_shapes, (adapter_style, name, shapes)
            assert all(torch.isfinite(output).all().item() for output in outputs)
            print(f"OK backbone {adapter_style}/{name} -> {shapes}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        default=os.path.join(REPO_ROOT, "configs"),
        help="Directory containing seg_*.yaml and detection_*.yaml configs.",
    )
    parser.add_argument(
        "--skip-backbone-smoke",
        action="store_true",
        help="Only validate task config builders.",
    )
    args = parser.parse_args()

    install_openmmlab_stubs()
    validate_segmentation_configs(args.config_dir)
    validate_detection_configs(args.config_dir)
    if not args.skip_backbone_smoke:
        validate_backbone_forward()

    print("Segmentation/detection validation passed.")


if __name__ == "__main__":
    main()
