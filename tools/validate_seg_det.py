#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight validation for segmentation and detection task code.

This script avoids real ADE20K/COCO data and mmcv-full C++ extensions. It
stubs the tiny OpenMMLab API surface needed to import the task modules, then
validates config construction for every dense-prediction YAML and runs CPU
forward smoke tests for the custom ViT dense backbones.
"""

from __future__ import annotations

import glob
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ConfigDict(dict):
    """Small attr-access dict compatible with the task config builders."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = _to_config_dict(value)


def _to_config_dict(value):
    if isinstance(value, dict) and not isinstance(value, ConfigDict):
        return ConfigDict({k: _to_config_dict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_config_dict(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_config_dict(v) for v in value)
    return value


class RegistryStub:
    def __init__(self, name):
        self.name = name
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        def _register(cls):
            key = name or cls.__name__
            if not force and key in self.module_dict:
                raise KeyError(f"{key} is already registered in {self.name}")
            self.module_dict[key] = cls
            return cls

        return _register(module) if module is not None else _register


def _install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = ConfigDict

    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object
    mmcv.parallel = mmcv_parallel

    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.parallel"] = mmcv_parallel

    mmseg_backbones = RegistryStub("mmseg_backbone")
    mmseg = types.ModuleType("mmseg")
    mmseg_apis = types.ModuleType("mmseg.apis")
    mmseg_apis.set_random_seed = lambda *args, **kwargs: None
    mmseg_apis.single_gpu_test = lambda *args, **kwargs: []
    mmseg_apis.train_segmentor = lambda *args, **kwargs: None
    mmseg_datasets = types.ModuleType("mmseg.datasets")
    mmseg_datasets.build_dataloader = lambda *args, **kwargs: None
    mmseg_datasets.build_dataset = lambda *args, **kwargs: None
    mmseg_models = types.ModuleType("mmseg.models")
    mmseg_models.build_segmentor = lambda *args, **kwargs: None
    mmseg_builder = types.ModuleType("mmseg.models.builder")
    mmseg_builder.BACKBONES = mmseg_backbones
    mmseg_models.builder = mmseg_builder
    mmseg.models = mmseg_models
    sys.modules["mmseg"] = mmseg
    sys.modules["mmseg.apis"] = mmseg_apis
    sys.modules["mmseg.datasets"] = mmseg_datasets
    sys.modules["mmseg.models"] = mmseg_models
    sys.modules["mmseg.models.builder"] = mmseg_builder

    mmdet_backbones = RegistryStub("mmdet_backbone")
    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = lambda *args, **kwargs: None
    mmdet_apis.train_detector = lambda *args, **kwargs: None
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = lambda *args, **kwargs: None
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = lambda *args, **kwargs: None
    mmdet_builder = types.ModuleType("mmdet.models.builder")
    mmdet_builder.BACKBONES = mmdet_backbones
    mmdet_models.builder = mmdet_builder
    mmdet.models = mmdet_models
    sys.modules["mmdet"] = mmdet
    sys.modules["mmdet.apis"] = mmdet_apis
    sys.modules["mmdet.datasets"] = mmdet_datasets
    sys.modules["mmdet.models"] = mmdet_models
    sys.modules["mmdet.models.builder"] = mmdet_builder


def _args_from_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    payload.setdefault("cfg", str(path))
    payload.setdefault("workers_per_gpu", None)
    return SimpleNamespace(**payload)


def validate_segmentation_configs():
    from tasks.segmentation import SegmentationTask

    paths = sorted(glob.glob(str(REPO_ROOT / "configs" / "seg_*.yaml")))
    if not paths:
        raise AssertionError("No segmentation configs found")

    for path in paths:
        args = _args_from_yaml(path)
        task = object.__new__(SegmentationTask)
        task.args = args
        task.run_name = Path(path).stem
        cfg = task._build_mmseg_config()

        assert cfg.model.type == "EncoderDecoder", path
        assert cfg.data.train.type == "ADE20KDataset", path
        assert cfg.data.workers_per_gpu == 4, path
        assert cfg.model.decode_head.num_classes == 150, path
        if args.model == "swin":
            assert cfg.model.backbone.type == "SwinTransformer", path
        else:
            assert cfg.model.backbone.type in {"ViTBackbone", "ViTCoPEBackbone", "ViTSCoPEBackbone"}, path
            assert cfg.model.backbone.image_size == getattr(args, "backbone_size", args.crop_size), path

    print(f"Validated {len(paths)} segmentation configs")


def validate_detection_configs():
    from tasks.detection import DetectionTask

    paths = sorted(glob.glob(str(REPO_ROOT / "configs" / "detection_*.yaml")))
    if not paths:
        raise AssertionError("No detection configs found")

    for path in paths:
        args = _args_from_yaml(path)
        task = object.__new__(DetectionTask)
        task.args = args
        task.run_name = Path(path).stem
        cfg = task._build_mmdet_config()

        assert cfg.model.type == "MaskRCNN", path
        assert cfg.data.train.type == "CocoDataset", path
        assert cfg.data.workers_per_gpu == 4, path
        assert cfg.model.roi_head.bbox_head.num_classes == 80, path
        assert cfg.model.roi_head.mask_head.num_classes == 80, path
        if args.model == "swin":
            assert cfg.model.backbone.type == "SwinTransformer", path
        else:
            assert cfg.model.backbone.type in {"ViTBackbone", "ViTCoPEBackbone", "ViTSCoPEBackbone"}, path
            assert cfg.model.backbone.image_size == args.size, path

    print(f"Validated {len(paths)} detection configs")


def validate_backbone_forwards():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for backbone forward validation") from exc

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    expected_hw = [(8, 8), (4, 4), (2, 2), (1, 1)]
    cases = [
        ("vit", ViTBackbone, {}),
        ("vitcope_no_cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("vitcope_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("vitscope_no_cls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("vitscope_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]

    count = 0
    with torch.no_grad():
        for style in ("resize", "simple_fpn"):
            for name, cls, extra in cases:
                model = cls(
                    image_size=32,
                    patch_size=16,
                    dim=16,
                    depth=4,
                    heads=2,
                    mlp_dim=32,
                    dim_head=8,
                    out_indices=(0, 1, 2, 3),
                    fpn_adapter_style=style,
                    **extra,
                )
                model.eval()
                outs = model(torch.randn(1, 3, 32, 32))
                assert len(outs) == 4, (name, style, len(outs))
                for idx, (out, hw) in enumerate(zip(outs, expected_hw)):
                    assert tuple(out.shape) == (1, 16, hw[0], hw[1]), (name, style, idx, tuple(out.shape))
                count += 1

    print(f"Validated {count} backbone forward smoke cases")


def main():
    os.chdir(REPO_ROOT)
    _install_openmmlab_stubs()
    validate_segmentation_configs()
    validate_detection_configs()
    validate_backbone_forwards()
    print("Segmentation and detection validation passed")


if __name__ == "__main__":
    main()
