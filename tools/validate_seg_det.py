#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight segmentation/detection validation for this repository.

The full MMDetection/MMSegmentation training stack needs mmcv-full CUDA
extensions. This script stubs only the small API surface used while building
task configs, then runs CPU smoke tests for the custom dense ViT backbones.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Config(dict):
    """Small stand-in for mmcv.Config with attribute assignment support."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class Registry:
    def __init__(self):
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        def decorator(cls):
            key = name or cls.__name__
            if key in self.module_dict and not force:
                raise KeyError(f"{key} is already registered")
            self.module_dict[key] = cls
            return cls

        if module is not None:
            return decorator(module)
        return decorator


MMSEG_BACKBONES = Registry()
MMDET_BACKBONES = Registry()


def _install_mm_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config

    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object

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
    mmseg_models_builder = types.ModuleType("mmseg.models.builder")
    mmseg_models_builder.BACKBONES = MMSEG_BACKBONES

    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = lambda *args, **kwargs: None
    mmdet_apis.train_detector = lambda *args, **kwargs: None

    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = lambda *args, **kwargs: None

    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = lambda *args, **kwargs: None
    mmdet_models_builder = types.ModuleType("mmdet.models.builder")
    mmdet_models_builder.BACKBONES = MMDET_BACKBONES

    modules = {
        "mmcv": mmcv,
        "mmcv.parallel": mmcv_parallel,
        "mmseg": mmseg,
        "mmseg.apis": mmseg_apis,
        "mmseg.datasets": mmseg_datasets,
        "mmseg.models": mmseg_models,
        "mmseg.models.builder": mmseg_models_builder,
        "mmdet": mmdet,
        "mmdet.apis": mmdet_apis,
        "mmdet.datasets": mmdet_datasets,
        "mmdet.models": mmdet_models,
        "mmdet.models.builder": mmdet_models_builder,
    }
    sys.modules.update(modules)


def _load_args(config_path):
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    args = SimpleNamespace(
        cfg=str(config_path),
        resume="",
        workers_per_gpu=None,
        run_tag=None,
        seed=None,
        final_eval=False,
        time_profile=False,
        time_profile_interval=1000,
    )
    for key, value in raw.items():
        setattr(args, key, value)
    return args, raw


def _require_common_train_fields(config_path, raw):
    missing = [key for key in ("task", "model", "size", "patch", "bs", "lr") if key not in raw]
    if missing:
        raise AssertionError(f"{config_path}: missing train.py summary fields: {missing}")


def _build_seg_config(config_path, SegmentationTask):
    args, raw = _load_args(config_path)
    _require_common_train_fields(config_path, raw)

    task = SegmentationTask.__new__(SegmentationTask)
    task.args = args
    task.run_name = task._build_run_name()
    cfg = task._build_mmseg_config()

    assert cfg.model["type"] == "EncoderDecoder", config_path
    assert cfg.data["workers_per_gpu"] == 4, config_path
    assert cfg.data["train"]["type"] == "ADE20KDataset", config_path
    return cfg


def _build_det_config(config_path, DetectionTask):
    args, raw = _load_args(config_path)
    _require_common_train_fields(config_path, raw)

    task = DetectionTask.__new__(DetectionTask)
    task.args = args
    task.run_name = task._build_run_name()
    cfg = task._build_mmdet_config()

    assert cfg.model["type"] == "MaskRCNN", config_path
    assert cfg.data["workers_per_gpu"] == 4, config_path
    assert cfg.data["train"]["type"] == "CocoDataset", config_path
    return cfg


def validate_configs():
    _install_mm_stubs()

    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    seg_configs = sorted(ROOT.glob("configs/seg_*.yaml"))
    det_configs = sorted(ROOT.glob("configs/detection_*.yaml"))
    if not seg_configs or not det_configs:
        raise AssertionError("Expected segmentation and detection config files")

    for path in seg_configs:
        _build_seg_config(path, SegmentationTask)
        print(f"OK config: {path.relative_to(ROOT)}")
    for path in det_configs:
        _build_det_config(path, DetectionTask)
        print(f"OK config: {path.relative_to(ROOT)}")


def _check_feature_shapes(name, outputs, expected_hw):
    if len(outputs) != len(expected_hw):
        raise AssertionError(f"{name}: expected {len(expected_hw)} outputs, got {len(outputs)}")
    for idx, (feat, hw) in enumerate(zip(outputs, expected_hw)):
        shape = tuple(feat.shape)
        expected = (1, 32, hw, hw)
        if shape != expected:
            raise AssertionError(f"{name} output {idx}: expected {expected}, got {shape}")


def validate_backbones():
    _install_mm_stubs()

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    cases = [
        ("ViTBackbone", ViTBackbone, {}),
        ("ViTCoPEBackbone_cls_false", ViTCoPEBackbone, {"use_cls_token": False}),
        ("ViTCoPEBackbone_cls_true", ViTCoPEBackbone, {"use_cls_token": True}),
        ("ViTSCoPEBackbone_cls_false", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("ViTSCoPEBackbone_cls_true", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]
    expected_hw = (8, 4, 2, 1)
    image = torch.randn(1, 3, 32, 32)

    for adapter_style in ("resize", "simple_fpn"):
        for name, cls, extra in cases:
            model = cls(
                image_size=32,
                patch_size=16,
                dim=32,
                depth=4,
                heads=2,
                mlp_dim=64,
                dim_head=16,
                out_indices=(0, 1, 2, 3),
                fpn_adapter_style=adapter_style,
                **extra,
            )
            model.eval()
            with torch.no_grad():
                outputs = model(image)
            _check_feature_shapes(f"{name}_{adapter_style}", outputs, expected_hw)
            print(f"OK backbone: {name} ({adapter_style})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-configs", action="store_true", help="Skip task config builder validation")
    parser.add_argument("--skip-backbones", action="store_true", help="Skip custom backbone forward smoke tests")
    args = parser.parse_args()

    if not args.skip_configs:
        validate_configs()
    if not args.skip_backbones:
        validate_backbones()
    print("Segmentation/detection validation passed.")


if __name__ == "__main__":
    main()
