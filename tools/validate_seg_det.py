#!/usr/bin/env python3
"""Lightweight segmentation/detection validation for this repository.

The full ADE20K/COCO training paths require mmcv-full CUDA extensions and
datasets. This script isolates the repository-owned config builders and custom
backbones so they can be checked in a basic CPU environment.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import types
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ConfigDict(dict):
    """Small subset of mmcv ConfigDict with attribute access."""

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, ConfigDict):
            return value
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._wrap(item) for item in value)
        return value

    def __init__(self, *args, **kwargs):
        super().__init__()
        data = dict(*args, **kwargs)
        for key, value in data.items():
            self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __setitem__(self, key, value):
        super().__setitem__(key, self._wrap(value))


class Config(ConfigDict):
    pass


class Registry:
    def __init__(self):
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False, **kwargs):
        def _register(cls):
            module_name = name or cls.__name__
            if not force and module_name in self.module_dict:
                raise KeyError(f"{module_name} is already registered")
            self.module_dict[module_name] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


class DummyModel:
    CLASSES = ()

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict, strict=False):
        return None


class DummyDataset:
    CLASSES = ("dummy",)
    PALETTE = None

    def evaluate(self, *args, **kwargs):
        return {}


class DummyDataParallel:
    def __init__(self, module, *args, **kwargs):
        self.module = module

    def __getattr__(self, key):
        return getattr(self.module, key)


def _install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = DummyDataParallel
    mmcv.parallel = mmcv_parallel
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.parallel"] = mmcv_parallel

    mmdet_backbones = Registry()
    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = lambda *args, **kwargs: None
    mmdet_apis.train_detector = lambda *args, **kwargs: None
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = lambda *args, **kwargs: DummyDataset()
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = lambda *args, **kwargs: DummyModel()
    mmdet_models_builder = types.ModuleType("mmdet.models.builder")
    mmdet_models_builder.BACKBONES = mmdet_backbones
    mmdet.models = mmdet_models
    mmdet.datasets = mmdet_datasets
    mmdet.apis = mmdet_apis
    mmdet_models.builder = mmdet_models_builder
    sys.modules["mmdet"] = mmdet
    sys.modules["mmdet.apis"] = mmdet_apis
    sys.modules["mmdet.datasets"] = mmdet_datasets
    sys.modules["mmdet.models"] = mmdet_models
    sys.modules["mmdet.models.builder"] = mmdet_models_builder

    mmseg_backbones = Registry()
    mmseg = types.ModuleType("mmseg")
    mmseg_apis = types.ModuleType("mmseg.apis")
    mmseg_apis.set_random_seed = lambda *args, **kwargs: None
    mmseg_apis.single_gpu_test = lambda *args, **kwargs: []
    mmseg_apis.train_segmentor = lambda *args, **kwargs: None
    mmseg_datasets = types.ModuleType("mmseg.datasets")
    mmseg_datasets.build_dataset = lambda *args, **kwargs: DummyDataset()
    mmseg_datasets.build_dataloader = lambda *args, **kwargs: []
    mmseg_models = types.ModuleType("mmseg.models")
    mmseg_models.build_segmentor = lambda *args, **kwargs: DummyModel()
    mmseg_models_builder = types.ModuleType("mmseg.models.builder")
    mmseg_models_builder.BACKBONES = mmseg_backbones
    mmseg.models = mmseg_models
    mmseg.datasets = mmseg_datasets
    mmseg.apis = mmseg_apis
    mmseg_models.builder = mmseg_models_builder
    sys.modules["mmseg"] = mmseg
    sys.modules["mmseg.apis"] = mmseg_apis
    sys.modules["mmseg.datasets"] = mmseg_datasets
    sys.modules["mmseg.models"] = mmseg_models
    sys.modules["mmseg.models.builder"] = mmseg_models_builder


def _load_args(config_path: str) -> argparse.Namespace:
    defaults = dict(
        cfg=config_path,
        resume="",
        workers_per_gpu=None,
        model=None,
        data_dir=None,
        time_profile=False,
        time_profile_interval=1000,
        run_tag=None,
    )
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle) or {}
    args = argparse.Namespace(**defaults)
    for key, value in raw_cfg.items():
        if key == "pretrained" and value == "null":
            value = None
        setattr(args, key, value)
    return args


def _validate_segmentation_configs(paths):
    from tasks.segmentation import SegmentationTask

    for path in paths:
        args = _load_args(path)
        if args.task != "seg":
            raise AssertionError(f"{path} has task={args.task!r}, expected 'seg'")
        task = SegmentationTask(args)
        if task.cfg.data.workers_per_gpu != 4:
            raise AssertionError(f"{path} did not default workers_per_gpu to 4")
        if not task.cfg.model.backbone.type:
            raise AssertionError(f"{path} produced an invalid backbone config")
    print(f"Validated {len(paths)} segmentation configs")


def _validate_detection_configs(paths):
    from tasks.detection import DetectionTask

    for path in paths:
        args = _load_args(path)
        if args.task != "det":
            raise AssertionError(f"{path} has task={args.task!r}, expected 'det'")
        task = DetectionTask(args)
        if task.cfg.data.workers_per_gpu != 4:
            raise AssertionError(f"{path} did not default workers_per_gpu to 4")
        if args.model == "swin" and task.cfg.model.backbone.type != "SwinTransformer":
            raise AssertionError(f"{path} produced an invalid Swin backbone config")
        if not task.cfg.model.backbone.type:
            raise AssertionError(f"{path} produced an invalid backbone config")
    print(f"Validated {len(paths)} detection configs")


def _assert_feature_shapes(name, outputs, expected_shapes):
    if not isinstance(outputs, tuple):
        raise AssertionError(f"{name} returned {type(outputs).__name__}, expected tuple")
    actual_shapes = [tuple(output.shape[-2:]) for output in outputs]
    if actual_shapes != expected_shapes:
        raise AssertionError(f"{name} feature shapes {actual_shapes}, expected {expected_shapes}")


def _validate_backbone_smoke_tests():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    torch.set_num_threads(1)
    image = torch.randn(2, 3, 32, 32)
    expected_shapes = [(8, 8), (4, 4), (2, 2), (1, 1)]
    common = dict(
        image_size=32,
        patch_size=16,
        dim=48,
        depth=4,
        heads=3,
        mlp_dim=96,
        dim_head=16,
        out_indices=(0, 1, 2, 3),
    )

    cases = [
        ("ViTBackbone", ViTBackbone, {}),
        ("ViTCoPEBackbone_no_cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("ViTCoPEBackbone_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("ViTSCoPEBackbone_no_cls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("ViTSCoPEBackbone_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]

    with torch.no_grad():
        for adapter_style in ("resize", "simple_fpn"):
            for name, cls, extra_kwargs in cases:
                model = cls(**common, fpn_adapter_style=adapter_style, **extra_kwargs).eval()
                outputs = model(image)
                _assert_feature_shapes(f"{name}_{adapter_style}", outputs, expected_shapes)
    print("Validated custom dense backbone CPU forward smoke tests")


def main():
    _install_openmmlab_stubs()
    seg_paths = sorted(glob.glob(os.path.join(ROOT, "configs", "seg_*.yaml")))
    det_paths = sorted(glob.glob(os.path.join(ROOT, "configs", "detection_*.yaml")))
    if not seg_paths:
        raise AssertionError("No segmentation configs found")
    if not det_paths:
        raise AssertionError("No detection configs found")

    _validate_segmentation_configs(seg_paths)
    _validate_detection_configs(det_paths)
    _validate_backbone_smoke_tests()
    print("Segmentation/detection validation passed")


if __name__ == "__main__":
    main()
