#!/usr/bin/env python3
"""Lightweight segmentation/detection validation.

This script validates the SCOPE-owned parts of the dense prediction stack
without requiring mmcv-full CUDA extensions or datasets:

* import segmentation/detection task modules with minimal OpenMMLab stubs;
* build every configs/seg_*.yaml and configs/detection_*.yaml task config;
* run CPU forward smoke tests for the custom ViT/CoPE/SCoPE backbones.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml


ROOT = Path(__file__).resolve().parents[1]


class AttrDict(dict):
    """A tiny mmcv.ConfigDict-compatible object for validation."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = _to_attr(value)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = _to_attr(value)
        return self


class Config(AttrDict):
    pass


def _to_attr(value):
    if isinstance(value, dict) and not isinstance(value, AttrDict):
        return AttrDict(value)
    if isinstance(value, list):
        return [_to_attr(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_attr(item) for item in value)
    return value


class Registry:
    def __init__(self):
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        def _register(cls):
            key = name or cls.__name__
            if key in self.module_dict and not force:
                raise KeyError(f"{key} is already registered")
            self.module_dict[key] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


def _noop(*args, **kwargs):
    return None


def _install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config

    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object
    mmcv.parallel = mmcv_parallel

    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.parallel"] = mmcv_parallel

    for package_name, train_name, build_model_name in (
        ("mmseg", "train_segmentor", "build_segmentor"),
        ("mmdet", "train_detector", "build_detector"),
    ):
        root = types.ModuleType(package_name)
        apis = types.ModuleType(f"{package_name}.apis")
        datasets = types.ModuleType(f"{package_name}.datasets")
        models = types.ModuleType(f"{package_name}.models")
        builder = types.ModuleType(f"{package_name}.models.builder")

        setattr(apis, "set_random_seed", _noop)
        setattr(apis, train_name, _noop)
        if package_name == "mmseg":
            setattr(apis, "single_gpu_test", lambda *args, **kwargs: [])
            datasets.build_dataloader = _noop
        datasets.build_dataset = _noop
        setattr(models, build_model_name, _noop)
        builder.BACKBONES = Registry()
        models.builder = builder

        root.apis = apis
        root.datasets = datasets
        root.models = models

        sys.modules[package_name] = root
        sys.modules[f"{package_name}.apis"] = apis
        sys.modules[f"{package_name}.datasets"] = datasets
        sys.modules[f"{package_name}.models"] = models
        sys.modules[f"{package_name}.models.builder"] = builder


def _load_args(path: Path) -> SimpleNamespace:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = {
        "cfg": str(path),
        "resume": "",
        "workers_per_gpu": None,
        "model": None,
        "data_dir": None,
        "time_profile": False,
        "time_profile_interval": 1000,
        "checkpoint_interval": None,
        "eval_interval": None,
        "log_interval": None,
        "seg_head_dim": None,
        "seg_aux_dim": None,
        "seg_aux_in_index": None,
        "seg_neck_dim": None,
    }
    defaults.update(data)
    return SimpleNamespace(**defaults)


def _validate_configs():
    _install_openmmlab_stubs()

    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    config_paths = sorted(ROOT.glob("configs/seg_*.yaml")) + sorted(ROOT.glob("configs/detection_*.yaml"))
    if not config_paths:
        raise AssertionError("No segmentation or detection configs found")

    validated = []
    for path in config_paths:
        args = _load_args(path)
        if args.task == "seg":
            task = object.__new__(SegmentationTask)
            task.args = args
            task.run_name = task._build_run_name()
            cfg = task._build_mmseg_config()
        elif args.task == "det":
            if not hasattr(args, "size"):
                raise AssertionError(f"{path} must define size because train.py reports it")
            task = object.__new__(DetectionTask)
            task.args = args
            task.run_name = task._build_run_name()
            cfg = task._build_mmdet_config()
        else:
            raise AssertionError(f"Unexpected task in {path}: {args.task}")

        if cfg.data.workers_per_gpu != 4:
            raise AssertionError(f"{path} workers_per_gpu default should resolve to 4")
        if not isinstance(cfg.model, dict):
            raise AssertionError(f"{path} model config was not built")
        validated.append(path.relative_to(ROOT).as_posix())

    print("Validated configs:")
    for item in validated:
        print(f"  - {item}")


def _validate_backbones():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    image = torch.randn(1, 3, 32, 32)
    expected_hw = [(8, 8), (4, 4), (2, 2), (1, 1)]
    cases = [
        ("vit", ViTBackbone, {}),
        ("vitcope_no_cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("vitcope_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("vitscope_no_cls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("vitscope_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]

    for style in ("resize", "simple_fpn"):
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
                fpn_adapter_style=style,
                **extra,
            )
            model.eval()
            with torch.no_grad():
                outputs = model(image)
            shapes = [tuple(output.shape) for output in outputs]
            expected = [(1, 32, h, w) for h, w in expected_hw]
            if shapes != expected:
                raise AssertionError(f"{name}/{style} produced {shapes}, expected {expected}")

    print("Validated custom backbone CPU forward paths")


def main():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    _validate_configs()
    _validate_backbones()
    print("Segmentation and detection validation passed")


if __name__ == "__main__":
    main()
