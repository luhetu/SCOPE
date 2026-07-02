#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight validation for segmentation and detection task wiring.

This script intentionally avoids constructing full MMSeg/MMDet models. The
local cloud/test environment often lacks compiled mmcv-full extensions, so it
uses small stubs to validate SCOPE's config builders and custom backbones.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Config(dict):
    """Small subset of mmcv.Config used by task config builders."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class Registry:
    def __init__(self, name):
        self.name = name
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        def _register(cls):
            key = name or cls.__name__
            if key in self.module_dict and not force:
                raise KeyError(f"{key} is already registered in {self.name}")
            self.module_dict[key] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


def _noop(*args, **kwargs):
    return None


def _install_openmmlab_stubs():
    """Install only the modules imported by tasks.segmentation/detection."""

    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object

    mmseg = types.ModuleType("mmseg")
    mmseg_apis = types.ModuleType("mmseg.apis")
    mmseg_apis.set_random_seed = _noop
    mmseg_apis.single_gpu_test = lambda *args, **kwargs: []
    mmseg_apis.train_segmentor = _noop
    mmseg_datasets = types.ModuleType("mmseg.datasets")
    mmseg_datasets.build_dataloader = _noop
    mmseg_datasets.build_dataset = _noop
    mmseg_models = types.ModuleType("mmseg.models")
    mmseg_models.build_segmentor = _noop
    mmseg_builder = types.ModuleType("mmseg.models.builder")
    mmseg_builder.BACKBONES = Registry("mmseg_backbone")
    mmseg_models.builder = mmseg_builder

    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = _noop
    mmdet_apis.train_detector = _noop
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = _noop
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = _noop
    mmdet_builder = types.ModuleType("mmdet.models.builder")
    mmdet_builder.BACKBONES = Registry("mmdet_backbone")
    mmdet_models.builder = mmdet_builder

    modules = {
        "mmcv": mmcv,
        "mmcv.parallel": mmcv_parallel,
        "mmseg": mmseg,
        "mmseg.apis": mmseg_apis,
        "mmseg.datasets": mmseg_datasets,
        "mmseg.models": mmseg_models,
        "mmseg.models.builder": mmseg_builder,
        "mmdet": mmdet,
        "mmdet.apis": mmdet_apis,
        "mmdet.datasets": mmdet_datasets,
        "mmdet.models": mmdet_models,
        "mmdet.models.builder": mmdet_builder,
    }
    sys.modules.update(modules)


def _load_args(path):
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    defaults = {
        "cfg": str(path),
        "workers_per_gpu": None,
        "seed": None,
        "run_tag": None,
        "amp": False,
        "nowandb": True,
        "warmup_epochs": 0,
        "n_epochs": 1,
        "bs": 1,
        "lr": 1e-4,
        "task": data.get("task"),
    }
    defaults.update(data)
    return SimpleNamespace(**defaults)


def _task_instance(task_cls, args):
    task = task_cls.__new__(task_cls)
    task.args = args
    task.device = "cpu"
    task.run_name = task._build_run_name()
    return task


def validate_configs():
    _install_openmmlab_stubs()

    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    config_paths = sorted(REPO_ROOT.glob("configs/seg_*.yaml")) + sorted(REPO_ROOT.glob("configs/detection_*.yaml"))
    if not config_paths:
        raise AssertionError("No segmentation or detection YAML configs found")

    for path in config_paths:
        args = _load_args(path)
        if path.name.startswith("seg_"):
            task = _task_instance(SegmentationTask, args)
            cfg = task._build_mmseg_config()
            assert cfg.model["type"] == "EncoderDecoder", path
            assert cfg.data["workers_per_gpu"] == 4, path
            assert len(cfg.model["decode_head"]["in_channels"]) == 4, path
            assert cfg.model["decode_head"]["num_classes"] == 150, path
        else:
            task = _task_instance(DetectionTask, args)
            cfg = task._build_mmdet_config()
            assert cfg.model["type"] == "MaskRCNN", path
            assert cfg.data["workers_per_gpu"] == 4, path
            assert cfg.model["roi_head"]["bbox_head"]["num_classes"] == 80, path
            assert cfg.model["roi_head"]["mask_head"]["num_classes"] == 80, path
        print(f"OK config: {path.relative_to(REPO_ROOT)}")


def validate_backbones():
    import torch
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    expected_shapes = [(1, 16, 8, 8), (1, 16, 4, 4), (1, 16, 2, 2), (1, 16, 1, 1)]
    cases = [
        ("vit", ViTBackbone, {}),
        ("vitcope_nocls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("vitcope_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("vitscope_nocls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("vitscope_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]

    torch.manual_seed(0)
    x = torch.randn(1, 3, 32, 32)
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
            ).eval()
            with torch.no_grad():
                outputs = model(x)
            shapes = [tuple(output.shape) for output in outputs]
            assert shapes == expected_shapes, f"{name}/{style}: {shapes}"
            print(f"OK backbone: {name}/{style} -> {shapes}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-backbone", action="store_true", help="Only validate task config builders")
    args = parser.parse_args()

    validate_configs()
    if not args.skip_backbone:
        validate_backbones()
    print("Segmentation/detection validation passed.")


if __name__ == "__main__":
    main()
