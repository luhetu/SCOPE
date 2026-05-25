#!/usr/bin/env python3
"""Lightweight validation for segmentation and detection configuration paths.

This script avoids constructing full MMSegmentation/MMDetection trainers, which
requires mmcv-full C++ extensions. It stubs the OpenMMLab entry points used by
the task config builders, then validates every seg/det YAML and runs CPU
forward passes through this repository's custom dense-prediction backbones.
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
    """Small mmcv.Config substitute for task config builder smoke tests."""

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
        if module is None:
            def decorator(cls):
                self._register(name or cls.__name__, cls, force)
                return cls

            return decorator

        self._register(name or module.__name__, module, force)
        return module

    def _register(self, name, module, force):
        if not force and name in self.module_dict:
            return
        self.module_dict[name] = module


def _install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    sys.modules["mmcv"] = mmcv

    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object
    sys.modules["mmcv.parallel"] = mmcv_parallel

    mmseg_registry = Registry()
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
    mmseg_builder.BACKBONES = mmseg_registry
    mmseg_models.builder = mmseg_builder
    sys.modules["mmseg"] = mmseg
    sys.modules["mmseg.apis"] = mmseg_apis
    sys.modules["mmseg.datasets"] = mmseg_datasets
    sys.modules["mmseg.models"] = mmseg_models
    sys.modules["mmseg.models.builder"] = mmseg_builder

    mmdet_registry = Registry()
    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = lambda *args, **kwargs: None
    mmdet_apis.train_detector = lambda *args, **kwargs: None
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = lambda *args, **kwargs: None
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = lambda *args, **kwargs: None
    mmdet_builder = types.ModuleType("mmdet.models.builder")
    mmdet_builder.BACKBONES = mmdet_registry
    mmdet_models.builder = mmdet_builder
    sys.modules["mmdet"] = mmdet
    sys.modules["mmdet.apis"] = mmdet_apis
    sys.modules["mmdet.datasets"] = mmdet_datasets
    sys.modules["mmdet.models"] = mmdet_models
    sys.modules["mmdet.models.builder"] = mmdet_builder


def _namespace_from_yaml(path: Path) -> SimpleNamespace:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    # Match train.py parser defaults that are not represented in most YAMLs.
    data.setdefault("cfg", str(path))
    data.setdefault("resume", "")
    data.setdefault("workers_per_gpu", None)
    data.setdefault("run_tag", None)
    data.setdefault("time_profile", False)
    data.setdefault("time_profile_interval", 1000)
    return SimpleNamespace(**data)


def _build_task_config(task_cls, args: SimpleNamespace, build_method: str):
    task = task_cls.__new__(task_cls)
    task.args = args
    task.device = "cpu"
    task.run_name = task._build_run_name()
    return getattr(task, build_method)()


def validate_configs(verbose: bool = False) -> tuple[int, int]:
    _install_openmmlab_stubs()
    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    seg_paths = sorted(REPO_ROOT.glob("configs/seg_*.yaml"))
    det_paths = sorted(REPO_ROOT.glob("configs/detection_*.yaml"))

    for path in seg_paths:
        args = _namespace_from_yaml(path)
        cfg = _build_task_config(SegmentationTask, args, "_build_mmseg_config")
        assert cfg.model["type"] == "EncoderDecoder"
        assert cfg.data["workers_per_gpu"] == 4
        if verbose:
            print(f"seg config ok: {path.relative_to(REPO_ROOT)}")

    for path in det_paths:
        args = _namespace_from_yaml(path)
        cfg = _build_task_config(DetectionTask, args, "_build_mmdet_config")
        assert cfg.model["type"] == "MaskRCNN"
        assert cfg.data["workers_per_gpu"] == 4
        if verbose:
            print(f"det config ok: {path.relative_to(REPO_ROOT)}")

    return len(seg_paths), len(det_paths)


def validate_backbone_forwards(verbose: bool = False) -> int:
    import torch
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    base_kwargs = dict(
        image_size=32,
        patch_size=16,
        dim=32,
        depth=4,
        heads=2,
        mlp_dim=64,
        dim_head=16,
        out_indices=(0, 1, 2, 3),
    )
    expected_shapes = [(1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1)]
    cases = []
    for style in ("resize", "simple_fpn"):
        cases.append(("ViTBackbone", ViTBackbone, style, {}))
        for use_cls_token in (False, True):
            cases.append(("ViTCoPEBackbone", ViTCoPEBackbone, style, {"use_cls_token": use_cls_token}))
            cases.append(("ViTSCoPEBackbone", ViTSCoPEBackbone, style, {"use_cls_token": use_cls_token}))

    image = torch.randn(1, 3, 32, 32)
    for name, cls, style, extra_kwargs in cases:
        model = cls(**base_kwargs, fpn_adapter_style=style, **extra_kwargs)
        model.eval()
        with torch.no_grad():
            outputs = model(image)
        shapes = [tuple(out.shape) for out in outputs]
        if shapes != expected_shapes:
            raise AssertionError(f"{name} style={style} kwargs={extra_kwargs} produced {shapes}, expected {expected_shapes}")
        if verbose:
            print(f"backbone ok: {name} style={style} kwargs={extra_kwargs}")

    return len(cases)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate segmentation/detection config builders and custom backbones")
    parser.add_argument("--skip-backbone", action="store_true", help="Only validate seg/det config builders")
    parser.add_argument("--verbose", action="store_true", help="Print each validation case")
    args = parser.parse_args()

    seg_count, det_count = validate_configs(verbose=args.verbose)
    backbone_count = 0
    if not args.skip_backbone:
        backbone_count = validate_backbone_forwards(verbose=args.verbose)

    print(f"Validated {seg_count} segmentation configs, {det_count} detection configs, {backbone_count} backbone cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
