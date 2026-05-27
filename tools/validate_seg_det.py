#!/usr/bin/env python3
"""Lightweight validation for segmentation/detection config and backbones.

This script intentionally avoids constructing full MMSegmentation/MMDetection
trainers because those require compiled MMCV ops and datasets. It validates the
repo-owned pieces: YAML-driven config builders and custom dense-prediction
backbones.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Config(dict):
    """Small stand-in for mmcv.Config with attribute access."""

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
            if not force and key in self.module_dict:
                raise KeyError(f"{key} already registered in {self.name}")
            self.module_dict[key] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


def _install_openmmlab_stubs():
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
    mmseg_builder = types.ModuleType("mmseg.models.builder")
    mmseg_builder.BACKBONES = Registry("mmseg_backbone")

    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = lambda *args, **kwargs: None
    mmdet_apis.train_detector = lambda *args, **kwargs: None
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = lambda *args, **kwargs: None
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = lambda *args, **kwargs: None
    mmdet_builder = types.ModuleType("mmdet.models.builder")
    mmdet_builder.BACKBONES = Registry("mmdet_backbone")

    sys.modules.update(
        {
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
    )


def _load_args(path: Path) -> SimpleNamespace:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    args = SimpleNamespace(
        cfg=str(path),
        resume="",
        run_tag=None,
        seed=None,
        workers_per_gpu=None,
        time_profile=False,
        time_profile_interval=1000,
    )
    for key, value in data.items():
        setattr(args, key, value)

    # Simulate common argparse defaults so builders handle explicit None values.
    for key in (
        "checkpoint_interval",
        "eval_interval",
        "final_eval",
        "log_interval",
        "seg_head_dim",
        "seg_aux_dim",
        "seg_aux_in_index",
        "seg_neck_dim",
        "seg_neck_style",
        "backbone_size",
        "det_neck_type",
        "out_indices",
        "min_pretrained_match_rate",
    ):
        if not hasattr(args, key):
            setattr(args, key, None)

    return args


def _validate_configs():
    _install_openmmlab_stubs()
    segmentation = importlib.import_module("tasks.segmentation")
    detection = importlib.import_module("tasks.detection")

    seg_paths = sorted((REPO_ROOT / "configs").glob("seg_*.yaml"))
    det_paths = sorted((REPO_ROOT / "configs").glob("detection_*.yaml"))

    failures = []
    for path in seg_paths:
        args = _load_args(path)
        task = object.__new__(segmentation.SegmentationTask)
        task.args = args
        task.run_name = path.stem
        try:
            cfg = task._build_mmseg_config()
            backbone = cfg.model["backbone"]
            assert cfg.data["workers_per_gpu"] == 4
            assert backbone["type"] in {"SwinTransformer", "ViTBackbone", "ViTCoPEBackbone", "ViTSCoPEBackbone"}
        except Exception as exc:  # pragma: no cover - printed for CLI diagnosis
            failures.append((path, exc))

    for path in det_paths:
        args = _load_args(path)
        task = object.__new__(detection.DetectionTask)
        task.args = args
        task.run_name = path.stem
        try:
            cfg = task._build_mmdet_config()
            backbone = cfg.model["backbone"]
            assert cfg.data["workers_per_gpu"] == 4
            assert backbone["type"] in {"SwinTransformer", "ViTBackbone", "ViTCoPEBackbone", "ViTSCoPEBackbone"}
        except Exception as exc:  # pragma: no cover - printed for CLI diagnosis
            failures.append((path, exc))

    if failures:
        for path, exc in failures:
            print(f"FAIL config {path.relative_to(REPO_ROOT)}: {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    print(f"OK config builders: {len(seg_paths)} segmentation, {len(det_paths)} detection")


def _validate_backbones():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    cases = [
        ("vit", ViTBackbone, {}),
        ("vitcope_no_cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("vitcope_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("vitscope_no_cls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("vitscope_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]
    expected_shapes = [(1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1)]

    for adapter_style in ("resize", "simple_fpn"):
        for name, cls, extra_kwargs in cases:
            model = cls(
                image_size=32,
                patch_size=16,
                dim=32,
                depth=4,
                heads=4,
                mlp_dim=64,
                dim_head=8,
                out_indices=(0, 1, 2, 3),
                fpn_adapter_style=adapter_style,
                **extra_kwargs,
            )
            model.eval()
            with torch.no_grad():
                outputs = model(torch.randn(1, 3, 32, 32))
            shapes = [tuple(output.shape) for output in outputs]
            if shapes != expected_shapes:
                raise AssertionError(f"{name}/{adapter_style} produced {shapes}, expected {expected_shapes}")

    print(f"OK custom backbone forwards: {len(cases) * 2} cases")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-only", action="store_true", help="skip custom backbone forward tests")
    args = parser.parse_args()

    _validate_configs()
    if not args.configs_only:
        _validate_backbones()


if __name__ == "__main__":
    main()
