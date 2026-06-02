#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight validation for segmentation/detection config and backbone code.

The training entrypoints depend on mmcv-full C++/CUDA extensions that are not
usually available in lightweight CI or cloud smoke-test environments. This
script stubs only the OpenMMLab APIs needed to import the task modules, then
checks that every seg_/detection_ YAML can build its runtime config and that
the custom ViT-family dense-prediction backbones run a CPU forward pass.
"""

from __future__ import annotations

import contextlib
import io
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Config(dict):
    """Small mmcv.Config substitute with attribute access."""

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
            if not force and key in self.module_dict:
                return cls
            self.module_dict[key] = cls
            return cls

        if module is not None:
            key = name or module.__name__
            if force or key not in self.module_dict:
                self.module_dict[key] = module
            return module
        return decorator


def _install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv.parallel = types.ModuleType("mmcv.parallel")
    mmcv.parallel.MMDataParallel = object
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.parallel"] = mmcv.parallel

    for root in ("mmdet", "mmseg"):
        sys.modules[root] = types.ModuleType(root)
        sys.modules[f"{root}.apis"] = types.ModuleType(f"{root}.apis")
        sys.modules[f"{root}.datasets"] = types.ModuleType(f"{root}.datasets")
        sys.modules[f"{root}.models"] = types.ModuleType(f"{root}.models")
        sys.modules[f"{root}.models.builder"] = types.ModuleType(f"{root}.models.builder")
        sys.modules[f"{root}.models.builder"].BACKBONES = Registry()

    sys.modules["mmdet.apis"].set_random_seed = lambda *args, **kwargs: None
    sys.modules["mmdet.apis"].train_detector = lambda *args, **kwargs: None
    sys.modules["mmdet.datasets"].build_dataset = lambda *args, **kwargs: None
    sys.modules["mmdet.models"].build_detector = lambda *args, **kwargs: None

    sys.modules["mmseg.apis"].set_random_seed = lambda *args, **kwargs: None
    sys.modules["mmseg.apis"].single_gpu_test = lambda *args, **kwargs: None
    sys.modules["mmseg.apis"].train_segmentor = lambda *args, **kwargs: None
    sys.modules["mmseg.datasets"].build_dataloader = lambda *args, **kwargs: None
    sys.modules["mmseg.datasets"].build_dataset = lambda *args, **kwargs: None
    sys.modules["mmseg.models"].build_segmentor = lambda *args, **kwargs: None


def _load_args(path: Path) -> SimpleNamespace:
    # Mirror train.py parser defaults that can otherwise mask task defaults with None.
    defaults = dict(
        cfg=str(path),
        resume="",
        workers_per_gpu=None,
        model=None,
        data_dir=None,
        time_profile=False,
        time_profile_interval=1000,
    )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults.update(data)
    return SimpleNamespace(**defaults)


def _build_task_config(task_cls, path: Path, method_name: str):
    args = _load_args(path)
    task = object.__new__(task_cls)
    task.args = args
    task.run_name = task._build_run_name()
    with contextlib.redirect_stdout(io.StringIO()):
        return getattr(task, method_name)()


def validate_configs():
    _install_openmmlab_stubs()

    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    seg_paths = sorted((REPO_ROOT / "configs").glob("seg_*.yaml"))
    det_paths = sorted((REPO_ROOT / "configs").glob("detection_*.yaml"))
    failures = []

    for path in seg_paths:
        try:
            cfg = _build_task_config(SegmentationTask, path, "_build_mmseg_config")
            assert cfg.model["type"] == "EncoderDecoder"
            assert cfg.data["train"]["type"] == "ADE20KDataset"
            assert cfg.data["workers_per_gpu"] == 4
        except Exception as exc:
            failures.append((path, exc))

    for path in det_paths:
        try:
            cfg = _build_task_config(DetectionTask, path, "_build_mmdet_config")
            assert cfg.model["type"] == "MaskRCNN"
            assert cfg.data["train"]["type"] == "CocoDataset"
            assert cfg.data["workers_per_gpu"] == 4
        except Exception as exc:
            failures.append((path, exc))

    if failures:
        for path, exc in failures:
            print(f"FAIL config {path.relative_to(REPO_ROOT)}: {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    print(f"Validated configs: {len(seg_paths)} segmentation, {len(det_paths)} detection")


def validate_backbones():
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    torch.set_num_threads(1)
    x = torch.randn(1, 3, 32, 32)
    expected_shapes = [(1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1)]
    cases = [
        ("vit", ViTBackbone, {}),
        ("vitcope_no_cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("vitcope_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("vitscope_no_cls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("vitscope_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]

    checked = 0
    for style in ("resize", "simple_fpn"):
        for name, cls, extra in cases:
            model = cls(
                image_size=32,
                patch_size=16,
                dim=32,
                depth=4,
                heads=4,
                mlp_dim=64,
                dim_head=8,
                out_indices=(0, 1, 2, 3),
                fpn_adapter_style=style,
                **extra,
            ).eval()
            with torch.no_grad():
                outputs = model(x)
            shapes = [tuple(out.shape) for out in outputs]
            if shapes != expected_shapes:
                raise SystemExit(f"FAIL backbone {name}/{style}: {shapes} != {expected_shapes}")
            checked += 1

    print(f"Validated backbones: {checked} CPU forward cases")


def main():
    validate_configs()
    validate_backbones()
    print("Segmentation/detection smoke validation passed.")


if __name__ == "__main__":
    main()
