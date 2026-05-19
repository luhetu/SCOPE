#!/usr/bin/env python3
"""Lightweight segmentation/detection verification for this repository.

The full training/evaluation paths require mmcv-full, CUDA, and ADE20K/COCO
data. This script verifies the local integration code that can run on CPU:

* every configs/seg_*.yaml and configs/detection_*.yaml builds its task config
* CLI defaults such as workers_per_gpu=None fall back to task defaults
* custom ViT/CoPE/SCoPE dense backbones produce the expected feature pyramid

OpenMMLab modules are stubbed only far enough to import the task glue and
custom backbones; no dataset or model training is performed.
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


ROOT = Path(__file__).resolve().parents[1]


class ConfigDict(dict):
    """Small subset of mmcv.ConfigDict used by the task builders."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = _to_config(value)

    def update(self, *args, **kwargs):
        values = dict(*args, **kwargs)
        for key, value in values.items():
            self[key] = _to_config(value)


class Config(ConfigDict):
    pass


def _to_config(value):
    if isinstance(value, dict) and not isinstance(value, ConfigDict):
        return ConfigDict(value)
    if isinstance(value, list):
        return [_to_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_config(item) for item in value)
    return value


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


def _install_openmmlab_stubs():
    mmdet_backbones = Registry("mmdet_backbones")
    mmseg_backbones = Registry("mmseg_backbones")

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
    mmseg_datasets.build_dataloader = lambda *args, **kwargs: []
    mmseg_datasets.build_dataset = lambda *args, **kwargs: SimpleNamespace(CLASSES=(), PALETTE=())
    mmseg_models = types.ModuleType("mmseg.models")
    mmseg_models.build_segmentor = lambda *args, **kwargs: SimpleNamespace()
    mmseg_builder = types.ModuleType("mmseg.models.builder")
    mmseg_builder.BACKBONES = mmseg_backbones
    mmseg_models.builder = mmseg_builder

    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = lambda *args, **kwargs: None
    mmdet_apis.train_detector = lambda *args, **kwargs: None
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = lambda *args, **kwargs: SimpleNamespace(CLASSES=())
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = lambda *args, **kwargs: SimpleNamespace()
    mmdet_builder = types.ModuleType("mmdet.models.builder")
    mmdet_builder.BACKBONES = mmdet_backbones
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


def _load_args(config_path: Path) -> SimpleNamespace:
    # Match train.py parser defaults closely enough to catch None-default bugs.
    values = {
        "cfg": str(config_path),
        "resume": "",
        "workers_per_gpu": None,
        "model": None,
        "data_dir": None,
        "time_profile": False,
        "time_profile_interval": 1000,
    }
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for key, value in raw.items():
        values[key] = None if value == "null" else value
    return SimpleNamespace(**values)


def _quiet_call(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def _build_task_config(task_cls, args: SimpleNamespace, method_name: str):
    task = task_cls.__new__(task_cls)
    task.args = args
    task.device = "cpu"
    task.run_name = task._build_run_name()
    return _quiet_call(lambda: getattr(task, method_name)())


def _validate_segmentation_configs() -> int:
    from tasks.segmentation import SegmentationTask

    count = 0
    for path in sorted((ROOT / "configs").glob("seg_*.yaml")):
        args = _load_args(path)
        if args.task != "seg":
            raise AssertionError(f"{path} should declare task: seg")
        cfg = _build_task_config(SegmentationTask, args, "_build_mmseg_config")
        if cfg.model.type != "EncoderDecoder":
            raise AssertionError(f"{path}: expected EncoderDecoder, got {cfg.model.type}")
        if cfg.data.workers_per_gpu != 4:
            raise AssertionError(f"{path}: workers_per_gpu default did not resolve to 4")
        if not cfg.data.train.pipeline or not cfg.data.val.pipeline:
            raise AssertionError(f"{path}: missing segmentation train/val pipeline")
        count += 1
    return count


def _validate_detection_configs() -> int:
    from tasks.detection import DetectionTask

    count = 0
    for path in sorted((ROOT / "configs").glob("detection_*.yaml")):
        args = _load_args(path)
        if args.task != "det":
            raise AssertionError(f"{path} should declare task: det")
        cfg = _build_task_config(DetectionTask, args, "_build_mmdet_config")
        if cfg.model.type != "MaskRCNN":
            raise AssertionError(f"{path}: expected MaskRCNN, got {cfg.model.type}")
        if cfg.data.workers_per_gpu != 4:
            raise AssertionError(f"{path}: workers_per_gpu default did not resolve to 4")
        if not cfg.data.train.pipeline or not cfg.data.val.pipeline:
            raise AssertionError(f"{path}: missing detection train/val pipeline")
        if args.model == "swin" and cfg.model.backbone.type != "SwinTransformer":
            raise AssertionError(f"{path}: expected SwinTransformer backbone")
        count += 1
    return count


def _validate_backbone_forwards() -> int:
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    torch.set_num_threads(1)
    x = torch.randn(1, 3, 32, 32)
    expected_shapes = [
        (1, 32, 8, 8),
        (1, 32, 4, 4),
        (1, 32, 2, 2),
        (1, 32, 1, 1),
    ]
    cases = [
        (ViTBackbone, {}),
        (ViTCoPEBackbone, {"use_cls_token": False}),
        (ViTCoPEBackbone, {"use_cls_token": True}),
        (ViTSCoPEBackbone, {"use_cls_token": False}),
        (ViTSCoPEBackbone, {"use_cls_token": True}),
    ]

    count = 0
    for adapter_style in ("resize", "simple_fpn"):
        for cls, extra_kwargs in cases:
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
            ).eval()
            with torch.no_grad():
                outputs = model(x)
            shapes = [tuple(output.shape) for output in outputs]
            if shapes != expected_shapes:
                name = cls.__name__
                raise AssertionError(f"{name} {adapter_style}: expected {expected_shapes}, got {shapes}")
            count += 1
    return count


def main() -> int:
    sys.path.insert(0, str(ROOT))
    _install_openmmlab_stubs()

    seg_count = _validate_segmentation_configs()
    det_count = _validate_detection_configs()
    backbone_count = _validate_backbone_forwards()

    print(f"Validated {seg_count} segmentation configs")
    print(f"Validated {det_count} detection configs")
    print(f"Validated {backbone_count} custom backbone forward cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
