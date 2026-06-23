#!/usr/bin/env python3
"""Lightweight segmentation/detection validation for local smoke checks.

This script validates the repository-owned configuration and custom backbone
paths without requiring ADE20K/COCO data or a compiled mmcv-full installation.
It installs small in-process stubs for the OpenMMLab APIs that the task modules
import, then exercises their private config builders across every seg/det YAML.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ConfigDict(dict):
    """Attribute-access dict close enough for the task config builders."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = _to_config_value(value)


class Config(ConfigDict):
    pass


def _to_config_value(value):
    if isinstance(value, dict) and not isinstance(value, ConfigDict):
        return ConfigDict({key: _to_config_value(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_to_config_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_config_value(item) for item in value)
    return value


class Registry:
    def __init__(self):
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False, **_kwargs):
        def _register(cls):
            module_name = name or cls.__name__
            if module_name in self.module_dict and not force:
                raise KeyError(f"{module_name} is already registered")
            self.module_dict[module_name] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


def _noop(*_args, **_kwargs):
    return None


def _install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    parallel = types.ModuleType("mmcv.parallel")
    parallel.MMDataParallel = object
    mmcv.parallel = parallel
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.parallel"] = parallel

    for package_name in ("mmseg", "mmdet"):
        package = types.ModuleType(package_name)
        apis = types.ModuleType(f"{package_name}.apis")
        datasets = types.ModuleType(f"{package_name}.datasets")
        models = types.ModuleType(f"{package_name}.models")
        builder = types.ModuleType(f"{package_name}.models.builder")

        builder.BACKBONES = Registry()
        apis.set_random_seed = _noop
        apis.single_gpu_test = _noop
        apis.train_segmentor = _noop
        apis.train_detector = _noop
        datasets.build_dataloader = _noop
        datasets.build_dataset = _noop
        models.build_segmentor = _noop
        models.build_detector = _noop
        models.builder = builder

        package.apis = apis
        package.datasets = datasets
        package.models = models

        sys.modules[package_name] = package
        sys.modules[f"{package_name}.apis"] = apis
        sys.modules[f"{package_name}.datasets"] = datasets
        sys.modules[f"{package_name}.models"] = models
        sys.modules[f"{package_name}.models.builder"] = builder


def _load_args(config_path: Path) -> SimpleNamespace:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    args = SimpleNamespace(
        cfg=str(config_path),
        resume="",
        workers_per_gpu=None,
        data_dir=None,
        model=None,
        time_profile=False,
        time_profile_interval=1000,
    )
    for key, value in data.items():
        setattr(args, key, value)
    return args


def _require_fields(args: SimpleNamespace, config_path: Path):
    common = ("task", "model", "size", "patch", "bs", "lr", "n_epochs")
    missing = [name for name in common if not hasattr(args, name)]
    if missing:
        raise AssertionError(f"{config_path}: missing required fields: {', '.join(missing)}")

    if args.model == "swin":
        model_fields = ("embed_dim", "depths", "num_heads", "window_size")
    else:
        model_fields = ("dim", "depth", "heads", "mlp_dim")
    missing = [name for name in model_fields if not hasattr(args, name)]
    if missing:
        raise AssertionError(f"{config_path}: missing {args.model} fields: {', '.join(missing)}")


def _task_probe(task_cls, args: SimpleNamespace, builder_name: str):
    task = task_cls.__new__(task_cls)
    task.args = args
    task.run_name = Path(args.cfg).stem
    return getattr(task, builder_name)()


def _validate_seg_config(config_path: Path):
    args = _load_args(config_path)
    _require_fields(args, config_path)
    module = importlib.import_module("tasks.segmentation")
    cfg = _task_probe(module.SegmentationTask, args, "_build_mmseg_config")

    assert cfg.model.type == "EncoderDecoder"
    assert cfg.model.decode_head.type == "UPerHead"
    assert cfg.model.decode_head.num_classes == 150
    assert cfg.data.workers_per_gpu == 4
    assert cfg.data.train.type == "ADE20KDataset"
    assert cfg.data.val.type == "ADE20KDataset"
    assert len(cfg.model.decode_head.in_channels) == 4
    return cfg


def _validate_det_config(config_path: Path):
    args = _load_args(config_path)
    _require_fields(args, config_path)
    module = importlib.import_module("tasks.detection")
    cfg = _task_probe(module.DetectionTask, args, "_build_mmdet_config")

    assert cfg.model.type == "MaskRCNN"
    assert cfg.model.roi_head.bbox_head.num_classes == 80
    assert cfg.model.roi_head.mask_head.num_classes == 80
    assert cfg.data.workers_per_gpu == 4
    assert cfg.data.train.type == "CocoDataset"
    assert cfg.data.val.type == "CocoDataset"
    assert cfg.model.backbone.type in {"SwinTransformer", "ViTBackbone", "ViTCoPEBackbone", "ViTSCoPEBackbone"}
    return cfg


def _validate_configs():
    config_paths = sorted(ROOT.glob("configs/seg_*.yaml")) + sorted(ROOT.glob("configs/detection_*.yaml"))
    if not config_paths:
        raise AssertionError("No segmentation or detection configs found")

    checked = []
    for config_path in config_paths:
        args = _load_args(config_path)
        if args.task == "seg":
            _validate_seg_config(config_path)
        elif args.task == "det":
            _validate_det_config(config_path)
        else:
            raise AssertionError(f"{config_path}: unexpected task {args.task!r}")
        checked.append(config_path.relative_to(ROOT))
    return checked


def _check_backbone_shapes(name, outputs, expected_hw):
    shapes = [tuple(output.shape) for output in outputs]
    if len(shapes) != 4:
        raise AssertionError(f"{name}: expected 4 feature maps, got {len(shapes)}")
    for shape, hw in zip(shapes, expected_hw):
        if shape != (1, 32, hw, hw):
            raise AssertionError(f"{name}: expected (1, 32, {hw}, {hw}), got {shape}")


def _validate_backbone_forward():
    try:
        import torch
        from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone
    except Exception as exc:
        return f"SKIP backbone forward smoke test: {type(exc).__name__}: {exc}"

    cases = [
        ("ViTBackbone", ViTBackbone, {}),
        ("ViTCoPEBackbone_no_cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("ViTCoPEBackbone_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("ViTSCoPEBackbone_no_cls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("ViTSCoPEBackbone_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]
    expected_hw = [8, 4, 2, 1]
    image = torch.randn(1, 3, 32, 32)

    for style in ("resize", "simple_fpn"):
        for case_name, cls, extra_kwargs in cases:
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
                **extra_kwargs,
            ).eval()
            with torch.no_grad():
                outputs = model(image)
            _check_backbone_shapes(f"{case_name}:{style}", outputs, expected_hw)
    return "PASS backbone forward smoke test"


def main():
    parser = argparse.ArgumentParser(description="Validate SCOPE segmentation/detection code paths")
    parser.add_argument("--skip-backbone", action="store_true", help="Only validate seg/det config builders")
    args = parser.parse_args()

    _install_openmmlab_stubs()

    checked = _validate_configs()
    print(f"PASS config builders: {len(checked)} configs")
    for path in checked:
        print(f"  - {path}")

    if not args.skip_backbone:
        print(_validate_backbone_forward())


if __name__ == "__main__":
    main()
