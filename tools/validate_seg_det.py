#!/usr/bin/env python3
"""Lightweight segmentation/detection validation for local development.

The full training stack needs mmcv-full plus ADE20K/COCO data. This script
checks repo-local glue without those heavy runtime requirements: YAML loading,
task config builders, and (when torch/timm/einops are installed) custom
backbone forward shapes.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Config(dict):
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
        def _register(cls):
            key = name or cls.__name__
            if not force and key in self.module_dict:
                return cls
            self.module_dict[key] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


@contextmanager
def argv(args: Iterable[str]):
    old = sys.argv[:]
    sys.argv = [old[0], *args]
    try:
        yield
    finally:
        sys.argv = old


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _install_stubs(use_dummy_backbones: bool) -> None:
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    sys.modules["mmcv"] = mmcv

    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object
    sys.modules["mmcv.parallel"] = mmcv_parallel

    for package in ("mmseg", "mmdet"):
        root = types.ModuleType(package)
        apis = types.ModuleType(f"{package}.apis")
        datasets = types.ModuleType(f"{package}.datasets")
        models = types.ModuleType(f"{package}.models")
        builder = types.ModuleType(f"{package}.models.builder")

        registry = Registry()
        builder.BACKBONES = registry
        apis.set_random_seed = lambda *args, **kwargs: None
        datasets.build_dataset = lambda *args, **kwargs: None

        if package == "mmseg":
            apis.single_gpu_test = lambda *args, **kwargs: []
            apis.train_segmentor = lambda *args, **kwargs: None
            datasets.build_dataloader = lambda *args, **kwargs: None
            models.build_segmentor = lambda *args, **kwargs: None
        else:
            apis.train_detector = lambda *args, **kwargs: None
            models.build_detector = lambda *args, **kwargs: None

        root.apis = apis
        root.datasets = datasets
        root.models = models
        models.builder = builder

        sys.modules[package] = root
        sys.modules[f"{package}.apis"] = apis
        sys.modules[f"{package}.datasets"] = datasets
        sys.modules[f"{package}.models"] = models
        sys.modules[f"{package}.models.builder"] = builder

    if use_dummy_backbones:
        vit_backbone = types.ModuleType("models.vit_backbone")

        class _DummyBackbone:
            pass

        vit_backbone.ViTBackbone = _DummyBackbone
        vit_backbone.ViTCoPEBackbone = _DummyBackbone
        vit_backbone.ViTSCoPEBackbone = _DummyBackbone
        sys.modules["models.vit_backbone"] = vit_backbone


def _parser_for_config(path: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Validate {path.name}")
    parser.add_argument("--cfg", type=str, default=str(path))
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--workers_per_gpu", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--time_profile", action="store_true")
    parser.add_argument("--time_profile_interval", type=int, default=1000)
    return parser


def _load_args(path: Path):
    from utils.cfg import load_cfg

    with argv(["--cfg", str(path)]):
        return load_cfg(_parser_for_config(path))


def _require(args, names: Iterable[str], path: Path) -> None:
    missing = [name for name in names if getattr(args, name, None) is None]
    if missing:
        raise AssertionError(f"{path}: missing required fields: {', '.join(missing)}")


def _validate_common_fields(args, path: Path) -> None:
    _require(args, ("task", "model", "data_dir", "bs", "lr"), path)
    if args.task not in {"seg", "det"}:
        raise AssertionError(f"{path}: expected task seg/det, got {args.task!r}")

    # train.py prints size/patch for dense tasks before dispatching to the task.
    _require(args, ("size", "patch"), path)

    if args.model == "swin":
        _require(args, ("embed_dim", "depths", "num_heads", "window_size"), path)
    else:
        _require(args, ("dim", "depth", "heads", "mlp_dim"), path)

    if args.task == "seg" and getattr(args, "max_iters", None) is None:
        _require(args, ("n_epochs",), path)
    if args.task == "det":
        _require(args, ("n_epochs",), path)


def _build_task_config(args):
    if args.task == "seg":
        from tasks.segmentation import SegmentationTask

        task = SegmentationTask.__new__(SegmentationTask)
        task.args = args
        task.run_name = task._build_run_name()
        task.use_wandb = False
        return task._build_mmseg_config()

    from tasks.detection import DetectionTask

    task = DetectionTask.__new__(DetectionTask)
    task.args = args
    task.run_name = task._build_run_name()
    task.use_wandb = False
    return task._build_mmdet_config()


def validate_configs() -> None:
    paths = sorted((ROOT / "configs").glob("seg_*.yaml")) + sorted((ROOT / "configs").glob("detection_*.yaml"))
    if not paths:
        raise AssertionError("No segmentation or detection configs found")

    counts = {"seg": 0, "det": 0}
    for path in paths:
        args = _load_args(path)
        _validate_common_fields(args, path)
        with redirect_stdout(StringIO()):
            cfg = _build_task_config(args)
        if args.task == "seg":
            assert cfg.model["type"] == "EncoderDecoder"
            assert cfg.data["train"]["type"] == "ADE20KDataset"
        else:
            assert cfg.model["type"] == "MaskRCNN"
            assert cfg.data["train"]["type"] == "CocoDataset"
        assert isinstance(cfg.data["workers_per_gpu"], int)
        counts[args.task] += 1

    print(f"Config validation passed: {counts['seg']} segmentation, {counts['det']} detection")


def validate_backbones() -> None:
    if not (_has_module("torch") and _has_module("timm") and _has_module("einops")):
        print("Backbone smoke skipped: install torch, timm, and einops to enable it")
        return

    import torch
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    cases = [
        ("ViTBackbone", ViTBackbone, {}),
        ("ViTCoPEBackbone(no-cls)", ViTCoPEBackbone, {"use_cls_token": False}),
        ("ViTCoPEBackbone(cls)", ViTCoPEBackbone, {"use_cls_token": True}),
        ("ViTSCoPEBackbone(no-cls)", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("ViTSCoPEBackbone(cls)", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]
    expected = [(1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1)]

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
                outputs = model(torch.randn(1, 3, 32, 32))
            shapes = [tuple(output.shape) for output in outputs]
            if shapes != expected:
                raise AssertionError(f"{name} ({style}) shapes {shapes}, expected {expected}")

    print("Backbone smoke passed: ViT, ViT-CoPE, and ViT-SCoPE")


def main() -> None:
    use_dummy_backbones = not (_has_module("torch") and _has_module("timm") and _has_module("einops"))
    _install_stubs(use_dummy_backbones=use_dummy_backbones)
    validate_configs()
    validate_backbones()


if __name__ == "__main__":
    main()
