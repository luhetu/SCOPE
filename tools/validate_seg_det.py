#!/usr/bin/env python3
"""Lightweight validation for segmentation and detection glue code.

This script avoids requiring a full mmcv-full/OpenMMLab runtime by installing
small import stubs, then validates repo-local config builders and custom
backbone forward passes on CPU.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = self._wrap(value)

    def __setitem__(self, key, value):
        super().__setitem__(key, self._wrap(value))

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            wrapped = AttrDict()
            for key, item in value.items():
                wrapped[key] = item
            return wrapped
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._wrap(item) for item in value)
        return value


class Config(AttrDict):
    pass


class Registry:
    def __init__(self):
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        def _register(cls):
            module_name = name or cls.__name__
            if not force and module_name in self.module_dict:
                raise KeyError(f"{module_name} is already registered")
            self.module_dict[module_name] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


class DummyDataset:
    CLASSES = tuple(str(idx) for idx in range(150))
    PALETTE = None

    def evaluate(self, *args, **kwargs):
        return {}


class DummyModel:
    CLASSES = ()

    def state_dict(self):
        return {}

    def load_state_dict(self, *args, **kwargs):
        return None


def _make_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def install_openmmlab_stubs():
    mmdet_backbones = Registry()
    mmseg_backbones = Registry()

    mmcv = _make_module("mmcv", Config=Config)
    mmcv_parallel = _make_module("mmcv.parallel", MMDataParallel=lambda model, device_ids=None: model)

    mmdet_builder = _make_module("mmdet.models.builder", BACKBONES=mmdet_backbones)
    mmdet_models = _make_module("mmdet.models", build_detector=lambda *args, **kwargs: DummyModel())
    mmdet_models.builder = mmdet_builder
    mmdet_apis = _make_module(
        "mmdet.apis",
        set_random_seed=lambda *args, **kwargs: None,
        train_detector=lambda *args, **kwargs: None,
    )
    mmdet_datasets = _make_module("mmdet.datasets", build_dataset=lambda *args, **kwargs: DummyDataset())
    mmdet = _make_module("mmdet", apis=mmdet_apis, datasets=mmdet_datasets, models=mmdet_models)

    mmseg_builder = _make_module("mmseg.models.builder", BACKBONES=mmseg_backbones)
    mmseg_models = _make_module("mmseg.models", build_segmentor=lambda *args, **kwargs: DummyModel())
    mmseg_models.builder = mmseg_builder
    mmseg_apis = _make_module(
        "mmseg.apis",
        set_random_seed=lambda *args, **kwargs: None,
        single_gpu_test=lambda *args, **kwargs: [],
        train_segmentor=lambda *args, **kwargs: None,
    )
    mmseg_datasets = _make_module(
        "mmseg.datasets",
        build_dataloader=lambda *args, **kwargs: [],
        build_dataset=lambda *args, **kwargs: DummyDataset(),
    )
    mmseg = _make_module("mmseg", apis=mmseg_apis, datasets=mmseg_datasets, models=mmseg_models)

    modules = {
        "mmcv": mmcv,
        "mmcv.parallel": mmcv_parallel,
        "mmdet": mmdet,
        "mmdet.apis": mmdet_apis,
        "mmdet.datasets": mmdet_datasets,
        "mmdet.models": mmdet_models,
        "mmdet.models.builder": mmdet_builder,
        "mmseg": mmseg,
        "mmseg.apis": mmseg_apis,
        "mmseg.datasets": mmseg_datasets,
        "mmseg.models": mmseg_models,
        "mmseg.models.builder": mmseg_builder,
    }
    sys.modules.update(modules)


def base_parser():
    parser = argparse.ArgumentParser(description="Validate seg/det configs")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--workers_per_gpu", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--time_profile", action="store_true")
    parser.add_argument("--time_profile_interval", type=int, default=1000)
    return parser


def load_args_for_config(cfg_path):
    from utils.cfg import load_cfg

    old_argv = sys.argv[:]
    sys.argv = [old_argv[0], "--cfg", str(cfg_path)]
    try:
        return load_cfg(base_parser())
    finally:
        sys.argv = old_argv


def apply_train_schedule_defaults(args):
    if args.task == "seg":
        iters_per_epoch = 20210 // args.bs
        if getattr(args, "max_iters", None) is not None:
            args.max_iters = int(args.max_iters)
        else:
            args.max_iters = int(args.n_epochs * iters_per_epoch)
        if getattr(args, "warmup_iters", None) is not None:
            args.warmup_iters = int(args.warmup_iters)
        else:
            warmup_epochs = getattr(args, "warmup_epochs", None) or 0
            args.warmup_iters = int(warmup_epochs * iters_per_epoch) if warmup_epochs > 0 else 1500
        args.iters_per_epoch = iters_per_epoch
    elif args.task == "det":
        iters_per_epoch = 118287 // args.bs
        if getattr(args, "warmup_iters", None) is not None:
            args.warmup_iters = int(args.warmup_iters)
        else:
            warmup_epochs = getattr(args, "warmup_epochs", None) or 0
            args.warmup_iters = int(warmup_epochs * iters_per_epoch) if warmup_epochs > 0 else 500


def assert_common_train_fields(args, cfg_path):
    for field in ("task", "model", "data_dir", "bs", "lr", "n_epochs"):
        if getattr(args, field, None) is None:
            raise AssertionError(f"{cfg_path} missing required field: {field}")
    if getattr(args, "model", None) in {"vit", "vitcope", "vitscope", "swin"}:
        for field in ("size", "patch"):
            if getattr(args, field, None) is None:
                raise AssertionError(f"{cfg_path} missing train.py logging field: {field}")


def validate_segmentation_configs():
    from tasks.segmentation import SegmentationTask

    config_paths = sorted(glob.glob(str(PROJECT_ROOT / "configs" / "seg_*.yaml")))
    for cfg_path in config_paths:
        args = load_args_for_config(cfg_path)
        apply_train_schedule_defaults(args)
        assert_common_train_fields(args, cfg_path)
        task = object.__new__(SegmentationTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmseg_config()
        assert cfg.model.type == "EncoderDecoder"
        assert cfg.data.train.type == "ADE20KDataset"
        assert cfg.runner.type == "IterBasedRunner"
    print(f"Validated {len(config_paths)} segmentation configs")


def validate_detection_configs():
    from tasks.detection import DetectionTask

    config_paths = sorted(glob.glob(str(PROJECT_ROOT / "configs" / "detection_*.yaml")))
    for cfg_path in config_paths:
        args = load_args_for_config(cfg_path)
        apply_train_schedule_defaults(args)
        assert_common_train_fields(args, cfg_path)
        task = object.__new__(DetectionTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmdet_config()
        assert cfg.model.type == "MaskRCNN"
        assert cfg.data.train.type == "CocoDataset"
        assert cfg.runner.type == "EpochBasedRunner"
    print(f"Validated {len(config_paths)} detection configs")


def validate_backbone_forward():
    import torch
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    torch.set_num_threads(1)
    image = torch.randn(1, 3, 32, 32)
    expected_shapes = [(1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1)]
    cases = [
        ("vit", ViTBackbone, {}),
        ("vitcope_no_cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("vitcope_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("vitscope_no_cls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("vitscope_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]

    for style in ("resize", "simple_fpn"):
        for name, cls, extra_kwargs in cases:
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
            )
            model.eval()
            with torch.no_grad():
                outputs = model(image)
            shapes = [tuple(output.shape) for output in outputs]
            if shapes != expected_shapes:
                raise AssertionError(f"{name}/{style} shapes {shapes}, expected {expected_shapes}")
    print("Validated custom backbone CPU forward smoke tests")


def main():
    os.chdir(PROJECT_ROOT)
    install_openmmlab_stubs()
    validate_segmentation_configs()
    validate_detection_configs()
    validate_backbone_forward()
    print("Segmentation and detection validation passed")


if __name__ == "__main__":
    main()
