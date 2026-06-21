#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight validation for segmentation and detection task wiring.

This script intentionally stubs MMDetection/MMSegmentation entry points so it can
run in a plain CPU environment without mmcv-full native extensions. It verifies
that task configs are buildable and that the custom dense-prediction backbones
produce four feature maps with the expected FPN shapes.
"""

import argparse
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
        def register(cls):
            key = name or cls.__name__
            if key in self.module_dict and not force:
                raise KeyError(f"{key} is already registered")
            self.module_dict[key] = cls
            return cls

        return register(module) if module is not None else register


class MMDataParallel:
    def __init__(self, module, device_ids=None):
        self.module = module
        self.device_ids = device_ids


def _noop(*args, **kwargs):
    return None


def install_framework_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.__path__ = []
    mmcv.Config = Config
    sys.modules["mmcv"] = mmcv

    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = MMDataParallel
    sys.modules["mmcv.parallel"] = mmcv_parallel

    for root in ("mmseg", "mmdet"):
        package = types.ModuleType(root)
        package.__path__ = []
        sys.modules[root] = package

        apis = types.ModuleType(f"{root}.apis")
        apis.set_random_seed = _noop
        apis.single_gpu_test = lambda *args, **kwargs: []
        apis.train_segmentor = _noop
        apis.train_detector = _noop
        sys.modules[f"{root}.apis"] = apis

        datasets = types.ModuleType(f"{root}.datasets")
        datasets.build_dataset = _noop
        datasets.build_dataloader = _noop
        sys.modules[f"{root}.datasets"] = datasets

        models = types.ModuleType(f"{root}.models")
        models.build_segmentor = _noop
        models.build_detector = _noop
        sys.modules[f"{root}.models"] = models

        builder = types.ModuleType(f"{root}.models.builder")
        builder.BACKBONES = Registry()
        sys.modules[f"{root}.models.builder"] = builder


def load_train_args(cfg_path):
    from utils.cfg import load_cfg

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cfg", default=str(cfg_path))
    parser.add_argument("--resume", default="")
    parser.add_argument("--workers_per_gpu", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--time_profile", action="store_true")
    parser.add_argument("--time_profile_interval", type=int, default=1000)

    old_argv = sys.argv
    sys.argv = [old_argv[0]]
    try:
        return load_cfg(parser)
    finally:
        sys.argv = old_argv


def validate_task_configs(config_paths):
    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    failures = []
    for cfg_path in config_paths:
        try:
            args = load_train_args(cfg_path)
            task_cls = SegmentationTask if args.task == "seg" else DetectionTask
            task = task_cls.__new__(task_cls)
            task.args = args
            task.run_name = task._build_run_name()
            if args.task == "seg":
                task._build_mmseg_config()
            elif args.task == "det":
                task._build_mmdet_config()
            else:
                raise ValueError(f"Unexpected task in {cfg_path}: {args.task}")
            print(f"[OK] config: {cfg_path}")
        except Exception as exc:
            failures.append((cfg_path, exc))
            print(f"[FAIL] config: {cfg_path}: {type(exc).__name__}: {exc}")
    return failures


def validate_backbone_forwards():
    import torch
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    torch.set_num_threads(1)
    expected_shapes = [(1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1)]
    common = dict(
        image_size=32,
        patch_size=16,
        dim=32,
        depth=4,
        heads=2,
        mlp_dim=64,
        dim_head=16,
        out_indices=(0, 1, 2, 3),
    )
    cases = [
        ("vit", ViTBackbone, {}),
        ("vitcope_no_cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("vitcope_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("vitscope_no_cls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("vitscope_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]

    failures = []
    x = torch.randn(1, 3, 32, 32)
    for style in ("resize", "simple_fpn"):
        for name, cls, extra in cases:
            try:
                model = cls(**common, fpn_adapter_style=style, **extra).eval()
                with torch.no_grad():
                    outputs = model(x)
                shapes = [tuple(out.shape) for out in outputs]
                if shapes != expected_shapes:
                    raise AssertionError(f"expected {expected_shapes}, got {shapes}")
                print(f"[OK] backbone: {name} ({style}) -> {shapes}")
            except Exception as exc:
                failures.append((f"{name} ({style})", exc))
                print(f"[FAIL] backbone: {name} ({style}): {type(exc).__name__}: {exc}")
    return failures


def default_config_paths():
    configs_dir = REPO_ROOT / "configs"
    return sorted(configs_dir.glob("seg_*.yaml")) + sorted(configs_dir.glob("detection_*.yaml"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", type=Path, help="Specific config files to validate")
    args = parser.parse_args()

    install_framework_stubs()
    config_paths = args.configs or default_config_paths()
    config_failures = validate_task_configs(config_paths)
    backbone_failures = validate_backbone_forwards()

    failures = config_failures + backbone_failures
    if failures:
        print(f"\nValidation failed with {len(failures)} failure(s).")
        return 1
    print("\nSegmentation/detection validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
