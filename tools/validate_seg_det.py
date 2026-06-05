#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-test segmentation/detection config builders and custom backbones.

This validation intentionally stubs OpenMMLab entrypoints so it can run in a
lightweight Python environment without mmcv-full, COCO, or ADE20K installed.
"""

import argparse
import contextlib
import io
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Config(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class Registry:
    def __init__(self):
        self.module_dict = {}

    def register_module(self, name=None, module=None, **kwargs):
        def decorator(cls):
            self.module_dict[name or cls.__name__] = cls
            return cls

        return decorator(module) if module is not None else decorator


def install_openmmlab_stubs():
    backbones = Registry()

    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    sys.modules["mmcv"] = mmcv

    parallel = types.ModuleType("mmcv.parallel")
    parallel.MMDataParallel = object
    sys.modules["mmcv.parallel"] = parallel

    for root in ("mmdet", "mmseg"):
        sys.modules[root] = types.ModuleType(root)

        apis = types.ModuleType(f"{root}.apis")
        apis.set_random_seed = lambda *args, **kwargs: None
        if root == "mmdet":
            apis.train_detector = lambda *args, **kwargs: None
        else:
            apis.single_gpu_test = lambda *args, **kwargs: []
            apis.train_segmentor = lambda *args, **kwargs: None
        sys.modules[f"{root}.apis"] = apis

        datasets = types.ModuleType(f"{root}.datasets")
        datasets.build_dataset = lambda *args, **kwargs: None
        if root == "mmseg":
            datasets.build_dataloader = lambda *args, **kwargs: None
        sys.modules[f"{root}.datasets"] = datasets

        models = types.ModuleType(f"{root}.models")
        models.build_detector = lambda *args, **kwargs: None
        models.build_segmentor = lambda *args, **kwargs: None
        sys.modules[f"{root}.models"] = models

        builder = types.ModuleType(f"{root}.models.builder")
        builder.BACKBONES = backbones
        sys.modules[f"{root}.models.builder"] = builder


def build_parser(cfg_path):
    parser = argparse.ArgumentParser(description="seg/det validation parser")
    parser.add_argument("--cfg", type=str, default=str(cfg_path))
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--workers_per_gpu", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--time_profile", action="store_true")
    parser.add_argument("--time_profile_interval", type=int, default=1000)

    # Optional overrides that commonly exist as argparse defaults of None.
    parser.add_argument("--checkpoint_interval", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--seg_head_dim", type=int, default=None)
    parser.add_argument("--seg_aux_dim", type=int, default=None)
    parser.add_argument("--seg_neck_dim", type=int, default=None)
    parser.add_argument("--seg_aux_in_index", type=int, default=None)
    parser.add_argument("--dim_head", type=int, default=None)
    parser.add_argument("--drop_path_rate", type=float, default=None)
    parser.add_argument("--layer_decay_rate", type=float, default=None)
    parser.add_argument("--min_pretrained_match_rate", type=float, default=None)
    return parser


def load_args(cfg_path):
    from utils.cfg import load_cfg

    old_argv = sys.argv[:]
    sys.argv = ["validate_seg_det.py", "--cfg", str(cfg_path)]
    try:
        return load_cfg(build_parser(cfg_path))
    finally:
        sys.argv = old_argv


def validate_config_builders():
    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    results = []
    config_paths = sorted((REPO_ROOT / "configs").glob("seg_*.yaml"))
    config_paths += sorted((REPO_ROOT / "configs").glob("detection_*.yaml"))
    if not config_paths:
        raise RuntimeError("No segmentation or detection configs found.")

    for cfg_path in config_paths:
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                args = load_args(cfg_path)
                task_cls = SegmentationTask if args.task == "seg" else DetectionTask
                task = task_cls.__new__(task_cls)
                task.args = args
                task.run_name = task._build_run_name()
                cfg = task._build_mmseg_config() if args.task == "seg" else task._build_mmdet_config()
            backbone_type = cfg.model["backbone"]["type"]
            results.append((args.task, cfg_path.name, args.model, backbone_type))
        except Exception:
            print(captured.getvalue(), end="")
            raise

    seg_count = sum(1 for task, *_ in results if task == "seg")
    det_count = sum(1 for task, *_ in results if task == "det")
    print(f"Config builders: OK ({seg_count} segmentation, {det_count} detection)")
    for task, name, model, backbone_type in results:
        print(f"  {task:3s} {name:32s} model={model:9s} backbone={backbone_type}")


def validate_backbone_forward():
    import torch
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    cases = [
        ("vit", ViTBackbone, {}),
        ("vitcope_no_cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("vitcope_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("vitscope_no_cls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("vitscope_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]
    styles = ("resize", "simple_fpn")
    expected_shapes = [
        (1, 32, 8, 8),
        (1, 32, 4, 4),
        (1, 32, 2, 2),
        (1, 32, 1, 1),
    ]

    torch.manual_seed(0)
    x = torch.randn(1, 3, 32, 32)
    checked = 0
    for style in styles:
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
                outputs = model(x)
            shapes = [tuple(output.shape) for output in outputs]
            if shapes != expected_shapes:
                raise AssertionError(f"{name}/{style} shapes {shapes} != {expected_shapes}")
            checked += 1
            print(f"  backbone {name:15s} style={style:10s} shapes={shapes}")

    print(f"Backbone forward: OK ({checked} CPU cases)")


def main():
    install_openmmlab_stubs()
    validate_config_builders()
    validate_backbone_forward()
    print("Segmentation/detection validation passed.")


if __name__ == "__main__":
    main()
