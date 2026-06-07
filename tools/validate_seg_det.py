#!/usr/bin/env python3
"""Lightweight segmentation/detection validation for local development.

This script intentionally stubs the small OpenMMLab surface needed by the task
config builders. It validates the repository code paths that can run without
COCO/ADE20K data or mmcv-full C++ extensions, then runs real CPU forward smoke
tests for the custom ViT/CoPE/SCoPE dense-prediction backbones.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import types
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Config(dict):
    """Minimal mmcv.Config replacement used by the task config builders."""

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
            if force or key not in self.module_dict:
                self.module_dict[key] = cls
            return cls

        if module is not None:
            key = name or module.__name__
            if force or key not in self.module_dict:
                self.module_dict[key] = module
            return module
        return decorator


def install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    sys.modules["mmcv"] = mmcv

    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object
    sys.modules["mmcv.parallel"] = mmcv_parallel

    for root in ("mmdet", "mmseg"):
        sys.modules[root] = types.ModuleType(root)

        apis = types.ModuleType(f"{root}.apis")
        apis.set_random_seed = lambda *args, **kwargs: None
        apis.single_gpu_test = lambda *args, **kwargs: []
        apis.train_detector = lambda *args, **kwargs: None
        apis.train_segmentor = lambda *args, **kwargs: None
        sys.modules[f"{root}.apis"] = apis

        datasets = types.ModuleType(f"{root}.datasets")
        datasets.build_dataloader = lambda *args, **kwargs: None
        datasets.build_dataset = lambda *args, **kwargs: None
        sys.modules[f"{root}.datasets"] = datasets

        models = types.ModuleType(f"{root}.models")
        models.build_detector = lambda *args, **kwargs: None
        models.build_segmentor = lambda *args, **kwargs: None
        sys.modules[f"{root}.models"] = models

        builder = types.ModuleType(f"{root}.models.builder")
        builder.BACKBONES = Registry()
        sys.modules[f"{root}.models.builder"] = builder


OPTIONAL_NONE_DEFAULTS = (
    "backbone_size",
    "betas",
    "checkpoint_interval",
    "eval_interval",
    "final_eval",
    "img_scale",
    "layer_decay_rate",
    "log_interval",
    "min_pretrained_match_rate",
    "out_indices",
    "seg_aux_dim",
    "seg_aux_in_index",
    "seg_head_dim",
    "seg_neck_dim",
    "seg_neck_style",
    "seg_norm_type",
    "seed",
    "test_img_scale",
    "workers_per_gpu",
)


def load_args(config_path: Path) -> argparse.Namespace:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    args = argparse.Namespace(**data)
    args.cfg = str(config_path.relative_to(REPO_ROOT))
    args.run_tag = None

    # Match argparse behavior where optional CLI overrides may exist but remain
    # None, which previously masked task-level defaults.
    for name in OPTIONAL_NONE_DEFAULTS:
        if not hasattr(args, name):
            setattr(args, name, None)
    return args


def validate_segmentation_configs():
    from tasks.segmentation import SegmentationTask

    config_paths = sorted(REPO_ROOT.glob("configs/seg_*.yaml"))
    for config_path in config_paths:
        args = load_args(config_path)
        task = SegmentationTask.__new__(SegmentationTask)
        task.args = args
        task.run_name = config_path.stem
        cfg = task._build_mmseg_config()

        assert cfg.model["type"] == "EncoderDecoder", config_path
        assert cfg.data["train"]["type"] == "ADE20KDataset", config_path
        assert cfg.data["workers_per_gpu"] == 4, config_path
        if args.model == "swin":
            assert cfg.model["backbone"]["type"] == "SwinTransformer", config_path
        else:
            assert cfg.model["backbone"]["type"].startswith("ViT"), config_path
    print(f"Validated {len(config_paths)} segmentation configs")


def validate_detection_configs():
    from tasks.detection import DetectionTask

    config_paths = sorted(REPO_ROOT.glob("configs/detection_*.yaml"))
    for config_path in config_paths:
        args = load_args(config_path)
        task = DetectionTask.__new__(DetectionTask)
        task.args = args
        task.run_name = config_path.stem
        cfg = task._build_mmdet_config()

        assert cfg.model["type"] == "MaskRCNN", config_path
        assert cfg.data["train"]["type"] == "CocoDataset", config_path
        assert cfg.data["workers_per_gpu"] == 4, config_path
        if args.model == "swin":
            assert cfg.model["backbone"]["type"] == "SwinTransformer", config_path
        else:
            assert cfg.model["backbone"]["type"].startswith("ViT"), config_path
    print(f"Validated {len(config_paths)} detection configs")


def validate_backbone_forwards():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    cases = []
    for adapter_style in ("resize", "simple_fpn"):
        common = dict(
            image_size=32,
            patch_size=16,
            dim=32,
            depth=4,
            heads=2,
            mlp_dim=64,
            dim_head=16,
            out_indices=(0, 1, 2, 3),
            fpn_adapter_style=adapter_style,
        )
        cases.append((f"vit/{adapter_style}", ViTBackbone(**common)))
        cases.append((f"vitcope-nocls/{adapter_style}", ViTCoPEBackbone(use_cls_token=False, **common)))
        cases.append((f"vitcope-cls/{adapter_style}", ViTCoPEBackbone(use_cls_token=True, **common)))
        cases.append((f"vitscope-nocls/{adapter_style}", ViTSCoPEBackbone(use_cls_token=False, **common)))
        cases.append((f"vitscope-cls/{adapter_style}", ViTSCoPEBackbone(use_cls_token=True, **common)))

    expected_sizes = (8, 4, 2, 1)
    x = torch.randn(2, 3, 32, 32)
    for name, model in cases:
        model.eval()
        with torch.no_grad():
            outputs = model(x)
        assert len(outputs) == 4, name
        for output, size in zip(outputs, expected_sizes):
            assert tuple(output.shape) == (2, 32, size, size), (name, tuple(output.shape), size)
    print(f"Validated {len(cases)} custom backbone forward cases")


def main():
    os.chdir(REPO_ROOT)
    install_openmmlab_stubs()
    validate_segmentation_configs()
    validate_detection_configs()
    validate_backbone_forwards()
    print("Segmentation and detection validation passed")


if __name__ == "__main__":
    main()
