#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight validation for segmentation/detection config builders.

The full train entrypoints require mmcv-full CUDA extensions. This script keeps
the check CPU-only by stubbing the MMDetection/MMSegmentation builder APIs while
still importing and running the custom SCOPE backbones with real PyTorch.
"""

import argparse
import sys
import types
from pathlib import Path

import torch
import yaml


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
        def decorator(cls):
            key = name or cls.__name__
            if force or key not in self.module_dict:
                self.module_dict[key] = cls
            return cls

        if module is not None:
            return decorator(module)
        return decorator


def _dummy_model():
    return types.SimpleNamespace(
        CLASSES=(),
        state_dict=lambda: {},
        load_state_dict=lambda *args, **kwargs: None,
    )


def install_mm_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.parallel"] = mmcv_parallel

    for root, train_name, build_name in (
        ("mmseg", "train_segmentor", "build_segmentor"),
        ("mmdet", "train_detector", "build_detector"),
    ):
        pkg = types.ModuleType(root)
        pkg.__path__ = []

        apis = types.ModuleType(f"{root}.apis")
        apis.set_random_seed = lambda *args, **kwargs: None
        apis.single_gpu_test = lambda *args, **kwargs: []
        setattr(apis, train_name, lambda *args, **kwargs: None)

        datasets = types.ModuleType(f"{root}.datasets")
        datasets.build_dataloader = lambda *args, **kwargs: None
        datasets.build_dataset = lambda *args, **kwargs: types.SimpleNamespace(CLASSES=())

        models = types.ModuleType(f"{root}.models")
        setattr(models, build_name, lambda *args, **kwargs: _dummy_model())

        builder = types.ModuleType(f"{root}.models.builder")
        builder.BACKBONES = Registry()

        sys.modules[root] = pkg
        sys.modules[f"{root}.apis"] = apis
        sys.modules[f"{root}.datasets"] = datasets
        sys.modules[f"{root}.models"] = models
        sys.modules[f"{root}.models.builder"] = builder


def _load_args(path):
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    defaults = {
        "cfg": str(path),
        "resume": "",
        "workers_per_gpu": None,
        "checkpoint_interval": None,
        "eval_interval": None,
        "log_interval": None,
        "seg_head_dim": None,
        "seg_aux_dim": None,
        "seg_aux_in_index": None,
        "seg_neck_dim": None,
        "seg_norm_type": None,
    }
    defaults.update(data)
    return argparse.Namespace(**defaults)


def validate_config_builders():
    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    checks = []
    for path in sorted((REPO_ROOT / "configs").glob("seg*.yaml")):
        args = _load_args(path)
        task = object.__new__(SegmentationTask)
        task.args = args
        task.run_name = path.stem
        cfg = task._build_mmseg_config()
        assert cfg.model["type"] == "EncoderDecoder"
        assert isinstance(cfg.data["workers_per_gpu"], int)
        checks.append(("seg", path.name))

    for path in sorted((REPO_ROOT / "configs").glob("detection*.yaml")):
        args = _load_args(path)
        task = object.__new__(DetectionTask)
        task.args = args
        task.run_name = path.stem
        cfg = task._build_mmdet_config()
        assert cfg.model["type"] == "MaskRCNN"
        assert isinstance(cfg.data["workers_per_gpu"], int)
        checks.append(("det", path.name))

    print(f"Validated {sum(kind == 'seg' for kind, _ in checks)} segmentation configs")
    print(f"Validated {sum(kind == 'det' for kind, _ in checks)} detection configs")


def _expected_feature_shapes(dim):
    return [
        (1, dim, 8, 8),
        (1, dim, 4, 4),
        (1, dim, 2, 2),
        (1, dim, 1, 1),
    ]


def validate_backbone_forwards():
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    dim = 32
    common = dict(
        image_size=32,
        patch_size=16,
        dim=dim,
        depth=4,
        heads=4,
        mlp_dim=64,
        dim_head=8,
        out_indices=(0, 1, 2, 3),
    )
    cases = [
        ("vit", ViTBackbone, {}),
        ("vitcope_nocls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("vitcope_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("vitscope_nocls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("vitscope_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]

    image = torch.randn(1, 3, 32, 32)
    expected = _expected_feature_shapes(dim)
    passed = 0
    for style in ("resize", "simple_fpn"):
        for name, cls, extra in cases:
            model = cls(**common, fpn_adapter_style=style, **extra).eval()
            with torch.no_grad():
                outputs = model(image)
            shapes = [tuple(out.shape) for out in outputs]
            if shapes != expected:
                raise AssertionError(f"{name}/{style} shapes {shapes} != {expected}")
            passed += 1

    print(f"Validated {passed} custom backbone forward cases")


def main():
    install_mm_stubs()
    validate_config_builders()
    validate_backbone_forwards()
    print("Segmentation/detection validation passed")


if __name__ == "__main__":
    main()
