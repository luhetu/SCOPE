#!/usr/bin/env python3
"""Lightweight smoke tests for segmentation and detection configuration code.

The full seg/det training path depends on mmcv-full CUDA extensions and real
ADE20K/COCO datasets. This script validates the repo-owned code paths that can
run in a minimal CPU environment: task config builders and custom ViT backbones.
"""

from __future__ import annotations

import glob
import os
import sys
import types
from argparse import Namespace

import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class FakeConfig(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class FakeRegistry:
    def __init__(self):
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        def _register(cls):
            module_name = name or cls.__name__
            if not force and module_name in self.module_dict:
                return cls
            self.module_dict[module_name] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


def _make_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _noop(*args, **kwargs):
    return None


def _install_openmmlab_stubs():
    mmseg_backbones = FakeRegistry()
    mmdet_backbones = FakeRegistry()

    mmcv = _make_module("mmcv", Config=FakeConfig)
    mmcv_parallel = _make_module("mmcv.parallel", MMDataParallel=lambda model, *args, **kwargs: model)

    mmseg_models_builder = _make_module("mmseg.models.builder", BACKBONES=mmseg_backbones)
    mmseg_models = _make_module("mmseg.models", build_segmentor=lambda *args, **kwargs: None)
    mmseg_apis = _make_module(
        "mmseg.apis",
        set_random_seed=_noop,
        single_gpu_test=lambda *args, **kwargs: [],
        train_segmentor=_noop,
    )
    mmseg_datasets = _make_module(
        "mmseg.datasets",
        build_dataloader=lambda *args, **kwargs: None,
        build_dataset=lambda *args, **kwargs: None,
    )
    mmseg = _make_module("mmseg", apis=mmseg_apis, datasets=mmseg_datasets, models=mmseg_models)

    mmdet_models_builder = _make_module("mmdet.models.builder", BACKBONES=mmdet_backbones)
    mmdet_models = _make_module("mmdet.models", build_detector=lambda *args, **kwargs: None)
    mmdet_apis = _make_module("mmdet.apis", set_random_seed=_noop, train_detector=_noop)
    mmdet_datasets = _make_module("mmdet.datasets", build_dataset=lambda *args, **kwargs: None)
    mmdet = _make_module("mmdet", apis=mmdet_apis, datasets=mmdet_datasets, models=mmdet_models)

    sys.modules.update(
        {
            "mmcv": mmcv,
            "mmcv.parallel": mmcv_parallel,
            "mmseg": mmseg,
            "mmseg.apis": mmseg_apis,
            "mmseg.datasets": mmseg_datasets,
            "mmseg.models": mmseg_models,
            "mmseg.models.builder": mmseg_models_builder,
            "mmdet": mmdet,
            "mmdet.apis": mmdet_apis,
            "mmdet.datasets": mmdet_datasets,
            "mmdet.models": mmdet_models,
            "mmdet.models.builder": mmdet_models_builder,
        }
    )


def _load_args(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    values.setdefault("cfg", cfg_path)
    values.setdefault("workers_per_gpu", None)
    values.setdefault("seed", None)
    values.setdefault("run_tag", None)
    values.setdefault("resume", "")
    values.setdefault("final_eval", True)
    values.setdefault("seg_neck_style", "xcit_fpn")
    values.setdefault("det_neck_type", "fpn")
    return Namespace(**values)


def _validate_config_builders():
    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    cfg_paths = sorted(glob.glob(os.path.join(ROOT, "configs", "seg_*.yaml")))
    cfg_paths.extend(sorted(glob.glob(os.path.join(ROOT, "configs", "detection_*.yaml"))))
    if not cfg_paths:
        raise RuntimeError("No segmentation or detection configs found")

    for cfg_path in cfg_paths:
        args = _load_args(cfg_path)
        if args.task == "seg":
            task = SegmentationTask.__new__(SegmentationTask)
            task.args = args
            task.run_name = task._build_run_name()
            cfg = task._build_mmseg_config()
        elif args.task == "det":
            task = DetectionTask.__new__(DetectionTask)
            task.args = args
            task.run_name = task._build_run_name()
            cfg = task._build_mmdet_config()
        else:
            raise AssertionError(f"Unexpected task in {cfg_path}: {args.task}")

        if cfg.data.workers_per_gpu != 4:
            raise AssertionError(f"{cfg_path}: expected workers_per_gpu default 4")
        if not cfg.model.get("backbone"):
            raise AssertionError(f"{cfg_path}: missing model.backbone config")

    print(f"Validated {len(cfg_paths)} seg/det config builders")


def _expected_feature_shapes(dim):
    return [(1, dim, 8, 8), (1, dim, 4, 4), (1, dim, 2, 2), (1, dim, 1, 1)]


def _validate_backbone_forwards():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    torch.set_num_threads(1)
    model_specs = [
        (ViTBackbone, {}),
        (ViTCoPEBackbone, {"use_cls_token": False}),
        (ViTCoPEBackbone, {"use_cls_token": True}),
        (ViTSCoPEBackbone, {"use_cls_token": False}),
        (ViTSCoPEBackbone, {"use_cls_token": True}),
    ]
    styles = ["resize", "simple_fpn"]
    dim = 32

    for model_cls, extra_kwargs in model_specs:
        for style in styles:
            model = model_cls(
                image_size=32,
                patch_size=16,
                dim=dim,
                depth=4,
                heads=4,
                mlp_dim=64,
                dim_head=8,
                out_indices=(0, 1, 2, 3),
                fpn_adapter_style=style,
                **extra_kwargs,
            ).eval()
            with torch.no_grad():
                outputs = model(torch.randn(1, 3, 32, 32))
            shapes = [tuple(output.shape) for output in outputs]
            expected = _expected_feature_shapes(dim)
            if shapes != expected:
                raise AssertionError(f"{model_cls.__name__}({extra_kwargs}, {style}) shapes {shapes} != {expected}")

    print("Validated custom backbone forward smoke tests")


def main():
    _install_openmmlab_stubs()
    _validate_config_builders()
    _validate_backbone_forwards()
    print("Segmentation and detection lightweight validation passed")


if __name__ == "__main__":
    main()
