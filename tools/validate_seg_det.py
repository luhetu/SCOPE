#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke validation for segmentation/detection configs and custom backbones.

This script intentionally avoids constructing full MMSeg/MMDet trainers. Those
paths require mmcv-full compiled extensions, while config construction and the
custom ViT-family backbones can be validated with lightweight stubs.
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import sys
import types


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
                raise KeyError(f"{key} is already registered")
            self.module_dict[key] = cls
            return cls

        return _register(module) if module is not None else _register


def _noop(*args, **kwargs):
    return None


def install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object

    mmseg = types.ModuleType("mmseg")
    mmseg_apis = types.ModuleType("mmseg.apis")
    mmseg_apis.set_random_seed = _noop
    mmseg_apis.single_gpu_test = _noop
    mmseg_apis.train_segmentor = _noop
    mmseg_datasets = types.ModuleType("mmseg.datasets")
    mmseg_datasets.build_dataloader = _noop
    mmseg_datasets.build_dataset = _noop
    mmseg_models = types.ModuleType("mmseg.models")
    mmseg_models.build_segmentor = _noop
    mmseg_models_builder = types.ModuleType("mmseg.models.builder")
    mmseg_models_builder.BACKBONES = Registry()

    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = _noop
    mmdet_apis.train_detector = _noop
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = _noop
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = _noop
    mmdet_models_builder = types.ModuleType("mmdet.models.builder")
    mmdet_models_builder.BACKBONES = Registry()

    sys.modules.update({
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
    })


@contextlib.contextmanager
def _argv(argv):
    old_argv = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old_argv


def load_args(cfg_path):
    from utils.cfg import load_cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=str(cfg_path))
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--workers_per_gpu", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--time_profile", action="store_true")
    parser.add_argument("--time_profile_interval", type=int, default=1000)
    with _argv(["validate_seg_det"]):
        return load_cfg(parser)


def validate_segmentation_configs(paths):
    from tasks.segmentation import SegmentationTask

    for cfg_path in paths:
        args = load_args(cfg_path)
        task = SegmentationTask.__new__(SegmentationTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmseg_config()
        assert cfg.model["type"] == "EncoderDecoder", cfg_path
        assert cfg.data["workers_per_gpu"] == 4, cfg_path
        assert cfg.data["train"]["type"] == "ADE20KDataset", cfg_path
    print(f"validated {len(paths)} segmentation configs")


def validate_detection_configs(paths):
    from tasks.detection import DetectionTask

    for cfg_path in paths:
        args = load_args(cfg_path)
        task = DetectionTask.__new__(DetectionTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmdet_config()
        assert cfg.model["type"] == "MaskRCNN", cfg_path
        assert cfg.data["workers_per_gpu"] == 4, cfg_path
        assert cfg.data["train"]["type"] == "CocoDataset", cfg_path
    print(f"validated {len(paths)} detection configs")


def validate_backbones():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    expected_shapes = [(1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1)]
    common = dict(
        image_size=32,
        patch_size=16,
        dim=32,
        depth=4,
        heads=4,
        mlp_dim=64,
        dim_head=8,
        out_indices=(0, 1, 2, 3),
    )
    cases = []
    for style in ("resize", "simple_fpn"):
        cases.append((ViTBackbone, dict(fpn_adapter_style=style)))
        for use_cls_token in (False, True):
            cases.append((ViTCoPEBackbone, dict(fpn_adapter_style=style, use_cls_token=use_cls_token)))
            cases.append((ViTSCoPEBackbone, dict(fpn_adapter_style=style, use_cls_token=use_cls_token)))

    x = torch.randn(1, 3, 32, 32)
    for cls, extra in cases:
        model = cls(**common, **extra).eval()
        with torch.no_grad():
            outputs = model(x)
        shapes = [tuple(out.shape) for out in outputs]
        assert shapes == expected_shapes, f"{cls.__name__} {extra}: {shapes}"
    print(f"validated {len(cases)} custom backbone forward cases")


def main():
    os.chdir(ROOT)
    install_openmmlab_stubs()
    seg_paths = sorted(ROOT.glob("configs/seg_*.yaml"))
    det_paths = sorted(ROOT.glob("configs/detection_*.yaml"))
    validate_segmentation_configs(seg_paths)
    validate_detection_configs(det_paths)
    validate_backbones()
    print("segmentation/detection smoke validation passed")


if __name__ == "__main__":
    main()
