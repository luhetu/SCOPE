#!/usr/bin/env python3
"""Lightweight validation for segmentation and detection wiring.

This script intentionally avoids dataset loading and compiled MMCV ops. It
checks the Python/config paths that commonly break before a real training job:
all seg/det YAML files must build their MMSeg/MMDet configs, and the custom
ViT-family dense-prediction backbones must produce four feature maps.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import types
from contextlib import contextmanager


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class Config(dict):
    """Small attribute-access dict compatible with the task config builders."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = self._convert(value)

    def __setitem__(self, name, value):
        super().__setitem__(name, self._convert(value))

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    @classmethod
    def _convert(cls, value):
        if isinstance(value, dict) and not isinstance(value, Config):
            return Config(value)
        if isinstance(value, list):
            return [cls._convert(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._convert(item) for item in value)
        return value


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

        if module is not None:
            return _register(module)
        return _register


def _install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object

    mmseg_backbones = Registry()
    mmseg_builder = types.ModuleType("mmseg.models.builder")
    mmseg_builder.BACKBONES = mmseg_backbones
    mmseg_models = types.ModuleType("mmseg.models")
    mmseg_models.build_segmentor = lambda *args, **kwargs: None
    mmseg_models.builder = mmseg_builder
    mmseg_apis = types.ModuleType("mmseg.apis")
    mmseg_apis.set_random_seed = lambda *args, **kwargs: None
    mmseg_apis.single_gpu_test = lambda *args, **kwargs: []
    mmseg_apis.train_segmentor = lambda *args, **kwargs: None
    mmseg_datasets = types.ModuleType("mmseg.datasets")
    mmseg_datasets.build_dataloader = lambda *args, **kwargs: None
    mmseg_datasets.build_dataset = lambda *args, **kwargs: None
    mmseg = types.ModuleType("mmseg")
    mmseg.apis = mmseg_apis
    mmseg.datasets = mmseg_datasets
    mmseg.models = mmseg_models

    mmdet_backbones = Registry()
    mmdet_builder = types.ModuleType("mmdet.models.builder")
    mmdet_builder.BACKBONES = mmdet_backbones
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = lambda *args, **kwargs: None
    mmdet_models.builder = mmdet_builder
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = lambda *args, **kwargs: None
    mmdet_apis.train_detector = lambda *args, **kwargs: None
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = lambda *args, **kwargs: None
    mmdet = types.ModuleType("mmdet")
    mmdet.apis = mmdet_apis
    mmdet.datasets = mmdet_datasets
    mmdet.models = mmdet_models

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


@contextmanager
def _argv(argv):
    old = sys.argv[:]
    sys.argv = argv[:]
    try:
        yield
    finally:
        sys.argv = old


def _load_args(cfg_path):
    from utils.cfg import load_cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--workers_per_gpu", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--time_profile", action="store_true")
    parser.add_argument("--time_profile_interval", type=int, default=1000)
    with _argv(["validate_seg_det.py", "--cfg", cfg_path]):
        return load_cfg(parser)


def _task_shell(task_cls, args):
    task = object.__new__(task_cls)
    task.args = args
    task.device = "cpu"
    task.run_name = task._build_run_name()
    return task


def validate_configs():
    _install_openmmlab_stubs()

    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    seg_paths = sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "seg_*.yaml")))
    det_paths = sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "detection_*.yaml")))
    if not seg_paths or not det_paths:
        raise AssertionError("Expected both segmentation and detection configs")

    for path in seg_paths:
        args = _load_args(path)
        task = _task_shell(SegmentationTask, args)
        cfg = task._build_mmseg_config()
        assert cfg.model.type == "EncoderDecoder"
        assert cfg.data.workers_per_gpu == 4
        assert cfg.checkpoint_config.interval > 0

    for path in det_paths:
        args = _load_args(path)
        task = _task_shell(DetectionTask, args)
        cfg = task._build_mmdet_config()
        assert cfg.model.type == "MaskRCNN"
        assert cfg.data.workers_per_gpu == 4
        assert cfg.log_config.interval > 0

    print(f"Config validation passed: {len(seg_paths)} segmentation, {len(det_paths)} detection")


def _assert_feature_shapes(outputs, expected_shapes, label):
    got = [tuple(item.shape) for item in outputs]
    if got != expected_shapes:
        raise AssertionError(f"{label}: expected {expected_shapes}, got {got}")


def validate_backbones():
    _install_openmmlab_stubs()

    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

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
    expected = [
        (1, 32, 8, 8),
        (1, 32, 4, 4),
        (1, 32, 2, 2),
        (1, 32, 1, 1),
    ]
    cases = []
    for style in ("resize", "simple_fpn"):
        cases.append((f"vit/{style}", ViTBackbone(fpn_adapter_style=style, **common)))
        for use_cls in (False, True):
            cases.append((f"vitcope/{style}/cls={use_cls}", ViTCoPEBackbone(use_cls_token=use_cls, fpn_adapter_style=style, **common)))
            cases.append((f"vitscope/{style}/cls={use_cls}", ViTSCoPEBackbone(use_cls_token=use_cls, fpn_adapter_style=style, **common)))

    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        for label, model in cases:
            model.eval()
            outputs = model(image)
            _assert_feature_shapes(outputs, expected, label)

    print(f"Backbone validation passed: {len(cases)} CPU forward cases")


def main():
    validate_configs()
    validate_backbones()


if __name__ == "__main__":
    main()
