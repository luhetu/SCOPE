#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-test segmentation/detection config builders and custom backbones.

This validation intentionally stubs the small OpenMMLab API surface needed by
the task config builders. It does not replace full COCO/ADE20K training, but it
does catch broken YAML/task wiring and verifies the dense-prediction ViT
backbones with real CPU tensor forwards.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ConfigDict(dict):
    @staticmethod
    def _convert(value):
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            return ConfigDict({k: ConfigDict._convert(v) for k, v in value.items()})
        if isinstance(value, list):
            return [ConfigDict._convert(v) for v in value]
        if isinstance(value, tuple):
            return tuple(ConfigDict._convert(v) for v in value)
        return value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = self._convert(value)

    def __setitem__(self, name, value):
        super().__setitem__(name, self._convert(value))


class Config(ConfigDict):
    pass


class Registry:
    def __init__(self):
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        def decorator(cls):
            key = name or cls.__name__
            if not force and key in self.module_dict:
                raise KeyError(f"{key} is already registered")
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
        cuda=lambda *args, **kwargs: None,
        eval=lambda: None,
    )


def _dummy_dataset():
    return types.SimpleNamespace(CLASSES=(), PALETTE=(), evaluate=lambda *args, **kwargs: {})


def install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv.parallel = types.ModuleType("mmcv.parallel")
    mmcv.parallel.MMDataParallel = object
    mmcv.runner = types.ModuleType("mmcv.runner")
    mmcv.runner.load_checkpoint = lambda *args, **kwargs: {}
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.parallel"] = mmcv.parallel
    sys.modules["mmcv.runner"] = mmcv.runner

    for package in ("mmseg", "mmdet"):
        base = types.ModuleType(package)
        apis = types.ModuleType(f"{package}.apis")
        apis.set_random_seed = lambda *args, **kwargs: None
        if package == "mmseg":
            apis.single_gpu_test = lambda *args, **kwargs: []
            apis.train_segmentor = lambda *args, **kwargs: None
        else:
            apis.train_detector = lambda *args, **kwargs: None

        datasets = types.ModuleType(f"{package}.datasets")
        datasets.build_dataset = lambda *args, **kwargs: _dummy_dataset()
        if package == "mmseg":
            datasets.build_dataloader = lambda *args, **kwargs: None

        models = types.ModuleType(f"{package}.models")
        models.builder = types.ModuleType(f"{package}.models.builder")
        models.builder.BACKBONES = Registry()
        if package == "mmseg":
            models.build_segmentor = lambda *args, **kwargs: _dummy_model()
        else:
            models.build_detector = lambda *args, **kwargs: _dummy_model()

        sys.modules[package] = base
        sys.modules[f"{package}.apis"] = apis
        sys.modules[f"{package}.datasets"] = datasets
        sys.modules[f"{package}.models"] = models
        sys.modules[f"{package}.models.builder"] = models.builder


def make_parser():
    parser = argparse.ArgumentParser(description="Validate seg/det configs")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--workers_per_gpu", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--seg_head_dim", type=int, default=None)
    parser.add_argument("--seg_aux_dim", type=int, default=None)
    parser.add_argument("--seg_neck_dim", type=int, default=None)
    parser.add_argument("--final_eval", default=None)
    return parser


@contextlib.contextmanager
def argv_for_config(path):
    old_argv = sys.argv[:]
    sys.argv = ["validate_seg_det.py", "--cfg", str(path)]
    try:
        yield
    finally:
        sys.argv = old_argv


def load_args(path):
    from utils.cfg import load_cfg

    with argv_for_config(path):
        return load_cfg(make_parser())


def new_task(task_cls, args, run_name):
    task = task_cls.__new__(task_cls)
    task.args = args
    task.run_name = run_name
    return task


def validate_segmentation_configs():
    from tasks.segmentation import SegmentationTask

    paths = sorted((ROOT / "configs").glob("seg_*.yaml"))
    for path in paths:
        args = load_args(path)
        task = new_task(SegmentationTask, args, path.stem)
        cfg = task._build_mmseg_config()
        assert cfg.model.type == "EncoderDecoder"
        assert cfg.model.backbone.type
        assert isinstance(cfg.data.workers_per_gpu, int)
        assert isinstance(cfg.log_config.interval, int)
        assert isinstance(cfg.checkpoint_config.interval, int)
    print(f"[ok] segmentation configs: {len(paths)}")


def validate_detection_configs():
    from tasks.detection import DetectionTask

    paths = sorted((ROOT / "configs").glob("detection_*.yaml"))
    for path in paths:
        args = load_args(path)
        task = new_task(DetectionTask, args, path.stem)
        cfg = task._build_mmdet_config()
        assert cfg.model.type == "MaskRCNN"
        assert cfg.model.backbone.type
        assert isinstance(cfg.data.workers_per_gpu, int)
        assert isinstance(cfg.log_config.interval, int)
    print(f"[ok] detection configs: {len(paths)}")


def validate_backbone_forwards():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    base_kwargs = dict(
        image_size=32,
        patch_size=16,
        dim=32,
        depth=4,
        heads=4,
        mlp_dim=64,
        dim_head=8,
        out_indices=(0, 1, 2, 3),
    )
    expected_shapes = [(2, 32, 8, 8), (2, 32, 4, 4), (2, 32, 2, 2), (2, 32, 1, 1)]
    cases = [
        ("vit", ViTBackbone, {}),
        ("vitcope_no_cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("vitcope_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("vitscope_no_cls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("vitscope_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]

    checked = 0
    torch.manual_seed(0)
    sample = torch.randn(2, 3, 32, 32)
    for adapter_style in ("resize", "simple_fpn"):
        for name, cls, extra_kwargs in cases:
            model = cls(**base_kwargs, fpn_adapter_style=adapter_style, **extra_kwargs)
            model.eval()
            with torch.no_grad():
                outputs = model(sample)
            shapes = [tuple(output.shape) for output in outputs]
            assert shapes == expected_shapes, f"{name}/{adapter_style}: {shapes}"
            checked += 1
    print(f"[ok] backbone CPU forward cases: {checked}")


def main():
    install_openmmlab_stubs()
    validate_segmentation_configs()
    validate_detection_configs()
    validate_backbone_forwards()
    print("[ok] segmentation and detection validation complete")


if __name__ == "__main__":
    main()
