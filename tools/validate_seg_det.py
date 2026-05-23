#!/usr/bin/env python3
"""Lightweight segmentation/detection validation.

This script checks dense-prediction configuration builders without requiring
compiled mmcv-full extensions, then runs small CPU forward passes through the
custom ViT backbones used by segmentation and detection.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import types
from types import SimpleNamespace

import yaml


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
            if key in self.module_dict and not force:
                return cls
            self.module_dict[key] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


def install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.parallel"] = mmcv_parallel

    mmseg_backbones = Registry()
    mmseg = types.ModuleType("mmseg")
    mmseg_apis = types.ModuleType("mmseg.apis")
    mmseg_apis.set_random_seed = lambda *args, **kwargs: None
    mmseg_apis.single_gpu_test = lambda *args, **kwargs: []
    mmseg_apis.train_segmentor = lambda *args, **kwargs: None
    mmseg_datasets = types.ModuleType("mmseg.datasets")
    mmseg_datasets.build_dataloader = lambda *args, **kwargs: None
    mmseg_datasets.build_dataset = lambda *args, **kwargs: None
    mmseg_models = types.ModuleType("mmseg.models")
    mmseg_models.build_segmentor = lambda *args, **kwargs: None
    mmseg_builder = types.ModuleType("mmseg.models.builder")
    mmseg_builder.BACKBONES = mmseg_backbones
    sys.modules["mmseg"] = mmseg
    sys.modules["mmseg.apis"] = mmseg_apis
    sys.modules["mmseg.datasets"] = mmseg_datasets
    sys.modules["mmseg.models"] = mmseg_models
    sys.modules["mmseg.models.builder"] = mmseg_builder

    mmdet_backbones = Registry()
    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = lambda *args, **kwargs: None
    mmdet_apis.train_detector = lambda *args, **kwargs: None
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = lambda *args, **kwargs: None
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = lambda *args, **kwargs: None
    mmdet_builder = types.ModuleType("mmdet.models.builder")
    mmdet_builder.BACKBONES = mmdet_backbones
    sys.modules["mmdet"] = mmdet
    sys.modules["mmdet.apis"] = mmdet_apis
    sys.modules["mmdet.datasets"] = mmdet_datasets
    sys.modules["mmdet.models"] = mmdet_models
    sys.modules["mmdet.models.builder"] = mmdet_builder


def load_args(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    # Match common argparse defaults from train.py and exercise None handling.
    data.setdefault("cfg", path)
    data.setdefault("workers_per_gpu", None)
    data.setdefault("log_interval", None)
    data.setdefault("checkpoint_interval", None)
    data.setdefault("eval_interval", None)
    data.setdefault("run_tag", None)
    return SimpleNamespace(**data)


def build_segmentation_configs(paths):
    from tasks.segmentation import SegmentationTask

    for path in paths:
        task = object.__new__(SegmentationTask)
        task.args = load_args(path)
        task.run_name = os.path.splitext(os.path.basename(path))[0]
        cfg = task._build_mmseg_config()
        assert cfg.model["type"] == "EncoderDecoder"
        assert cfg.data["workers_per_gpu"] == 4
        assert cfg.log_config["interval"] == 100
        print(f"OK seg config: {path}")


def build_detection_configs(paths):
    from tasks.detection import DetectionTask

    for path in paths:
        task = object.__new__(DetectionTask)
        task.args = load_args(path)
        task.run_name = os.path.splitext(os.path.basename(path))[0]
        cfg = task._build_mmdet_config()
        assert cfg.model["type"] == "MaskRCNN"
        assert cfg.data["workers_per_gpu"] == 4
        assert cfg.log_config["interval"] == 50
        print(f"OK det config: {path}")


def validate_backbone_forward():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    cases = [
        ("vit-resize", ViTBackbone, {"fpn_adapter_style": "resize"}),
        ("vit-simple-fpn", ViTBackbone, {"fpn_adapter_style": "simple_fpn"}),
        ("cope-resize-no-cls", ViTCoPEBackbone, {"fpn_adapter_style": "resize", "use_cls_token": False}),
        ("cope-simple-fpn-no-cls", ViTCoPEBackbone, {"fpn_adapter_style": "simple_fpn", "use_cls_token": False}),
        ("cope-resize-cls", ViTCoPEBackbone, {"fpn_adapter_style": "resize", "use_cls_token": True}),
        ("cope-simple-fpn-cls", ViTCoPEBackbone, {"fpn_adapter_style": "simple_fpn", "use_cls_token": True}),
        ("scope-resize-no-cls", ViTSCoPEBackbone, {"fpn_adapter_style": "resize", "use_cls_token": False}),
        ("scope-simple-fpn-no-cls", ViTSCoPEBackbone, {"fpn_adapter_style": "simple_fpn", "use_cls_token": False}),
        ("scope-resize-cls", ViTSCoPEBackbone, {"fpn_adapter_style": "resize", "use_cls_token": True}),
        ("scope-simple-fpn-cls", ViTSCoPEBackbone, {"fpn_adapter_style": "simple_fpn", "use_cls_token": True}),
    ]
    expected_hw = [(8, 8), (4, 4), (2, 2), (1, 1)]
    image = torch.randn(1, 3, 32, 32)

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
            **extra,
        )
        model.eval()
        with torch.no_grad():
            outputs = model(image)
        shapes = [tuple(output.shape) for output in outputs]
        assert len(shapes) == 4, (name, shapes)
        for shape, (height, width) in zip(shapes, expected_hw):
            assert shape == (1, 32, height, width), (name, shapes)
        print(f"OK backbone: {name} -> {shapes}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-dir", default="configs")
    args = parser.parse_args()

    install_openmmlab_stubs()
    seg_paths = sorted(glob.glob(os.path.join(args.configs_dir, "seg_*.yaml")))
    det_paths = sorted(glob.glob(os.path.join(args.configs_dir, "detection_*.yaml")))
    if not seg_paths or not det_paths:
        raise RuntimeError("Expected both segmentation and detection config files")

    build_segmentation_configs(seg_paths)
    build_detection_configs(det_paths)
    validate_backbone_forward()
    print(f"Validated {len(seg_paths)} segmentation configs, {len(det_paths)} detection configs, and custom backbone forwards.")


if __name__ == "__main__":
    main()
