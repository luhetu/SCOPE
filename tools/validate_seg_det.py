#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight validation for segmentation/detection config builders.

This script avoids dataset access and mmcv-full extension imports by stubbing the
small API surface needed to instantiate task configs. It also runs CPU forward
smoke tests for the custom ViT/CoPE/SCoPE dense-prediction backbones.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import os
import sys
import types

import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


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
            if force or key not in self.module_dict:
                self.module_dict[key] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


def _install_stub_modules():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.parallel"] = mmcv_parallel

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
    mmseg_builder.BACKBONES = Registry()
    mmseg_models.builder = mmseg_builder
    sys.modules["mmseg"] = mmseg
    sys.modules["mmseg.apis"] = mmseg_apis
    sys.modules["mmseg.datasets"] = mmseg_datasets
    sys.modules["mmseg.models"] = mmseg_models
    sys.modules["mmseg.models.builder"] = mmseg_builder

    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = lambda *args, **kwargs: None
    mmdet_apis.train_detector = lambda *args, **kwargs: None
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = lambda *args, **kwargs: None
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = lambda *args, **kwargs: None
    mmdet_builder = types.ModuleType("mmdet.models.builder")
    mmdet_builder.BACKBONES = Registry()
    mmdet_models.builder = mmdet_builder
    sys.modules["mmdet"] = mmdet
    sys.modules["mmdet.apis"] = mmdet_apis
    sys.modules["mmdet.datasets"] = mmdet_datasets
    sys.modules["mmdet.models"] = mmdet_models
    sys.modules["mmdet.models.builder"] = mmdet_builder


def _namespace_from_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    defaults = {
        "cfg": path,
        "resume": "",
        "workers_per_gpu": None,
        "model": None,
        "data_dir": None,
        "run_tag": None,
        "time_profile": False,
        "time_profile_interval": 1000,
        "checkpoint_interval": None,
        "eval_interval": None,
        "log_interval": None,
        "final_eval": None,
        "seg_head_dim": None,
        "seg_aux_dim": None,
        "seg_aux_in_index": None,
        "seg_norm_type": None,
        "seg_neck_dim": None,
        "dim_head": None,
        "drop_path_rate": None,
        "layer_decay_rate": None,
        "weight_decay": None,
    }
    defaults.update(values)
    return argparse.Namespace(**defaults)


def _validate_task_configs():
    _install_stub_modules()
    segmentation = importlib.import_module("tasks.segmentation")
    detection = importlib.import_module("tasks.detection")

    seg_paths = sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "seg*.yaml")))
    det_paths = sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "detection*.yaml")))
    if not seg_paths or not det_paths:
        raise RuntimeError("Expected both segmentation and detection config files")

    for path in seg_paths:
        args = _namespace_from_yaml(path)
        task = segmentation.SegmentationTask.__new__(segmentation.SegmentationTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmseg_config()
        assert cfg.model["type"] == "EncoderDecoder"
        assert cfg.data["workers_per_gpu"] == 4

    for path in det_paths:
        args = _namespace_from_yaml(path)
        task = detection.DetectionTask.__new__(detection.DetectionTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmdet_config()
        assert cfg.model["type"] == "MaskRCNN"
        assert cfg.data["workers_per_gpu"] == 4

    print(f"Validated {len(seg_paths)} segmentation configs and {len(det_paths)} detection configs")


def _validate_backbone_forward():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Backbone smoke tests require torch. Install torch before running this script.") from exc

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    expected_shapes = [(1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1)]
    cases = []
    for adapter_style in ("resize", "simple_fpn"):
        common = dict(
            image_size=32,
            patch_size=16,
            dim=32,
            depth=4,
            heads=2,
            mlp_dim=64,
            out_indices=(0, 1, 2, 3),
            fpn_adapter_style=adapter_style,
        )
        cases.append(("vit", ViTBackbone(**common)))
        cases.append(("vitcope_no_cls", ViTCoPEBackbone(**common, use_cls_token=False)))
        cases.append(("vitcope_cls", ViTCoPEBackbone(**common, use_cls_token=True)))
        cases.append(("vitscope_no_cls", ViTSCoPEBackbone(**common, use_cls_token=False)))
        cases.append(("vitscope_cls", ViTSCoPEBackbone(**common, use_cls_token=True)))

    sample = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        for name, model in cases:
            model.eval()
            outputs = model(sample)
            shapes = [tuple(output.shape) for output in outputs]
            if shapes != expected_shapes:
                raise AssertionError(f"{name} produced {shapes}, expected {expected_shapes}")

    print(f"Validated {len(cases)} custom backbone forward cases")


def main():
    _validate_task_configs()
    _validate_backbone_forward()
    print("Segmentation/detection validation passed")


if __name__ == "__main__":
    main()
