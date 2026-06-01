#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight segmentation/detection validation without mmcv-full runtime.

The real training stack needs OpenMMLab's compiled extensions. This script
stubs just enough of mmcv/mmseg/mmdet to exercise config builders and then
smoke-tests the custom ViT backbones on CPU.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = _to_attr(value)

    def __setitem__(self, name, value):
        super().__setitem__(name, _to_attr(value))


class Config(AttrDict):
    pass


def _to_attr(value):
    if isinstance(value, dict) and not isinstance(value, AttrDict):
        return AttrDict({key: _to_attr(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_attr(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_attr(item) for item in value)
    return value


class Registry:
    def __init__(self):
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        def _register(cls):
            key = name or cls.__name__
            if key in self.module_dict and not force:
                raise KeyError(f"{key} is already registered")
            self.module_dict[key] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


def _install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    sys.modules["mmcv"] = mmcv

    mmcv_parallel = types.ModuleType("mmcv.parallel")

    class MMDataParallel:
        def __init__(self, module, *args, **kwargs):
            self.module = module

    mmcv_parallel.MMDataParallel = MMDataParallel
    sys.modules["mmcv.parallel"] = mmcv_parallel

    mmseg_backbones = Registry()
    mmdet_backbones = Registry()

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
    mmseg_models.builder = mmseg_builder
    mmseg.models = mmseg_models
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
    mmdet_builder.BACKBONES = mmdet_backbones
    mmdet_models.builder = mmdet_builder
    mmdet.models = mmdet_models
    sys.modules["mmdet"] = mmdet
    sys.modules["mmdet.apis"] = mmdet_apis
    sys.modules["mmdet.datasets"] = mmdet_datasets
    sys.modules["mmdet.models"] = mmdet_models
    sys.modules["mmdet.models.builder"] = mmdet_builder


def _coerce_config_value(key, value):
    if key in {"lr", "min_lr"}:
        return float(value)
    if key in {
        "bs",
        "size",
        "n_epochs",
        "max_iters",
        "warmup_iters",
        "checkpoint_interval",
        "eval_interval",
        "log_interval",
        "patch",
        "dim",
        "depth",
        "heads",
        "mlp_dim",
        "dim_head",
        "seg_head_dim",
        "seg_aux_dim",
        "seg_neck_dim",
        "seg_aux_in_index",
    }:
        return int(value)
    if key in {"warmup_epochs", "drop_path_rate", "weight_decay", "dropout", "emb_dropout", "layer_decay_rate"}:
        return float(value)
    if key in {"amp", "aug", "nowandb", "use_cls_token", "final_eval"}:
        return bool(value)
    if key == "pretrained" and value == "null":
        return None
    return value


def _load_args(path):
    args = argparse.Namespace(
        cfg=str(path),
        resume="",
        workers_per_gpu=None,
        model=None,
        data_dir=None,
        time_profile=False,
        time_profile_interval=1000,
        run_tag=None,
    )
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    for key, value in data.items():
        setattr(args, key, _coerce_config_value(key, value))
    return args


def _validate_segmentation_configs(paths):
    from tasks.segmentation import SegmentationTask

    for path in paths:
        args = _load_args(path)
        task = SegmentationTask.__new__(SegmentationTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmseg_config()
        if cfg.data.workers_per_gpu != 4:
            raise AssertionError(f"{path}: expected default workers_per_gpu=4")
        if cfg.checkpoint_config.interval <= 0:
            raise AssertionError(f"{path}: checkpoint interval must be positive")
    return len(paths)


def _validate_detection_configs(paths):
    from tasks.detection import DetectionTask

    for path in paths:
        args = _load_args(path)
        task = DetectionTask.__new__(DetectionTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmdet_config()
        if cfg.data.workers_per_gpu != 4:
            raise AssertionError(f"{path}: expected default workers_per_gpu=4")
        if args.model == "swin" and cfg.model.backbone.type != "SwinTransformer":
            raise AssertionError(f"{path}: expected SwinTransformer backbone")
    return len(paths)


def _validate_backbones():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    cases = []
    for style in ("resize", "simple_fpn"):
        common = dict(
            image_size=32,
            patch_size=16,
            dim=16,
            depth=4,
            heads=2,
            mlp_dim=32,
            dim_head=8,
            out_indices=(0, 1, 2, 3),
            fpn_adapter_style=style,
        )
        cases.append((f"ViTBackbone/{style}", ViTBackbone(**common)))
        for use_cls_token in (False, True):
            cases.append((f"ViTCoPEBackbone/{style}/cls={use_cls_token}", ViTCoPEBackbone(use_cls_token=use_cls_token, **common)))
            cases.append((f"ViTSCoPEBackbone/{style}/cls={use_cls_token}", ViTSCoPEBackbone(use_cls_token=use_cls_token, **common)))

    expected_hw = [(8, 8), (4, 4), (2, 2), (1, 1)]
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        for name, model in cases:
            model.eval()
            outputs = model(x)
            if len(outputs) != 4:
                raise AssertionError(f"{name}: expected 4 outputs, got {len(outputs)}")
            for idx, (out, hw) in enumerate(zip(outputs, expected_hw)):
                expected_shape = (2, 16, hw[0], hw[1])
                if tuple(out.shape) != expected_shape:
                    raise AssertionError(f"{name} output {idx}: {tuple(out.shape)} != {expected_shape}")
    return len(cases)


def main():
    _install_openmmlab_stubs()

    config_dir = REPO_ROOT / "configs"
    seg_paths = sorted(config_dir.glob("seg_*.yaml"))
    det_paths = sorted(config_dir.glob("detection_*.yaml"))
    if not seg_paths:
        raise RuntimeError("No segmentation configs found")
    if not det_paths:
        raise RuntimeError("No detection configs found")

    seg_count = _validate_segmentation_configs(seg_paths)
    det_count = _validate_detection_configs(det_paths)
    backbone_count = _validate_backbones()
    print(f"Validated {seg_count} segmentation configs")
    print(f"Validated {det_count} detection configs")
    print(f"Validated {backbone_count} backbone smoke cases")


if __name__ == "__main__":
    main()
