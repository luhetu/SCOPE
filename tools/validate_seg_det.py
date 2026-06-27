#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight checks for segmentation/detection config and backbone code."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Config(dict):
    """Minimal mmcv.Config stand-in for task config builder validation."""

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

        if module is not None:
            return _register(module)
        return _register


def _noop(*args, **kwargs):
    return None


def install_openmmlab_stubs():
    """Install enough stubs to import task modules without mmcv-full."""

    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object
    mmcv.parallel = mmcv_parallel

    mmseg_backbones = Registry()
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
    mmseg_builder = types.ModuleType("mmseg.models.builder")
    mmseg_builder.BACKBONES = mmseg_backbones
    mmseg.models = mmseg_models
    mmseg.datasets = mmseg_datasets
    mmseg.apis = mmseg_apis

    mmdet_backbones = Registry()
    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = _noop
    mmdet_apis.train_detector = _noop
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = _noop
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = _noop
    mmdet_builder = types.ModuleType("mmdet.models.builder")
    mmdet_builder.BACKBONES = mmdet_backbones
    mmdet.models = mmdet_models
    mmdet.datasets = mmdet_datasets
    mmdet.apis = mmdet_apis

    sys.modules.update(
        {
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
    )


def load_args(config_path: Path) -> SimpleNamespace:
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}

    defaults = {
        "cfg": str(config_path),
        "workers_per_gpu": None,
        "pretrained": None,
        "seed": None,
        "run_tag": None,
        "nowandb": True,
        "amp": False,
        "betas": None,
        "min_lr": 0.0,
        "warmup_epochs": 0,
        "warmup_iters": None,
        "max_iters": None,
        "checkpoint_interval": 5000,
        "eval_interval": 2001,
        "log_interval": 100,
        "weight_decay": 0.01,
        "drop_path_rate": 0.0,
        "layer_decay_rate": 1.0,
        "dim_head": 64,
        "out_indices": (3, 5, 7, 11),
        "crop_size": 512,
        "img_scale": None,
        "test_img_scale": None,
        "seg_aux_in_index": 2,
        "seg_norm_type": "SyncBN",
        "seg_neck_style": "xcit_fpn",
        "final_eval": True,
        "det_neck_type": "fpn",
        "use_cls_token": False,
    }
    defaults.update(values)
    if defaults.get("pretrained") == "null":
        defaults["pretrained"] = None
    return SimpleNamespace(**defaults)


def validate_config_builders():
    install_openmmlab_stubs()

    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    seg_configs = sorted((ROOT / "configs").glob("seg_*.yaml"))
    det_configs = sorted((ROOT / "configs").glob("detection_*.yaml"))
    if not seg_configs:
        raise AssertionError("No segmentation configs found")
    if not det_configs:
        raise AssertionError("No detection configs found")

    for path in seg_configs:
        args = load_args(path)
        task = object.__new__(SegmentationTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmseg_config()
        assert cfg.data["workers_per_gpu"] == 4, path
        assert cfg.model["type"] == "EncoderDecoder", path
        assert "backbone" in cfg.model, path

    for path in det_configs:
        args = load_args(path)
        task = object.__new__(DetectionTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmdet_config()
        assert cfg.data["workers_per_gpu"] == 4, path
        assert cfg.model["type"] == "MaskRCNN", path
        backbone = cfg.model["backbone"]
        assert "type" in backbone, path
        if args.model == "swin":
            assert backbone["type"] == "SwinTransformer", path
            assert "dim" not in backbone and "depth" not in backbone, path

    print(f"Validated {len(seg_configs)} segmentation configs")
    print(f"Validated {len(det_configs)} detection configs")


def validate_custom_backbones():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    torch.manual_seed(0)
    image = torch.randn(1, 3, 32, 32)
    expected_shapes = [(1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1)]
    cases = [
        (ViTBackbone, {}),
        (ViTCoPEBackbone, {"use_cls_token": False}),
        (ViTCoPEBackbone, {"use_cls_token": True}),
        (ViTSCoPEBackbone, {"use_cls_token": False}),
        (ViTSCoPEBackbone, {"use_cls_token": True}),
    ]

    for adapter_style in ("resize", "simple_fpn"):
        for cls, extra_kwargs in cases:
            model = cls(
                image_size=32,
                patch_size=16,
                dim=32,
                depth=4,
                heads=4,
                mlp_dim=64,
                dim_head=8,
                out_indices=(0, 1, 2, 3),
                fpn_adapter_style=adapter_style,
                **extra_kwargs,
            ).eval()
            with torch.no_grad():
                outputs = model(image)
            shapes = [tuple(output.shape) for output in outputs]
            if shapes != expected_shapes:
                raise AssertionError(f"{cls.__name__} {adapter_style}: {shapes}")

    print("Validated custom backbone CPU forward smoke tests")


def main():
    validate_config_builders()
    validate_custom_backbones()
    print("Segmentation/detection validation passed")


if __name__ == "__main__":
    main()
