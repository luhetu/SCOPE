#!/usr/bin/env python3
"""Lightweight validation for segmentation and detection task wiring.

The full training path needs mmcv-full CUDA/C++ operators. This script stubs
the small OpenMMLab surface needed to import the task modules, then verifies
that every seg/det YAML builds a task config and that the custom dense-prediction
backbones run a CPU forward pass.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml


ROOT = Path(__file__).resolve().parents[1]


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
        self[name] = self._wrap(value)

    def update(self, *args, **kwargs):
        values = dict(*args, **kwargs)
        for key, value in values.items():
            self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, dict) and not isinstance(value, Config):
            return Config(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._wrap(item) for item in value)
        return value


class Registry:
    def __init__(self):
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        def decorator(cls):
            key = name or cls.__name__
            if not force and key in self.module_dict and self.module_dict[key] is not cls:
                raise KeyError(f"{key} is already registered")
            self.module_dict[key] = cls
            return cls

        if module is not None:
            return decorator(module)
        return decorator


def _noop(*args, **kwargs):
    return None


def install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    sys.modules["mmcv"] = mmcv

    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object
    sys.modules["mmcv.parallel"] = mmcv_parallel

    for package_name, build_name, train_name in (
        ("mmseg", "build_segmentor", "train_segmentor"),
        ("mmdet", "build_detector", "train_detector"),
    ):
        registry = Registry()

        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package

        apis = types.ModuleType(f"{package_name}.apis")
        apis.set_random_seed = _noop
        setattr(apis, train_name, _noop)
        if package_name == "mmseg":
            apis.single_gpu_test = lambda *args, **kwargs: []
        sys.modules[f"{package_name}.apis"] = apis

        datasets = types.ModuleType(f"{package_name}.datasets")
        datasets.build_dataset = _noop
        if package_name == "mmseg":
            datasets.build_dataloader = _noop
        sys.modules[f"{package_name}.datasets"] = datasets

        models = types.ModuleType(f"{package_name}.models")
        setattr(models, build_name, _noop)
        sys.modules[f"{package_name}.models"] = models

        builder = types.ModuleType(f"{package_name}.models.builder")
        builder.BACKBONES = registry
        sys.modules[f"{package_name}.models.builder"] = builder


DEFAULT_ARGS = {
    "amp": False,
    "backbone_size": None,
    "betas": None,
    "bs": 1,
    "checkpoint_interval": None,
    "crop_size": None,
    "data_dir": "/tmp/dataset",
    "det_neck_type": None,
    "dim_head": None,
    "drop_path_rate": None,
    "eval_interval": None,
    "final_eval": False,
    "img_scale": None,
    "layer_decay_rate": None,
    "log_interval": None,
    "lr": 1e-4,
    "max_iters": None,
    "min_lr": 0.0,
    "model": "vit",
    "n_epochs": 1,
    "nowandb": True,
    "out_indices": None,
    "pretrained": None,
    "run_tag": None,
    "seed": None,
    "seg_aux_dim": None,
    "seg_aux_in_index": None,
    "seg_head_dim": None,
    "seg_neck_dim": None,
    "seg_neck_style": None,
    "seg_norm_type": None,
    "size": 224,
    "task": None,
    "test_img_scale": None,
    "use_cls_token": None,
    "warmup_epochs": None,
    "warmup_iters": None,
    "weight_decay": None,
    "workers_per_gpu": None,
}


def load_args(path: Path) -> SimpleNamespace:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    values = dict(DEFAULT_ARGS)
    values.update(raw)
    values["cfg"] = str(path)
    return SimpleNamespace(**values)


def validate_segmentation_configs(paths):
    from tasks.segmentation import SegmentationTask

    for path in paths:
        args = load_args(path)
        task = SegmentationTask.__new__(SegmentationTask)
        task.args = args
        task.device = "cpu"
        task.run_name = task._build_run_name()
        cfg = task._build_mmseg_config()

        assert cfg.model.type == "EncoderDecoder"
        assert cfg.model.decode_head.num_classes == 150
        assert cfg.data.train.type == "ADE20KDataset"
        assert cfg.data.workers_per_gpu == 4
        assert len(cfg.model.decode_head.in_channels) == 4
        print(f"[seg-config] OK {path}")


def validate_detection_configs(paths):
    from tasks.detection import DetectionTask

    for path in paths:
        args = load_args(path)
        task = DetectionTask.__new__(DetectionTask)
        task.args = args
        task.device = "cpu"
        task.run_name = task._build_run_name()
        cfg = task._build_mmdet_config()

        assert cfg.model.type == "MaskRCNN"
        assert cfg.model.roi_head.bbox_head.num_classes == 80
        assert cfg.model.roi_head.mask_head.num_classes == 80
        assert cfg.data.train.type == "CocoDataset"
        assert cfg.data.workers_per_gpu == 4
        assert len(cfg.model.neck.in_channels) == 4
        print(f"[det-config] OK {path}")


def validate_custom_backbones():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    common = dict(
        image_size=32,
        patch_size=16,
        dim=32,
        depth=4,
        heads=2,
        mlp_dim=64,
        dim_head=16,
        out_indices=(0, 1, 2, 3),
    )
    expected_shapes = [(1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1)]
    cases = []
    for adapter_style in ("resize", "simple_fpn"):
        cases.extend(
            [
                ("vit", ViTBackbone(fpn_adapter_style=adapter_style, **common)),
                ("vitcope-no-cls", ViTCoPEBackbone(use_cls_token=False, fpn_adapter_style=adapter_style, **common)),
                ("vitcope-cls", ViTCoPEBackbone(use_cls_token=True, fpn_adapter_style=adapter_style, **common)),
                ("vitscope-cls", ViTSCoPEBackbone(use_cls_token=True, fpn_adapter_style=adapter_style, **common)),
                ("vitscope-no-cls", ViTSCoPEBackbone(use_cls_token=False, fpn_adapter_style=adapter_style, **common)),
            ]
        )

    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        for name, model in cases:
            model.eval()
            outputs = model(x)
            shapes = [tuple(output.shape) for output in outputs]
            assert shapes == expected_shapes, f"{name}: expected {expected_shapes}, got {shapes}"
            print(f"[backbone] OK {name}: {shapes}")


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    install_openmmlab_stubs()

    seg_configs = sorted((ROOT / "configs").glob("seg_*.yaml"))
    det_configs = sorted((ROOT / "configs").glob("detection_*.yaml"))
    assert seg_configs, "No segmentation configs found"
    assert det_configs, "No detection configs found"

    validate_segmentation_configs(seg_configs)
    validate_detection_configs(det_configs)
    validate_custom_backbones()
    print(f"Validated {len(seg_configs)} segmentation configs, {len(det_configs)} detection configs, and 10 backbone cases.")


if __name__ == "__main__":
    main()
