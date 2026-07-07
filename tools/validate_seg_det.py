#!/usr/bin/env python3
"""Lightweight validation for segmentation and detection task wiring.

This script validates the code owned by this repository without requiring a
full mmcv-full installation. It stubs the small OpenMMLab API surface needed to
import the task modules, then checks all seg/det YAML files and smoke-tests the
custom dense-prediction backbones on CPU.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Config(dict):
    """Small mmcv.Config-compatible mapping for task config builders."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(dict(*args, **kwargs))

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = self._wrap(value)

    def update(self, other=None, **kwargs):
        data = {}
        if other:
            data.update(other)
        data.update(kwargs)
        for key, value in data.items():
            self[key] = self._wrap(value)
        return None

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, dict) and not isinstance(value, Config):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
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
    mmcv.parallel = mmcv_parallel

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
    mmseg_models_builder = types.ModuleType("mmseg.models.builder")
    mmseg_models_builder.BACKBONES = mmseg_backbones
    mmseg.models = mmseg_models
    mmseg.datasets = mmseg_datasets
    mmseg.apis = mmseg_apis
    mmseg_models.builder = mmseg_models_builder

    mmdet_backbones = Registry()
    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = lambda *args, **kwargs: None
    mmdet_apis.train_detector = lambda *args, **kwargs: None
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = lambda *args, **kwargs: None
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = lambda *args, **kwargs: None
    mmdet_models_builder = types.ModuleType("mmdet.models.builder")
    mmdet_models_builder.BACKBONES = mmdet_backbones
    mmdet.models = mmdet_models
    mmdet.datasets = mmdet_datasets
    mmdet.apis = mmdet_apis
    mmdet_models.builder = mmdet_models_builder

    modules = {
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
    sys.modules.update(modules)


OPTIONAL_DEFAULTS = {
    "amp": None,
    "backbone_size": None,
    "betas": None,
    "checkpoint_interval": None,
    "crop_size": None,
    "det_neck_type": None,
    "dim_head": None,
    "drop_path_rate": None,
    "eval_interval": None,
    "final_eval": None,
    "img_scale": None,
    "layer_decay_rate": None,
    "log_interval": None,
    "min_lr": None,
    "min_pretrained_match_rate": None,
    "nowandb": None,
    "out_indices": None,
    "run_tag": None,
    "seed": None,
    "seg_aux_dim": None,
    "seg_aux_in_index": None,
    "seg_head_dim": None,
    "seg_neck_dim": None,
    "seg_neck_style": None,
    "seg_norm_type": None,
    "test_img_scale": None,
    "use_cls_token": None,
    "warmup_epochs": None,
    "warmup_iters": None,
    "weight_decay": None,
    "workers_per_gpu": None,
}


def _load_args(path: Path) -> SimpleNamespace:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    merged = dict(OPTIONAL_DEFAULTS)
    merged.update(data)
    merged["cfg"] = str(path)
    return SimpleNamespace(**merged)


def _validate_seg_config(path: Path, segmentation_module) -> None:
    args = _load_args(path)
    task = object.__new__(segmentation_module.SegmentationTask)
    task.args = args
    task.run_name = path.stem
    cfg = task._build_mmseg_config()
    assert cfg.model.type == "EncoderDecoder"
    assert cfg.model.backbone.type in {"ViTBackbone", "ViTCoPEBackbone", "ViTSCoPEBackbone", "SwinTransformer"}
    assert isinstance(cfg.data.workers_per_gpu, int)
    assert cfg.data.train.pipeline[0].type == "LoadImageFromFile"
    print(f"OK seg config: {path.relative_to(ROOT)}")


def _validate_det_config(path: Path, detection_module) -> None:
    args = _load_args(path)
    task = object.__new__(detection_module.DetectionTask)
    task.args = args
    task.run_name = path.stem
    cfg = task._build_mmdet_config()
    assert cfg.model.type == "MaskRCNN"
    assert cfg.model.backbone.type in {"ViTBackbone", "ViTCoPEBackbone", "ViTSCoPEBackbone", "SwinTransformer"}
    assert isinstance(cfg.data.workers_per_gpu, int)
    assert cfg.data.train.pipeline[0].type == "LoadImageFromFile"
    print(f"OK det config: {path.relative_to(ROOT)}")


def _validate_backbones() -> None:
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    expected_shapes = [(2, 32, 8, 8), (2, 32, 4, 4), (2, 32, 2, 2), (2, 32, 1, 1)]
    cases = [
        ("ViTBackbone", ViTBackbone, {}),
        ("ViTCoPEBackbone/no_cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("ViTCoPEBackbone/cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("ViTSCoPEBackbone/no_cls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("ViTSCoPEBackbone/cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]
    for adapter_style in ("resize", "simple_fpn"):
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
                fpn_adapter_style=adapter_style,
                **extra,
            )
            model.eval()
            with torch.no_grad():
                outs = model(torch.randn(2, 3, 32, 32))
            shapes = [tuple(out.shape) for out in outs]
            if shapes != expected_shapes:
                raise AssertionError(f"{name}/{adapter_style}: expected {expected_shapes}, got {shapes}")
            print(f"OK backbone: {name} ({adapter_style})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SCOPE segmentation/detection wiring")
    parser.add_argument("--skip-backbone", action="store_true", help="Only validate task YAML/config builders")
    args = parser.parse_args()

    _install_openmmlab_stubs()

    import tasks.detection as detection_module
    import tasks.segmentation as segmentation_module

    seg_paths = sorted((ROOT / "configs").glob("seg_*.yaml"))
    det_paths = sorted((ROOT / "configs").glob("detection_*.yaml"))
    if not seg_paths:
        raise RuntimeError("No segmentation configs found")
    if not det_paths:
        raise RuntimeError("No detection configs found")

    for path in seg_paths:
        _validate_seg_config(path, segmentation_module)
    for path in det_paths:
        _validate_det_config(path, detection_module)
    if not args.skip_backbone:
        _validate_backbones()

    print(f"Validated {len(seg_paths)} segmentation configs and {len(det_paths)} detection configs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
