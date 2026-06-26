#!/usr/bin/env python3
"""Lightweight checks for SCOPE segmentation and detection code paths.

This script intentionally stubs the OpenMMLab packages before importing the
task modules. It validates the repository-owned config builders and custom
ViT/CoPE/SCoPE backbones without requiring ADE20K, COCO, CUDA, or mmcv-full.
"""

from __future__ import annotations

import glob
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ConfigDict(dict):
    """Tiny attribute-access dict compatible with the task config builders."""

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

    def __setitem__(self, name, value):
        super().__setitem__(name, self._wrap(value))

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._wrap(item) for item in value)
        return value


class Config(ConfigDict):
    pass


class Registry:
    def __init__(self):
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False, **kwargs):
        def _register(cls):
            key = name or cls.__name__
            if not force and key in self.module_dict and self.module_dict[key] is not cls:
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
    mmdet_models.builder = mmdet_builder
    mmdet.models = mmdet_models
    mmdet.datasets = mmdet_datasets
    mmdet.apis = mmdet_apis

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
    mmseg_models.builder = mmseg_builder
    mmseg.models = mmseg_models
    mmseg.datasets = mmseg_datasets
    mmseg.apis = mmseg_apis

    sys.modules.update({
        "mmcv": mmcv,
        "mmcv.parallel": mmcv_parallel,
        "mmdet": mmdet,
        "mmdet.apis": mmdet_apis,
        "mmdet.datasets": mmdet_datasets,
        "mmdet.models": mmdet_models,
        "mmdet.models.builder": mmdet_builder,
        "mmseg": mmseg,
        "mmseg.apis": mmseg_apis,
        "mmseg.datasets": mmseg_datasets,
        "mmseg.models": mmseg_models,
        "mmseg.models.builder": mmseg_builder,
    })


def _coerce_config_types(cfg):
    int_keys = {
        "bs", "size", "n_epochs", "max_iters", "warmup_iters",
        "checkpoint_interval", "eval_interval", "log_interval", "patch",
        "dim", "depth", "heads", "mlp_dim", "dim_head", "seg_head_dim",
        "seg_aux_dim", "seg_aux_in_index", "seg_neck_dim",
    }
    float_keys = {
        "lr", "min_lr", "warmup_epochs", "drop_path_rate", "weight_decay",
        "dropout", "emb_dropout", "layer_decay_rate",
    }
    bool_keys = {"amp", "aug", "nowandb", "use_cls_token", "final_eval"}
    for key, value in list(cfg.items()):
        if value is None:
            continue
        if key in int_keys:
            cfg[key] = int(value)
        elif key in float_keys:
            cfg[key] = float(value)
        elif key in bool_keys:
            cfg[key] = bool(value)
    return cfg


def _load_args(path):
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = _coerce_config_types(raw)
    defaults = {
        "cfg": str(path),
        "run_tag": None,
        "workers_per_gpu": None,
    }
    defaults.update(raw)
    return SimpleNamespace(**defaults)


def _task_shell(task_cls, args):
    task = object.__new__(task_cls)
    task.args = args
    task.run_name = Path(args.cfg).stem
    return task


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_configs():
    _install_openmmlab_stubs()

    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    seg_paths = sorted(glob.glob(str(REPO_ROOT / "configs" / "seg_*.yaml")))
    det_paths = sorted(glob.glob(str(REPO_ROOT / "configs" / "detection_*.yaml")))
    _assert(seg_paths, "No segmentation configs found")
    _assert(det_paths, "No detection configs found")

    for path in seg_paths:
        args = _load_args(path)
        cfg = _task_shell(SegmentationTask, args)._build_mmseg_config()
        _assert(cfg.model.type == "EncoderDecoder", f"{path}: expected EncoderDecoder")
        _assert(cfg.model.decode_head.num_classes == 150, f"{path}: expected ADE20K classes")
        _assert(cfg.data.workers_per_gpu == 4, f"{path}: workers_per_gpu default should be 4")
        _assert(cfg.data.train.type == "ADE20KDataset", f"{path}: expected ADE20KDataset")
        _assert(cfg.runner.type == "IterBasedRunner", f"{path}: expected iter runner")

    for path in det_paths:
        args = _load_args(path)
        cfg = _task_shell(DetectionTask, args)._build_mmdet_config()
        _assert(cfg.model.type == "MaskRCNN", f"{path}: expected MaskRCNN")
        _assert(cfg.model.roi_head.bbox_head.num_classes == 80, f"{path}: expected COCO bbox classes")
        _assert(cfg.model.roi_head.mask_head.num_classes == 80, f"{path}: expected COCO mask classes")
        _assert(cfg.data.workers_per_gpu == 4, f"{path}: workers_per_gpu default should be 4")
        _assert(cfg.data.train.type == "CocoDataset", f"{path}: expected CocoDataset")
        _assert(cfg.runner.type == "EpochBasedRunner", f"{path}: expected epoch runner")

    print(f"Config builders OK: {len(seg_paths)} segmentation, {len(det_paths)} detection")


def _expected_shapes(dim):
    return [
        (1, dim, 8, 8),
        (1, dim, 4, 4),
        (1, dim, 2, 2),
        (1, dim, 1, 1),
    ]


def validate_backbones():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    dim = 32
    base_kwargs = dict(
        image_size=32,
        patch_size=16,
        dim=dim,
        depth=4,
        heads=4,
        mlp_dim=64,
        dim_head=8,
        out_indices=(0, 1, 2, 3),
    )
    cases = [
        ("ViTBackbone", ViTBackbone, {}),
        ("ViTCoPEBackbone/nocls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("ViTCoPEBackbone/cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("ViTSCoPEBackbone/nocls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("ViTSCoPEBackbone/cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]

    x = torch.randn(1, 3, 32, 32)
    expected = _expected_shapes(dim)
    for style in ("resize", "simple_fpn"):
        for name, cls, extra_kwargs in cases:
            model = cls(**base_kwargs, fpn_adapter_style=style, **extra_kwargs)
            model.eval()
            with torch.no_grad():
                outs = model(x)
            shapes = [tuple(out.shape) for out in outs]
            _assert(shapes == expected, f"{name} ({style}) shapes {shapes} != {expected}")

    print(f"Backbone forward OK: {len(cases)} variants x 2 adapter styles")


def main():
    try:
        import torch  # noqa: F401
        import timm  # noqa: F401
        import einops  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing lightweight dependency for validation: "
            f"{exc}. Install torch, timm, and einops first."
        ) from exc

    validate_configs()
    validate_backbones()
    print("Segmentation and detection lightweight validation passed.")


if __name__ == "__main__":
    main()
