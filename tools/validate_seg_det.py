#!/usr/bin/env python3
"""Lightweight smoke checks for segmentation/detection configuration code.

The full training path requires mmcv-full/mmseg/mmdet compiled extensions and
datasets. This script stubs only the imported OpenMMLab registries/APIs so the
SCOPE config builders and custom dense-prediction backbones can be validated in
a minimal CPU environment.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Config(dict):
    """Small stand-in for mmcv.Config with attribute access."""

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
                return cls
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

    modules = {
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
    }
    sys.modules.update(modules)


COMMON_DEFAULTS = {
    "cfg": "",
    "data_dir": ".",
    "bs": 1,
    "lr": 1e-4,
    "min_lr": 0.0,
    "n_epochs": 1,
    "warmup_epochs": 0,
    "warmup_iters": None,
    "weight_decay": None,
    "amp": False,
    "nowandb": True,
    "pretrained": None,
    "seed": None,
    "run_tag": None,
    "workers_per_gpu": None,
    "log_interval": None,
    "size": 224,
    "patch": 16,
    "dim": 192,
    "depth": 4,
    "heads": 3,
    "mlp_dim": 768,
    "dim_head": 64,
    "drop_path_rate": 0.0,
    "use_cls_token": None,
}

SEG_DEFAULTS = {
    "task": "seg",
    "crop_size": None,
    "img_scale": None,
    "test_img_scale": None,
    "max_iters": None,
    "checkpoint_interval": None,
    "eval_interval": None,
    "final_eval": True,
    "layer_decay_rate": 1.0,
    "seg_head_dim": None,
    "seg_aux_dim": None,
    "seg_aux_in_index": None,
    "seg_norm_type": None,
    "seg_neck_style": None,
    "seg_neck_dim": None,
}

DET_DEFAULTS = {
    "task": "det",
    "img_scale": None,
    "det_neck_type": None,
}


def _load_args(path: Path, extra_defaults: dict) -> SimpleNamespace:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    merged = {**COMMON_DEFAULTS, **extra_defaults, **data, "cfg": str(path)}
    return SimpleNamespace(**merged)


def _validate_seg_configs():
    from tasks.segmentation import SegmentationTask

    configs = sorted((REPO_ROOT / "configs").glob("seg_*.yaml"))
    if not configs:
        raise AssertionError("No segmentation configs found")

    for path in configs:
        args = _load_args(path, SEG_DEFAULTS)
        task = SegmentationTask.__new__(SegmentationTask)
        task.args = args
        task.run_name = SegmentationTask._build_run_name(task)
        cfg = SegmentationTask._build_mmseg_config(task)
        assert cfg.model["type"] == "EncoderDecoder", path
        assert cfg.data["samples_per_gpu"] == args.bs, path
        assert cfg.checkpoint_config["interval"] > 0, path
        assert cfg.log_config["interval"] > 0, path
        assert cfg.evaluation["interval"] > 0, path
        if args.model == "swin":
            assert cfg.model["backbone"]["type"] == "SwinTransformer", path
        else:
            assert len(cfg.model["decode_head"]["in_channels"]) == 4, path
    print(f"Validated {len(configs)} segmentation configs")


def _validate_det_configs():
    from tasks.detection import DetectionTask

    configs = sorted((REPO_ROOT / "configs").glob("detection_*.yaml"))
    if not configs:
        raise AssertionError("No detection configs found")

    for path in configs:
        args = _load_args(path, DET_DEFAULTS)
        task = DetectionTask.__new__(DetectionTask)
        task.args = args
        task.run_name = DetectionTask._build_run_name(task)
        cfg = DetectionTask._build_mmdet_config(task)
        assert cfg.model["type"] == "MaskRCNN", path
        assert cfg.data["samples_per_gpu"] == args.bs, path
        assert cfg.log_config["interval"] > 0, path
        if args.model == "swin":
            backbone = cfg.model["backbone"]
            assert backbone["type"] == "SwinTransformer", path
            assert "image_size" not in backbone, path
        else:
            assert cfg.model["backbone"]["type"] in {"ViTBackbone", "ViTCoPEBackbone", "ViTSCoPEBackbone"}, path
    print(f"Validated {len(configs)} detection configs")


def _validate_backbones():
    import torch
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    expected_shapes = [(1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1)]
    cases = []
    for style in ("resize", "simple_fpn"):
        cases.append((ViTBackbone, {}, style))
        for use_cls_token in (False, True):
            cases.append((ViTCoPEBackbone, {"use_cls_token": use_cls_token}, style))
            cases.append((ViTSCoPEBackbone, {"use_cls_token": use_cls_token}, style))

    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        for cls, extra_kwargs, style in cases:
            model = cls(
                image_size=32,
                patch_size=16,
                dim=32,
                depth=4,
                heads=2,
                mlp_dim=64,
                dim_head=16,
                out_indices=(0, 1, 2, 3),
                fpn_adapter_style=style,
                **extra_kwargs,
            )
            model.eval()
            outputs = model(x)
            shapes = [tuple(out.shape) for out in outputs]
            assert shapes == expected_shapes, (cls.__name__, extra_kwargs, style, shapes)
    print(f"Validated {len(cases)} backbone forward cases")


def main():
    _install_openmmlab_stubs()
    _validate_seg_configs()
    _validate_det_configs()
    _validate_backbones()
    print("Segmentation/detection smoke validation passed")


if __name__ == "__main__":
    main()
