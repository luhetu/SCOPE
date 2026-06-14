#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight smoke validation for segmentation and detection wiring.

The full train path requires mmcv-full CUDA extensions. This script stubs the
small OpenMMLab surface needed to validate config construction and custom
backbone forward passes in a plain CPU environment.
"""

import argparse
import contextlib
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ConfigDict(dict):
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
        for key, value in dict(*args, **kwargs).items():
            self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._wrap(item) for item in value)
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

        return _register(module) if module is not None else _register


def _install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = ConfigDict
    mmcv.__version__ = "1.3.17"

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

    mmseg.models = mmseg_models
    mmseg.datasets = mmseg_datasets
    mmseg.apis = mmseg_apis
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

    mmdet.models = mmdet_models
    mmdet.datasets = mmdet_datasets
    mmdet.apis = mmdet_apis
    mmdet_models.builder = mmdet_builder

    sys.modules["mmdet"] = mmdet
    sys.modules["mmdet.apis"] = mmdet_apis
    sys.modules["mmdet.datasets"] = mmdet_datasets
    sys.modules["mmdet.models"] = mmdet_models
    sys.modules["mmdet.models.builder"] = mmdet_builder


def _make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--workers_per_gpu", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--time_profile", action="store_true")
    parser.add_argument("--time_profile_interval", type=int, default=1000)
    return parser


@contextlib.contextmanager
def _argv_for_config(path):
    original = sys.argv[:]
    sys.argv = ["validate_seg_det.py", "--cfg", str(path)]
    try:
        yield
    finally:
        sys.argv = original


def _load_args(path):
    from utils.cfg import load_cfg

    with _argv_for_config(path):
        return load_cfg(_make_parser())


def _validate_segmentation_config(path):
    from tasks.segmentation import SegmentationTask

    args = _load_args(path)
    task = SegmentationTask.__new__(SegmentationTask)
    task.args = args
    task.device = "cpu"
    task.run_name = task._build_run_name()
    cfg = task._build_mmseg_config()

    assert cfg.model.type == "EncoderDecoder"
    assert cfg.data.workers_per_gpu == 4
    assert cfg.log_config.interval > 0
    assert cfg.evaluation.interval > 0
    if args.model == "swin":
        assert cfg.model.backbone.type == "SwinTransformer"
    else:
        assert cfg.model.backbone.type in {"ViTBackbone", "ViTCoPEBackbone", "ViTSCoPEBackbone"}
        assert len(cfg.model.backbone.out_indices) == 4
    return path.name


def _validate_detection_config(path):
    from tasks.detection import DetectionTask

    args = _load_args(path)
    task = DetectionTask.__new__(DetectionTask)
    task.args = args
    task.device = "cpu"
    task.run_name = task._build_run_name()
    cfg = task._build_mmdet_config()

    assert cfg.model.type == "MaskRCNN"
    assert cfg.data.workers_per_gpu == 4
    assert cfg.log_config.interval > 0
    if args.model == "swin":
        assert cfg.model.backbone.type == "SwinTransformer"
    else:
        assert cfg.model.backbone.type in {"ViTBackbone", "ViTCoPEBackbone", "ViTSCoPEBackbone"}
        assert len(cfg.model.backbone.out_indices) == 4
    return path.name


def _expected_feature_shapes(dim):
    return [
        (1, dim, 8, 8),
        (1, dim, 4, 4),
        (1, dim, 2, 2),
        (1, dim, 1, 1),
    ]


def _validate_backbones():
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    cases = []
    for style in ("simple_fpn", "resize"):
        cases.append(("vit", style, ViTBackbone, {}))
        cases.append(("vitcope_no_cls", style, ViTCoPEBackbone, {"use_cls_token": False}))
        cases.append(("vitcope_cls", style, ViTCoPEBackbone, {"use_cls_token": True}))
        cases.append(("vitscope_no_cls", style, ViTSCoPEBackbone, {"use_cls_token": False}))
        cases.append(("vitscope_cls", style, ViTSCoPEBackbone, {"use_cls_token": True}))

    x = torch.randn(1, 3, 32, 32)
    expected = _expected_feature_shapes(dim=32)
    validated = []
    with torch.no_grad():
        for name, style, cls, extra in cases:
            model = cls(
                image_size=32,
                patch_size=16,
                dim=32,
                depth=4,
                heads=4,
                mlp_dim=64,
                dim_head=8,
                out_indices=(0, 1, 2, 3),
                fpn_adapter_style=style,
                **extra,
            )
            model.eval()
            outputs = model(x)
            shapes = [tuple(output.shape) for output in outputs]
            assert shapes == expected, f"{name}/{style}: expected {expected}, got {shapes}"
            validated.append(f"{name}/{style}")
    return validated


def main():
    _install_openmmlab_stubs()

    seg_configs = sorted((ROOT / "configs").glob("seg*.yaml"))
    det_configs = sorted((ROOT / "configs").glob("detection_*.yaml"))

    seg_validated = [_validate_segmentation_config(path) for path in seg_configs]
    det_validated = [_validate_detection_config(path) for path in det_configs]
    backbone_validated = _validate_backbones()

    print(f"Validated {len(seg_validated)} segmentation configs")
    print(f"Validated {len(det_validated)} detection configs")
    print(f"Validated {len(backbone_validated)} custom backbone CPU forward cases")


if __name__ == "__main__":
    main()
