#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight validation for segmentation and detection integration.

The full MMDetection/MMSegmentation trainer needs mmcv-full CUDA extensions.
This script stubs only the small API surface required to import the task
builders, then validates generated configs and custom ViT backbone forwards.
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml


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
            return ConfigDict(value)
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
            if not force and key in self.module_dict:
                raise KeyError(f"{key} is already registered")
            self.module_dict[key] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


def _noop(*args, **kwargs):
    return None


def _install_stubs():
    mmdet_backbones = Registry()
    mmseg_backbones = Registry()

    mmcv = types.ModuleType("mmcv")
    mmcv.Config = ConfigDict
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object

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

    stubs = {
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
    sys.modules.update(stubs)


def _load_args(path):
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    args = SimpleNamespace(**data)
    args.cfg = str(path)
    args.workers_per_gpu = getattr(args, "workers_per_gpu", None)
    args.run_tag = getattr(args, "run_tag", None)
    args.seed = getattr(args, "seed", None)
    args.nowandb = True
    args.pretrained = getattr(args, "pretrained", None)
    return args


def _build_task_config(task_cls, args, method_name):
    task = task_cls.__new__(task_cls)
    task.args = args
    task.device = "cpu"
    task.run_name = task._build_run_name()
    return getattr(task, method_name)()


def _validate_segmentation_configs():
    from tasks.segmentation import SegmentationTask

    paths = sorted((ROOT / "configs").glob("seg_*.yaml"))
    for path in paths:
        args = _load_args(path)
        cfg = _build_task_config(SegmentationTask, args, "_build_mmseg_config")
        assert cfg.model.backbone.type.endswith("Backbone") or cfg.model.backbone.type == "SwinTransformer", path
        assert cfg.data.workers_per_gpu == 4, path
        assert cfg.data.train.type == "ADE20KDataset", path
        assert cfg.evaluation.metric == "mIoU", path
    print(f"Validated {len(paths)} segmentation configs")


def _validate_detection_configs():
    from tasks.detection import DetectionTask

    paths = sorted((ROOT / "configs").glob("detection_*.yaml"))
    for path in paths:
        args = _load_args(path)
        cfg = _build_task_config(DetectionTask, args, "_build_mmdet_config")
        assert cfg.model.type == "MaskRCNN", path
        assert cfg.model.backbone.type.endswith("Backbone") or cfg.model.backbone.type == "SwinTransformer", path
        assert cfg.data.workers_per_gpu == 4, path
        assert cfg.data.train.type == "CocoDataset", path
        if args.model == "swin":
            assert cfg.model.backbone.type == "SwinTransformer", path
            assert hasattr(args, "size"), path
    print(f"Validated {len(paths)} detection configs")


def _validate_backbone_forwards():
    import torch
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    cases = [
        ("ViTBackbone", ViTBackbone, {}),
        ("ViTCoPEBackbone/no-cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("ViTCoPEBackbone/cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("ViTSCoPEBackbone/no-cls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("ViTSCoPEBackbone/cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]
    expected_hw = [(8, 8), (4, 4), (2, 2), (1, 1)]
    count = 0
    x = torch.randn(2, 3, 32, 32)
    for style in ("resize", "simple_fpn"):
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
                fpn_adapter_style=style,
                **extra,
            ).eval()
            with torch.no_grad():
                outputs = model(x)
            shapes = [tuple(out.shape) for out in outputs]
            expected = [(2, 32, h, w) for h, w in expected_hw]
            assert shapes == expected, f"{name}/{style}: {shapes} != {expected}"
            count += 1
    print(f"Validated {count} backbone forward smoke cases")


def main():
    _install_stubs()
    _validate_segmentation_configs()
    _validate_detection_configs()
    _validate_backbone_forwards()
    print("Segmentation/detection lightweight validation passed")


if __name__ == "__main__":
    main()
