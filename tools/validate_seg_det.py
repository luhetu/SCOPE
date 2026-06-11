#!/usr/bin/env python3
"""Lightweight validation for segmentation and detection task wiring.

This intentionally avoids real ADE20K/COCO data and mmcv CUDA extensions. It
checks that all local seg/det YAML files build task configs and that the custom
backbones can run small CPU forward passes.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Config(dict):
    """Small stand-in for mmcv.Config used by the task builders."""

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


class MMDataParallel:
    def __init__(self, module, *args, **kwargs):
        self.module = module


def _noop(*args, **kwargs):
    return None


def _install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    parallel = types.ModuleType("mmcv.parallel")
    parallel.MMDataParallel = MMDataParallel
    mmcv.parallel = parallel
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.parallel"] = parallel

    mmseg = types.ModuleType("mmseg")
    mmseg_apis = types.ModuleType("mmseg.apis")
    mmseg_apis.set_random_seed = _noop
    mmseg_apis.single_gpu_test = lambda *args, **kwargs: []
    mmseg_apis.train_segmentor = _noop
    mmseg_datasets = types.ModuleType("mmseg.datasets")
    mmseg_datasets.build_dataloader = _noop
    mmseg_datasets.build_dataset = _noop
    mmseg_models_builder = types.ModuleType("mmseg.models.builder")
    mmseg_models_builder.BACKBONES = Registry()
    mmseg_models = types.ModuleType("mmseg.models")
    mmseg_models.build_segmentor = _noop
    mmseg_models.builder = mmseg_models_builder
    mmseg.apis = mmseg_apis
    mmseg.datasets = mmseg_datasets
    mmseg.models = mmseg_models
    sys.modules["mmseg"] = mmseg
    sys.modules["mmseg.apis"] = mmseg_apis
    sys.modules["mmseg.datasets"] = mmseg_datasets
    sys.modules["mmseg.models"] = mmseg_models
    sys.modules["mmseg.models.builder"] = mmseg_models_builder

    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = _noop
    mmdet_apis.train_detector = _noop
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = _noop
    mmdet_models_builder = types.ModuleType("mmdet.models.builder")
    mmdet_models_builder.BACKBONES = Registry()
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = _noop
    mmdet_models.builder = mmdet_models_builder
    mmdet.apis = mmdet_apis
    mmdet.datasets = mmdet_datasets
    mmdet.models = mmdet_models
    sys.modules["mmdet"] = mmdet
    sys.modules["mmdet.apis"] = mmdet_apis
    sys.modules["mmdet.datasets"] = mmdet_datasets
    sys.modules["mmdet.models"] = mmdet_models
    sys.modules["mmdet.models.builder"] = mmdet_models_builder


NONE_DEFAULTS = {
    "workers_per_gpu": None,
    "checkpoint_interval": None,
    "eval_interval": None,
    "log_interval": None,
    "final_eval": None,
    "warmup_epochs": None,
    "warmup_iters": None,
    "weight_decay": None,
    "drop_path_rate": None,
    "dim_head": None,
    "out_indices": None,
    "use_cls_token": None,
    "min_pretrained_match_rate": None,
    "seg_head_dim": None,
    "seg_aux_dim": None,
    "seg_aux_in_index": None,
    "seg_neck_dim": None,
    "seg_norm_type": None,
    "seg_neck_style": None,
    "backbone_size": None,
    "test_img_scale": None,
    "img_scale": None,
    "det_neck_type": None,
}


def _load_args(path: Path) -> argparse.Namespace:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    values = {
        "cfg": str(path),
        "resume": "",
        "data_dir": None,
        "time_profile": False,
        "time_profile_interval": 1000,
        **NONE_DEFAULTS,
        **cfg,
    }
    return argparse.Namespace(**values)


def _new_task(task_cls, args):
    task = object.__new__(task_cls)
    task.args = args
    task.run_name = task._build_run_name()
    return task


def validate_segmentation_configs():
    from tasks.segmentation import SegmentationTask

    paths = sorted((ROOT / "configs").glob("seg_*.yaml"))
    if not paths:
        raise AssertionError("No segmentation configs found")

    for path in paths:
        args = _load_args(path)
        task = _new_task(SegmentationTask, args)
        cfg = task._build_mmseg_config()
        assert cfg.model["type"] == "EncoderDecoder", path
        assert cfg.model["decode_head"]["num_classes"] == 150, path
        assert cfg.data["train"]["type"] == "ADE20KDataset", path
        assert isinstance(cfg.data["workers_per_gpu"], int), path
    print(f"Validated {len(paths)} segmentation configs")


def validate_detection_configs():
    from tasks.detection import DetectionTask

    paths = sorted((ROOT / "configs").glob("detection_*.yaml"))
    if not paths:
        raise AssertionError("No detection configs found")

    for path in paths:
        args = _load_args(path)
        task = _new_task(DetectionTask, args)
        cfg = task._build_mmdet_config()
        assert cfg.model["type"] == "MaskRCNN", path
        assert cfg.model["roi_head"]["bbox_head"]["num_classes"] == 80, path
        assert cfg.data["train"]["type"] == "CocoDataset", path
        assert isinstance(cfg.data["workers_per_gpu"], int), path
    print(f"Validated {len(paths)} detection configs")


def validate_backbone_forward():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    expected_shapes = [
        (1, 32, 8, 8),
        (1, 32, 4, 4),
        (1, 32, 2, 2),
        (1, 32, 1, 1),
    ]
    common = dict(
        image_size=32,
        patch_size=16,
        dim=32,
        depth=4,
        heads=4,
        mlp_dim=64,
        dim_head=8,
        out_indices=(0, 1, 2, 3),
    )
    cases = []
    for style in ("resize", "simple_fpn"):
        cases.append(("vit", ViTBackbone, {**common, "fpn_adapter_style": style}))
        for use_cls_token in (False, True):
            cases.append((
                f"vitcope_cls_{use_cls_token}",
                ViTCoPEBackbone,
                {**common, "fpn_adapter_style": style, "use_cls_token": use_cls_token},
            ))
            cases.append((
                f"vitscope_cls_{use_cls_token}",
                ViTSCoPEBackbone,
                {**common, "fpn_adapter_style": style, "use_cls_token": use_cls_token},
            ))

    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        for name, cls, kwargs in cases:
            model = cls(**kwargs).eval()
            outputs = model(image)
            shapes = [tuple(out.shape) for out in outputs]
            assert shapes == expected_shapes, f"{name}: {shapes}"
    print(f"Validated {len(cases)} custom backbone CPU forward cases")


def main():
    _install_openmmlab_stubs()
    validate_segmentation_configs()
    validate_detection_configs()
    validate_backbone_forward()
    print("Segmentation and detection smoke validation passed")


if __name__ == "__main__":
    main()
