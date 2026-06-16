#!/usr/bin/env python3
"""Smoke-test segmentation/detection config builders and custom backbones.

This validator intentionally avoids requiring mmcv-full's compiled extension.
It stubs the small OpenMMLab surface needed to import task modules, then checks
that every seg/det YAML can build its in-repo config and that custom dense
prediction backbones produce four feature maps.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Config(dict):
    """Minimal mmcv.Config replacement with attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class Registry:
    def __init__(self):
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        def _register(cls):
            module_name = name or cls.__name__
            if not force and module_name in self.module_dict:
                raise KeyError(f"{module_name} is already registered")
            self.module_dict[module_name] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


def _unavailable(*args, **kwargs):
    raise RuntimeError("This validator only exercises config construction.")


def install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv.__path__ = []

    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object

    mmseg = types.ModuleType("mmseg")
    mmseg.__path__ = []
    mmseg_apis = types.ModuleType("mmseg.apis")
    mmseg_apis.set_random_seed = lambda *args, **kwargs: None
    mmseg_apis.single_gpu_test = _unavailable
    mmseg_apis.train_segmentor = _unavailable
    mmseg_datasets = types.ModuleType("mmseg.datasets")
    mmseg_datasets.build_dataloader = _unavailable
    mmseg_datasets.build_dataset = _unavailable
    mmseg_models = types.ModuleType("mmseg.models")
    mmseg_models.build_segmentor = _unavailable
    mmseg_builder = types.ModuleType("mmseg.models.builder")
    mmseg_builder.BACKBONES = Registry()
    mmseg_models.builder = mmseg_builder

    mmdet = types.ModuleType("mmdet")
    mmdet.__path__ = []
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = lambda *args, **kwargs: None
    mmdet_apis.train_detector = _unavailable
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = _unavailable
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = _unavailable
    mmdet_builder = types.ModuleType("mmdet.models.builder")
    mmdet_builder.BACKBONES = Registry()
    mmdet_models.builder = mmdet_builder

    modules = {
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
    sys.modules.update(modules)


def make_train_parser():
    parser = argparse.ArgumentParser(description="seg/det validator")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--workers_per_gpu", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--time_profile", action="store_true")
    parser.add_argument("--time_profile_interval", type=int, default=1000)
    return parser


def load_args(config_path: Path):
    from utils.cfg import load_cfg

    old_argv = sys.argv[:]
    try:
        sys.argv = ["validate_seg_det.py", "--cfg", str(config_path)]
        return load_cfg(make_train_parser())
    finally:
        sys.argv = old_argv


def validate_segmentation_configs(config_paths):
    from tasks.segmentation import SegmentationTask

    checked = 0
    for config_path in config_paths:
        args = load_args(config_path)
        task = SegmentationTask.__new__(SegmentationTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmseg_config()
        assert cfg.model["type"] == "EncoderDecoder"
        assert cfg.model["decode_head"]["num_classes"] == 150
        assert cfg.data["workers_per_gpu"] >= 0
        checked += 1
    return checked


def validate_detection_configs(config_paths):
    from tasks.detection import DetectionTask

    checked = 0
    for config_path in config_paths:
        args = load_args(config_path)
        task = DetectionTask.__new__(DetectionTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmdet_config()
        assert cfg.model["type"] == "MaskRCNN"
        assert cfg.model["roi_head"]["bbox_head"]["num_classes"] == 80
        assert cfg.data["workers_per_gpu"] >= 0
        checked += 1
    return checked


def validate_backbones():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

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
    cases = [
        ("vit", ViTBackbone, {}),
        ("vitcope_no_cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("vitcope_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("vitscope_no_cls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("vitscope_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]
    expected_shapes = [(1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1)]

    checked = 0
    with torch.no_grad():
        x = torch.randn(1, 3, 32, 32)
        for style in ("resize", "simple_fpn"):
            for name, cls, extra in cases:
                model = cls(**common, fpn_adapter_style=style, **extra).eval()
                outputs = model(x)
                shapes = [tuple(out.shape) for out in outputs]
                if shapes != expected_shapes:
                    raise AssertionError(f"{name}/{style} produced {shapes}, expected {expected_shapes}")
                checked += 1
    return checked


def main():
    install_openmmlab_stubs()

    seg_configs = sorted((REPO_ROOT / "configs").glob("seg_*.yaml"))
    det_configs = sorted((REPO_ROOT / "configs").glob("detection_*.yaml"))
    if not seg_configs or not det_configs:
        raise RuntimeError("Expected both segmentation and detection configs.")

    seg_count = validate_segmentation_configs(seg_configs)
    det_count = validate_detection_configs(det_configs)
    backbone_count = validate_backbones()

    print("\nValidation passed")
    print(f"  segmentation configs: {seg_count}")
    print(f"  detection configs: {det_count}")
    print(f"  backbone forward cases: {backbone_count}")


if __name__ == "__main__":
    main()
