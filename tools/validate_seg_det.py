#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight validation for SCOPE segmentation and detection code.

The full training paths need mmcv-full CUDA extensions plus ADE20K/COCO data.
This script stubs the OpenMMLab entry points that are not needed for config
construction, then verifies:
  1. every configs/seg_*.yaml builds an MMSeg config;
  2. every configs/detection_*.yaml builds an MMDet config;
  3. custom ViT/CoPE/SCoPE backbones run CPU forward smoke tests.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Config(dict):
    """Minimal attribute-access dict compatible with the task config builders."""

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

    def register_module(self, name=None, module=None, force=False, **kwargs):
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


def _install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv._ext = types.ModuleType("mmcv._ext")
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object

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
    mmseg_models_builder = types.ModuleType("mmseg.models.builder")
    mmseg_models_builder.BACKBONES = Registry()
    mmseg_models.builder = mmseg_models_builder

    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = _noop
    mmdet_apis.train_detector = _noop
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = _noop
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = _noop
    mmdet_models_builder = types.ModuleType("mmdet.models.builder")
    mmdet_models_builder.BACKBONES = Registry()
    mmdet_models.builder = mmdet_models_builder

    modules = {
        "mmcv": mmcv,
        "mmcv._ext": mmcv._ext,
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


def _load_training_args(config_path: Path):
    from utils.cfg import load_cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--workers_per_gpu", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--time_profile", action="store_true")
    parser.add_argument("--time_profile_interval", type=int, default=1000)

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], "--cfg", str(config_path)]
        args = load_cfg(parser)
    finally:
        sys.argv = old_argv

    args.run_tag = "validate"
    return args


def _validate_seg_configs(configs):
    from tasks.segmentation import SegmentationTask

    for cfg_path in configs:
        args = _load_training_args(cfg_path)
        task = SegmentationTask.__new__(SegmentationTask)
        task.args = args
        task.run_name = cfg_path.stem
        cfg = task._build_mmseg_config()

        assert cfg.model["type"] == "EncoderDecoder"
        assert cfg.model["decode_head"]["num_classes"] == 150
        assert cfg.data["workers_per_gpu"] > 0
        assert cfg.data["train"]["type"] == "ADE20KDataset"
        print(f"[seg] ok: {cfg_path.relative_to(ROOT)}")


def _validate_det_configs(configs):
    from tasks.detection import DetectionTask

    for cfg_path in configs:
        args = _load_training_args(cfg_path)
        task = DetectionTask.__new__(DetectionTask)
        task.args = args
        task.run_name = cfg_path.stem
        cfg = task._build_mmdet_config()

        assert cfg.model["type"] == "MaskRCNN"
        assert cfg.model["roi_head"]["bbox_head"]["num_classes"] == 80
        assert cfg.data["workers_per_gpu"] > 0
        assert cfg.data["train"]["type"] == "CocoDataset"
        print(f"[det] ok: {cfg_path.relative_to(ROOT)}")


def _validate_backbones():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    expected_shapes = [(1, 64, 8, 8), (1, 64, 4, 4), (1, 64, 2, 2), (1, 64, 1, 1)]
    cases = [
        ("vit", ViTBackbone, {}),
        ("vitcope", ViTCoPEBackbone, {"use_cls_token": False}),
        ("vitcope_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("vitscope", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("vitscope_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]

    for adapter_style in ("resize", "simple_fpn"):
        for name, cls, extra_kwargs in cases:
            torch.manual_seed(0)
            model = cls(
                image_size=32,
                patch_size=16,
                dim=64,
                depth=4,
                heads=4,
                mlp_dim=128,
                dim_head=16,
                out_indices=(0, 1, 2, 3),
                fpn_adapter_style=adapter_style,
                **extra_kwargs,
            ).eval()
            with torch.no_grad():
                outputs = model(torch.randn(1, 3, 32, 32))
            shapes = [tuple(out.shape) for out in outputs]
            assert shapes == expected_shapes, f"{name}/{adapter_style}: {shapes}"
            print(f"[backbone] ok: {name} ({adapter_style})")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-backbone", action="store_true", help="skip CPU backbone smoke tests")
    return parser.parse_args()


def main():
    args = parse_args()
    _install_openmmlab_stubs()

    seg_configs = sorted((ROOT / "configs").glob("seg_*.yaml"))
    det_configs = sorted((ROOT / "configs").glob("detection_*.yaml"))
    if not seg_configs:
        raise RuntimeError("No segmentation configs found")
    if not det_configs:
        raise RuntimeError("No detection configs found")

    _validate_seg_configs(seg_configs)
    _validate_det_configs(det_configs)
    if not args.skip_backbone:
        _validate_backbones()

    print(
        f"Validated {len(seg_configs)} segmentation configs, "
        f"{len(det_configs)} detection configs, and custom backbones."
    )


if __name__ == "__main__":
    main()
