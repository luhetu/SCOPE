#!/usr/bin/env python3
"""Lightweight smoke checks for segmentation and detection configuration paths."""

from __future__ import annotations

import argparse
import sys
import tempfile
import types
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Config(dict):
    """Small mmcv.Config stand-in with attribute access."""

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

        return _register(module) if module is not None else _register


def _install_openmmlab_stubs():
    """Install enough OpenMMLab stubs to import task modules without mmcv-full."""

    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object
    mmcv.parallel = mmcv_parallel

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
    mmseg.apis = mmseg_apis
    mmseg.datasets = mmseg_datasets
    mmseg.models = mmseg_models

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
    mmdet.apis = mmdet_apis
    mmdet.datasets = mmdet_datasets
    mmdet.models = mmdet_models

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


def _parser():
    parser = argparse.ArgumentParser(description="Validate one SCOPE config")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--workers_per_gpu", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--time_profile", action="store_true")
    parser.add_argument("--time_profile_interval", type=int, default=1000)
    return parser


def _load_args(cfg_path: Path):
    from utils.cfg import load_cfg

    old_argv = sys.argv[:]
    try:
        sys.argv = ["validate_seg_det.py", "--cfg", str(cfg_path)]
        return load_cfg(_parser())
    finally:
        sys.argv = old_argv


def _validate_null_preservation():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "nulls.yaml"
        cfg_path.write_text(
            "\n".join(
                [
                    "task: seg",
                    "model: vit",
                    "lr: 1e-4",
                    "max_iters: null",
                    "warmup_iters: null",
                    "workers_per_gpu: null",
                    "pretrained: null",
                ]
            ),
            encoding="utf-8",
        )
        args = _load_args(cfg_path)
    for name in ("max_iters", "warmup_iters", "workers_per_gpu", "pretrained"):
        if getattr(args, name) is not None:
            raise AssertionError(f"Expected {name} to stay None")


def _build_task_config(task_cls, args, build_method):
    task = task_cls.__new__(task_cls)
    task.args = args
    task.run_name = task._build_run_name()
    return getattr(task, build_method)()


def _validate_segmentation_configs(paths):
    from tasks.segmentation import SegmentationTask

    for path in paths:
        args = _load_args(path)
        cfg = _build_task_config(SegmentationTask, args, "_build_mmseg_config")
        if cfg.data["workers_per_gpu"] != 4:
            raise AssertionError(f"{path}: default workers_per_gpu should be 4")
        if cfg.model["type"] != "EncoderDecoder":
            raise AssertionError(f"{path}: expected EncoderDecoder config")


def _validate_detection_configs(paths):
    from tasks.detection import DetectionTask

    for path in paths:
        args = _load_args(path)
        cfg = _build_task_config(DetectionTask, args, "_build_mmdet_config")
        if cfg.data["workers_per_gpu"] != 4:
            raise AssertionError(f"{path}: default workers_per_gpu should be 4")
        if cfg.model["type"] != "MaskRCNN":
            raise AssertionError(f"{path}: expected MaskRCNN config")


def _validate_backbone_forwards():
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    expected_shapes = [(1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1)]
    cases = [
        (ViTBackbone, {}),
        (ViTCoPEBackbone, {"use_cls_token": False}),
        (ViTCoPEBackbone, {"use_cls_token": True}),
        (ViTSCoPEBackbone, {"use_cls_token": False}),
        (ViTSCoPEBackbone, {"use_cls_token": True}),
    ]
    count = 0
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
            )
            model.eval()
            with torch.no_grad():
                outputs = model(torch.randn(1, 3, 32, 32))
            shapes = [tuple(output.shape) for output in outputs]
            if shapes != expected_shapes:
                raise AssertionError(f"{cls.__name__} {adapter_style}: got {shapes}")
            count += 1
    return count


def main():
    _install_openmmlab_stubs()
    _validate_null_preservation()

    seg_configs = sorted((REPO_ROOT / "configs").glob("seg_*.yaml"))
    det_configs = sorted((REPO_ROOT / "configs").glob("detection_*.yaml"))
    if not seg_configs:
        raise AssertionError("No segmentation configs found")
    if not det_configs:
        raise AssertionError("No detection configs found")

    _validate_segmentation_configs(seg_configs)
    _validate_detection_configs(det_configs)
    backbone_cases = _validate_backbone_forwards()

    print(f"Validated {len(seg_configs)} segmentation configs")
    print(f"Validated {len(det_configs)} detection configs")
    print(f"Validated {backbone_cases} backbone CPU forward cases")


if __name__ == "__main__":
    main()
