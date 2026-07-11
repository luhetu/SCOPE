#!/usr/bin/env python3
"""Validate segmentation/detection config glue and custom dense backbones.

This script intentionally stubs the small OpenMMLab surface needed by the task
builders so it can run in lightweight environments without mmcv-full.
"""

from __future__ import annotations

import argparse
import glob
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _Registry:
    def __init__(self):
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        def _register(cls):
            key = name or cls.__name__
            if key in self.module_dict and not force:
                raise KeyError(f"{key} is already registered")
            self.module_dict[key] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


def _install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = _Config
    mmcv._ext = types.SimpleNamespace()
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object

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
    mmseg_builder.BACKBONES = _Registry()

    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = lambda *args, **kwargs: None
    mmdet_apis.train_detector = lambda *args, **kwargs: None
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = lambda *args, **kwargs: None
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = lambda *args, **kwargs: None
    mmdet_builder = types.ModuleType("mmdet.models.builder")
    mmdet_builder.BACKBONES = _Registry()

    mmcv.parallel = mmcv_parallel
    mmseg.apis = mmseg_apis
    mmseg.datasets = mmseg_datasets
    mmseg.models = mmseg_models
    mmseg_models.builder = mmseg_builder
    mmdet.apis = mmdet_apis
    mmdet.datasets = mmdet_datasets
    mmdet.models = mmdet_models
    mmdet_models.builder = mmdet_builder

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


def _load_args(cfg_path: Path):
    from utils.cfg import load_cfg

    parser = argparse.ArgumentParser(description="Validate one SCOPE dense config")
    parser.add_argument("--cfg", type=str, default=str(cfg_path))
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--workers_per_gpu", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--time_profile", action="store_true")
    parser.add_argument("--time_profile_interval", type=int, default=1000)

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], "--cfg", str(cfg_path)]
        return load_cfg(parser)
    finally:
        sys.argv = old_argv


def _assert_required_args(args, cfg_path: Path):
    required = ["task", "model", "data_dir", "bs", "n_epochs", "lr", "size", "patch"]
    if args.model == "swin":
        required += ["embed_dim", "depths", "num_heads", "window_size"]
    else:
        required += ["dim", "depth", "heads", "mlp_dim"]

    missing = [name for name in required if getattr(args, name, None) is None]
    if missing:
        raise AssertionError(f"{cfg_path} is missing required fields: {missing}")


def _validate_config_builders():
    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    seg_paths = sorted(Path(path) for path in glob.glob(str(REPO_ROOT / "configs" / "seg_*.yaml")))
    det_paths = sorted(Path(path) for path in glob.glob(str(REPO_ROOT / "configs" / "detection_*.yaml")))
    if not seg_paths:
        raise AssertionError("No segmentation configs found")
    if not det_paths:
        raise AssertionError("No detection configs found")

    for cfg_path in seg_paths:
        args = _load_args(cfg_path)
        _assert_required_args(args, cfg_path)
        task = SegmentationTask.__new__(SegmentationTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmseg_config()
        assert cfg.model["type"] == "EncoderDecoder"
        assert cfg.data["workers_per_gpu"] == 4

    for cfg_path in det_paths:
        args = _load_args(cfg_path)
        _assert_required_args(args, cfg_path)
        task = DetectionTask.__new__(DetectionTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmdet_config()
        assert cfg.model["type"] == "MaskRCNN"
        assert cfg.data["workers_per_gpu"] == 4

    print(f"Validated {len(seg_paths)} segmentation configs and {len(det_paths)} detection configs")


def _validate_custom_backbones():
    try:
        import torch
        from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone
    except ImportError as exc:
        raise RuntimeError(
            "Backbone smoke tests require torch, timm, and einops. "
            "Install the project requirements before running this validator."
        ) from exc

    torch.set_num_threads(1)
    expected_shapes = [(1, 24, 8, 8), (1, 24, 4, 4), (1, 24, 2, 2), (1, 24, 1, 1)]
    cases = [
        (ViTBackbone, {}),
        (ViTCoPEBackbone, {"use_cls_token": False}),
        (ViTCoPEBackbone, {"use_cls_token": True}),
        (ViTSCoPEBackbone, {"use_cls_token": False}),
        (ViTSCoPEBackbone, {"use_cls_token": True}),
    ]

    image = torch.randn(1, 3, 32, 32)
    for adapter_style in ("resize", "simple_fpn"):
        for cls, extra_kwargs in cases:
            model = cls(
                image_size=32,
                patch_size=16,
                dim=24,
                depth=4,
                heads=3,
                mlp_dim=48,
                dim_head=8,
                out_indices=(0, 1, 2, 3),
                fpn_adapter_style=adapter_style,
                **extra_kwargs,
            )
            model.eval()
            with torch.no_grad():
                outputs = model(image)
            shapes = [tuple(output.shape) for output in outputs]
            if shapes != expected_shapes:
                raise AssertionError(f"{cls.__name__} ({adapter_style}) shapes {shapes} != {expected_shapes}")

    print("Validated custom ViT/CoPE/SCoPE dense backbone smoke tests")


def main():
    _install_openmmlab_stubs()
    _validate_config_builders()
    _validate_custom_backbones()
    print("Segmentation and detection validation passed")


if __name__ == "__main__":
    main()
