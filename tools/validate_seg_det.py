#!/usr/bin/env python3
"""Lightweight validation for segmentation and detection config builders."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class Config(AttrDict):
    pass


class Registry:
    def __init__(self):
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        def _register(cls):
            key = name or cls.__name__
            if not force and key in self.module_dict and self.module_dict[key] is not cls:
                raise KeyError(f"{key} is already registered")
            self.module_dict[key] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


MMDET_BACKBONES = Registry()
MMSEG_BACKBONES = Registry()


def _stub_modules():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.parallel"] = mmcv_parallel

    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = lambda *args, **kwargs: None
    mmdet_apis.train_detector = lambda *args, **kwargs: None
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = lambda *args, **kwargs: None
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = lambda *args, **kwargs: None
    mmdet_builder = types.ModuleType("mmdet.models.builder")
    mmdet_builder.BACKBONES = MMDET_BACKBONES
    sys.modules["mmdet"] = mmdet
    sys.modules["mmdet.apis"] = mmdet_apis
    sys.modules["mmdet.datasets"] = mmdet_datasets
    sys.modules["mmdet.models"] = mmdet_models
    sys.modules["mmdet.models.builder"] = mmdet_builder

    mmseg = types.ModuleType("mmseg")
    mmseg_apis = types.ModuleType("mmseg.apis")
    mmseg_apis.set_random_seed = lambda *args, **kwargs: None
    mmseg_apis.single_gpu_test = lambda *args, **kwargs: None
    mmseg_apis.train_segmentor = lambda *args, **kwargs: None
    mmseg_datasets = types.ModuleType("mmseg.datasets")
    mmseg_datasets.build_dataloader = lambda *args, **kwargs: None
    mmseg_datasets.build_dataset = lambda *args, **kwargs: None
    mmseg_models = types.ModuleType("mmseg.models")
    mmseg_models.build_segmentor = lambda *args, **kwargs: None
    mmseg_builder = types.ModuleType("mmseg.models.builder")
    mmseg_builder.BACKBONES = MMSEG_BACKBONES
    sys.modules["mmseg"] = mmseg
    sys.modules["mmseg.apis"] = mmseg_apis
    sys.modules["mmseg.datasets"] = mmseg_datasets
    sys.modules["mmseg.models"] = mmseg_models
    sys.modules["mmseg.models.builder"] = mmseg_builder


def _make_parser():
    parser = argparse.ArgumentParser(description="Validate SCOPE seg/det configs")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--workers_per_gpu", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--time_profile", action="store_true")
    parser.add_argument("--time_profile_interval", type=int, default=1000)
    return parser


OPTIONAL_NONE_DEFAULTS = (
    "backbone_size",
    "betas",
    "checkpoint_interval",
    "dim_head",
    "drop_path_rate",
    "eval_interval",
    "final_eval",
    "layer_decay_rate",
    "log_interval",
    "max_iters",
    "out_indices",
    "seg_aux_dim",
    "seg_aux_in_index",
    "seg_head_dim",
    "seg_neck_dim",
    "seg_neck_style",
    "seg_norm_type",
    "use_cls_token",
    "warmup_iters",
)


def _load_args(cfg_path):
    from utils.cfg import load_cfg

    parser = _make_parser()
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], "--cfg", str(cfg_path)]
        args = load_cfg(parser)
    finally:
        sys.argv = old_argv

    for name in OPTIONAL_NONE_DEFAULTS:
        if not hasattr(args, name):
            setattr(args, name, None)
    return args


def _build_task_config(task_cls, method_name, cfg_path):
    args = _load_args(cfg_path)
    task = object.__new__(task_cls)
    task.args = args
    task.run_name = task._build_run_name()
    cfg = getattr(task, method_name)()
    assert cfg.model["backbone"]["type"], cfg_path
    assert isinstance(cfg.data["workers_per_gpu"], int), cfg_path
    return args, cfg


def _validate_config_builders():
    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    seg_paths = sorted((ROOT / "configs").glob("seg_*.yaml"))
    det_paths = sorted((ROOT / "configs").glob("detection_*.yaml"))
    if not seg_paths or not det_paths:
        raise AssertionError("Expected segmentation and detection configs")

    for path in seg_paths:
        args, cfg = _build_task_config(SegmentationTask, "_build_mmseg_config", path)
        if args.model == "swin":
            assert cfg.model["backbone"]["type"] == "SwinTransformer", path
        else:
            assert cfg.model["backbone"]["image_size"], path

    for path in det_paths:
        args, cfg = _build_task_config(DetectionTask, "_build_mmdet_config", path)
        if args.model == "swin":
            assert cfg.model["backbone"]["type"] == "SwinTransformer", path
        else:
            assert cfg.model["backbone"]["image_size"], path

    print(f"Validated {len(seg_paths)} segmentation configs and {len(det_paths)} detection configs.")


def _validate_backbones():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    cases = [
        ("vit", ViTBackbone, {}),
        ("vitcope_no_cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("vitcope_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("vitscope_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
        ("vitscope_no_cls", ViTSCoPEBackbone, {"use_cls_token": False}),
    ]
    expected_shapes = [(1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1)]

    for style in ("resize", "simple_fpn"):
        for name, cls, extra in cases:
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
                **extra,
            )
            model.eval()
            with torch.no_grad():
                outputs = model(torch.randn(1, 3, 32, 32))
            shapes = [tuple(out.shape) for out in outputs]
            assert shapes == expected_shapes, f"{name}/{style}: {shapes}"

    print(f"Validated {len(cases) * 2} custom backbone forward cases.")


def main():
    _stub_modules()
    _validate_config_builders()
    _validate_backbones()
    print("Segmentation and detection validation passed.")


if __name__ == "__main__":
    main()
