#!/usr/bin/env python3
"""Lightweight segmentation/detection validation for environments without mmcv-full.

The full training entrypoints require OpenMMLab compiled extensions and datasets.
This smoke test validates the Python-side task config builders for every seg/det
YAML file and runs CPU forward checks for the custom dense-prediction backbones.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
            if not force and key in self.module_dict:
                raise KeyError(f"{key} is already registered")
            self.module_dict[key] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


def _install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = _Config
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
    mmseg_builder.BACKBONES = _Registry()
    mmseg_models.builder = mmseg_builder
    mmseg.models = mmseg_models
    mmseg.datasets = mmseg_datasets
    mmseg.apis = mmseg_apis

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
    mmdet_models.builder = mmdet_builder
    mmdet.models = mmdet_models
    mmdet.datasets = mmdet_datasets
    mmdet.apis = mmdet_apis

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


def _build_train_parser():
    parser = argparse.ArgumentParser(description="seg/det validation parser")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--workers_per_gpu", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--time_profile", action="store_true")
    parser.add_argument("--time_profile_interval", type=int, default=1000)
    return parser


@contextmanager
def _argv(args):
    original = sys.argv[:]
    sys.argv = args
    try:
        yield
    finally:
        sys.argv = original


def _load_args(cfg_path):
    from utils.cfg import load_cfg

    parser = _build_train_parser()
    with _argv(["validate_seg_det.py", "--cfg", str(cfg_path)]):
        args = load_cfg(parser)
    _apply_train_schedule(args)
    return args


def _apply_train_schedule(args):
    if args.task == "seg":
        num_images = 20210
        iters_per_epoch = num_images // args.bs
        explicit_max_iters = getattr(args, "max_iters", None)
        if explicit_max_iters is not None:
            args.max_iters = int(explicit_max_iters)
        else:
            args.max_iters = int(args.n_epochs * iters_per_epoch)

        if getattr(args, "warmup_iters", None) is not None:
            args.warmup_iters = int(args.warmup_iters)
        elif getattr(args, "warmup_epochs", 0) > 0:
            args.warmup_iters = int(args.warmup_epochs * iters_per_epoch)
        else:
            args.warmup_iters = 1500
        args.iters_per_epoch = iters_per_epoch
    elif args.task == "det":
        num_images = 118287
        iters_per_epoch = num_images // args.bs
        if getattr(args, "warmup_iters", None) is not None:
            args.warmup_iters = int(args.warmup_iters)
        elif getattr(args, "warmup_epochs", 0) > 0:
            args.warmup_iters = int(args.warmup_epochs * iters_per_epoch)
        else:
            args.warmup_iters = 500
        args.iters_per_epoch = iters_per_epoch
    else:
        raise AssertionError(f"Unexpected task type: {args.task}")


def _task_without_init(task_cls, args):
    task = object.__new__(task_cls)
    task.args = args
    task.device = "cpu"
    task.run_name = task._build_run_name()
    return task


def _validate_task_configs():
    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    seg_paths = sorted(Path(ROOT, "configs").glob("seg*.yaml"))
    det_paths = sorted(Path(ROOT, "configs").glob("detection*.yaml"))
    assert seg_paths, "No segmentation configs found"
    assert det_paths, "No detection configs found"

    for cfg_path in seg_paths:
        args = _load_args(cfg_path)
        task = _task_without_init(SegmentationTask, args)
        cfg = task._build_mmseg_config()
        backbone = cfg.model["backbone"]
        assert cfg.model["type"] == "EncoderDecoder"
        assert cfg.data["workers_per_gpu"] == 4
        assert cfg.data["train"]["pipeline"], f"Empty train pipeline: {cfg_path}"
        if args.model == "swin":
            assert backbone["type"] == "SwinTransformer"
        else:
            assert backbone["type"] in {"ViTBackbone", "ViTCoPEBackbone", "ViTSCoPEBackbone"}

    for cfg_path in det_paths:
        args = _load_args(cfg_path)
        task = _task_without_init(DetectionTask, args)
        cfg = task._build_mmdet_config()
        backbone = cfg.model["backbone"]
        assert cfg.model["type"] == "MaskRCNN"
        assert cfg.data["workers_per_gpu"] == 4
        assert cfg.data["train"]["pipeline"], f"Empty train pipeline: {cfg_path}"
        if args.model == "swin":
            assert backbone["type"] == "SwinTransformer"
        else:
            assert backbone["type"] in {"ViTBackbone", "ViTCoPEBackbone", "ViTSCoPEBackbone"}

    print(f"Validated {len(seg_paths)} segmentation configs and {len(det_paths)} detection configs")


def _validate_backbone_forwards():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    torch.set_num_threads(1)
    x = torch.randn(2, 3, 32, 32)
    expected_shapes = [(2, 32, 8, 8), (2, 32, 4, 4), (2, 32, 2, 2), (2, 32, 1, 1)]
    base_kwargs = dict(
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
        ("vit-simple-fpn", ViTBackbone, {"fpn_adapter_style": "simple_fpn"}),
        ("vit-resize", ViTBackbone, {"fpn_adapter_style": "resize"}),
        ("vitcope-simple-fpn-nocls", ViTCoPEBackbone, {"fpn_adapter_style": "simple_fpn", "use_cls_token": False}),
        ("vitcope-resize-nocls", ViTCoPEBackbone, {"fpn_adapter_style": "resize", "use_cls_token": False}),
        ("vitcope-simple-fpn-cls", ViTCoPEBackbone, {"fpn_adapter_style": "simple_fpn", "use_cls_token": True}),
        ("vitcope-resize-cls", ViTCoPEBackbone, {"fpn_adapter_style": "resize", "use_cls_token": True}),
        ("vitscope-simple-fpn-cls", ViTSCoPEBackbone, {"fpn_adapter_style": "simple_fpn", "use_cls_token": True}),
        ("vitscope-resize-cls", ViTSCoPEBackbone, {"fpn_adapter_style": "resize", "use_cls_token": True}),
        ("vitscope-simple-fpn-nocls", ViTSCoPEBackbone, {"fpn_adapter_style": "simple_fpn", "use_cls_token": False}),
        ("vitscope-resize-nocls", ViTSCoPEBackbone, {"fpn_adapter_style": "resize", "use_cls_token": False}),
    ]

    with torch.no_grad():
        for name, cls, extra_kwargs in cases:
            model = cls(**base_kwargs, **extra_kwargs).eval()
            outputs = model(x)
            shapes = [tuple(output.shape) for output in outputs]
            assert shapes == expected_shapes, f"{name} shapes {shapes} != {expected_shapes}"

    print(f"Validated {len(cases)} custom backbone CPU forward cases")


def main():
    os.chdir(ROOT)
    _install_openmmlab_stubs()
    _validate_task_configs()
    _validate_backbone_forwards()
    print("Segmentation/detection validation passed")


if __name__ == "__main__":
    main()
