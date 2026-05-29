#!/usr/bin/env python3
"""Lightweight validation for segmentation and detection wiring.

The full training entry points require datasets and mmcv-full CUDA/C++ ops.
This script stubs the OpenMMLab registries/APIs, then validates that all
segmentation/detection YAML files build task configs and that the custom
backbones can execute a small CPU forward pass.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import types
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Config(dict):
    """Small subset of mmcv.Config used by the task config builders."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class Registry:
    def __init__(self, name):
        self.name = name
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        def _register(cls):
            key = name or cls.__name__
            if not force and key in self.module_dict:
                raise KeyError(f"{key} is already registered in {self.name}")
            self.module_dict[key] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


def _install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.__version__ = "1.3.17"
    mmcv.Config = Config
    sys.modules["mmcv"] = mmcv

    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object
    sys.modules["mmcv.parallel"] = mmcv_parallel

    def noop(*args, **kwargs):
        return None

    for root_name, build_name, train_name in (
        ("mmseg", "build_segmentor", "train_segmentor"),
        ("mmdet", "build_detector", "train_detector"),
    ):
        root = types.ModuleType(root_name)
        sys.modules[root_name] = root

        apis = types.ModuleType(f"{root_name}.apis")
        apis.set_random_seed = noop
        setattr(apis, train_name, noop)
        if root_name == "mmseg":
            apis.single_gpu_test = lambda *args, **kwargs: []
        sys.modules[f"{root_name}.apis"] = apis

        datasets = types.ModuleType(f"{root_name}.datasets")
        datasets.build_dataset = noop
        if root_name == "mmseg":
            datasets.build_dataloader = noop
        sys.modules[f"{root_name}.datasets"] = datasets

        models = types.ModuleType(f"{root_name}.models")
        setattr(models, build_name, noop)
        sys.modules[f"{root_name}.models"] = models

        builder = types.ModuleType(f"{root_name}.models.builder")
        builder.BACKBONES = Registry(f"{root_name}_backbone")
        sys.modules[f"{root_name}.models.builder"] = builder


def _load_args(cfg_path: Path) -> argparse.Namespace:
    args = argparse.Namespace(
        cfg=str(cfg_path),
        resume="",
        workers_per_gpu=None,
        model=None,
        data_dir=None,
        time_profile=False,
        time_profile_interval=1000,
    )
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    for key, value in cfg.items():
        if key in {"lr", "min_lr"}:
            value = float(value)
        elif key in {
            "bs",
            "size",
            "n_epochs",
            "max_iters",
            "warmup_iters",
            "checkpoint_interval",
            "eval_interval",
            "log_interval",
            "patch",
            "dim",
            "depth",
            "heads",
            "mlp_dim",
            "dim_head",
            "seg_head_dim",
            "seg_aux_dim",
            "seg_neck_dim",
        }:
            value = int(value)
        elif key in {
            "warmup_epochs",
            "drop_path_rate",
            "weight_decay",
            "dropout",
            "emb_dropout",
            "layer_decay_rate",
        }:
            value = float(value)
        elif key in {"amp", "aug", "nowandb", "use_cls_token"}:
            value = bool(value)
        elif key == "pretrained" and value == "null":
            value = None
        setattr(args, key, value)
    return args


def _build_task_config(task_cls, args, builder_name):
    task = object.__new__(task_cls)
    task.args = args
    task.device = "cpu"
    task.run_name = task._build_run_name()
    with contextlib.redirect_stdout(io.StringIO()):
        return getattr(task, builder_name)()


def validate_configs():
    _install_openmmlab_stubs()

    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    seg_paths = sorted((REPO_ROOT / "configs").glob("seg_*.yaml"))
    det_paths = sorted((REPO_ROOT / "configs").glob("detection_*.yaml"))
    if not seg_paths:
        raise AssertionError("No segmentation configs found")
    if not det_paths:
        raise AssertionError("No detection configs found")

    for cfg_path in seg_paths:
        args = _load_args(cfg_path)
        cfg = _build_task_config(SegmentationTask, args, "_build_mmseg_config")
        assert cfg.model["type"] == "EncoderDecoder", cfg_path
        assert cfg.data["workers_per_gpu"] == 4, cfg_path
        assert cfg.model["backbone"]["type"], cfg_path

    for cfg_path in det_paths:
        args = _load_args(cfg_path)
        cfg = _build_task_config(DetectionTask, args, "_build_mmdet_config")
        assert cfg.model["type"] == "MaskRCNN", cfg_path
        assert cfg.data["workers_per_gpu"] == 4, cfg_path
        assert cfg.model["backbone"]["type"], cfg_path

    return len(seg_paths), len(det_paths)


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
    expected_shapes = ((1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1))
    cases = []
    for style in ("resize", "simple_fpn"):
        cases.append(("vit", ViTBackbone, dict(fpn_adapter_style=style)))
        for use_cls_token in (False, True):
            cases.append(("vitcope", ViTCoPEBackbone, dict(fpn_adapter_style=style, use_cls_token=use_cls_token)))
            cases.append(("vitscope", ViTSCoPEBackbone, dict(fpn_adapter_style=style, use_cls_token=use_cls_token)))

    image = torch.randn(1, 3, 32, 32)
    for name, cls, extra in cases:
        model = cls(**common, **extra).eval()
        with torch.no_grad():
            outputs = model(image)
        shapes = tuple(tuple(out.shape) for out in outputs)
        if shapes != expected_shapes:
            raise AssertionError(f"{name} {extra} produced {shapes}, expected {expected_shapes}")

    return len(cases)


def main():
    seg_count, det_count = validate_configs()
    backbone_count = validate_backbones()
    print(f"Validated {seg_count} segmentation configs")
    print(f"Validated {det_count} detection configs")
    print(f"Validated {backbone_count} custom backbone forward cases")


if __name__ == "__main__":
    main()
