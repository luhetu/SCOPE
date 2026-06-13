#!/usr/bin/env python3
"""Lightweight validation for segmentation and detection configuration code.

The real training entry points require mmcv-full C++/CUDA extensions. This
script stubs the OpenMMLab surface needed by the local task builders so config
construction and custom backbone forward paths can be checked in a plain CPU
environment.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import types
from typing import Iterable

import torch
import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class Config(dict):
    """Small attribute-access dict compatible with the task config builders."""

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
        def decorator(cls):
            key = name or cls.__name__
            if not force and key in self.module_dict:
                return cls
            self.module_dict[key] = cls
            return cls

        if module is not None:
            return decorator(module)
        return decorator


def install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.parallel"] = mmcv_parallel

    for root in ("mmdet", "mmseg"):
        root_mod = types.ModuleType(root)
        apis_mod = types.ModuleType(f"{root}.apis")
        apis_mod.set_random_seed = lambda *args, **kwargs: None
        apis_mod.train_detector = lambda *args, **kwargs: None
        apis_mod.train_segmentor = lambda *args, **kwargs: None
        apis_mod.single_gpu_test = lambda *args, **kwargs: []

        datasets_mod = types.ModuleType(f"{root}.datasets")
        datasets_mod.build_dataset = lambda *args, **kwargs: types.SimpleNamespace(CLASSES=(), PALETTE=())
        datasets_mod.build_dataloader = lambda *args, **kwargs: []

        models_mod = types.ModuleType(f"{root}.models")
        models_mod.build_detector = lambda *args, **kwargs: types.SimpleNamespace(
            state_dict=lambda: {},
            load_state_dict=lambda *load_args, **load_kwargs: None,
        )
        models_mod.build_segmentor = models_mod.build_detector

        builder_mod = types.ModuleType(f"{root}.models.builder")
        builder_mod.BACKBONES = Registry()

        root_mod.apis = apis_mod
        root_mod.datasets = datasets_mod
        root_mod.models = models_mod
        models_mod.builder = builder_mod

        sys.modules[root] = root_mod
        sys.modules[f"{root}.apis"] = apis_mod
        sys.modules[f"{root}.datasets"] = datasets_mod
        sys.modules[f"{root}.models"] = models_mod
        sys.modules[f"{root}.models.builder"] = builder_mod


def load_args(path):
    defaults = dict(
        cfg=path,
        resume="",
        workers_per_gpu=None,
        model=None,
        data_dir=None,
        time_profile=False,
        time_profile_interval=1000,
    )
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    defaults.update(data)
    defaults["cfg"] = path
    return argparse.Namespace(**defaults)


def validate_config_builders() -> tuple[int, int]:
    install_openmmlab_stubs()
    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    seg_count = 0
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "seg_*.yaml"))):
        args = load_args(path)
        task = object.__new__(SegmentationTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmseg_config()
        assert cfg.model["type"] == "EncoderDecoder"
        assert "train" in cfg.data and "val" in cfg.data
        seg_count += 1

    det_count = 0
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "detection_*.yaml"))):
        args = load_args(path)
        task = object.__new__(DetectionTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmdet_config()
        assert cfg.model["type"] == "MaskRCNN"
        assert "train" in cfg.data and "val" in cfg.data
        det_count += 1

    return seg_count, det_count


def expected_feature_shapes(style: str) -> list[tuple[int, int]]:
    if style in ("resize", "simple_fpn"):
        return [(8, 8), (4, 4), (2, 2), (1, 1)]
    return [(2, 2)] * 4


def assert_backbone_outputs(name: str, model: torch.nn.Module, expected_hw: Iterable[tuple[int, int]]):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 32, 32))
    outputs = tuple(outputs)
    expected_hw = list(expected_hw)
    assert len(outputs) == len(expected_hw), f"{name}: expected {len(expected_hw)} outputs, got {len(outputs)}"
    for idx, (feat, (height, width)) in enumerate(zip(outputs, expected_hw)):
        assert tuple(feat.shape) == (2, 32, height, width), f"{name}[{idx}] shape {tuple(feat.shape)}"


def validate_backbones() -> int:
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

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
    cases = []
    for style in ("resize", "simple_fpn"):
        cases.extend(
            [
                (f"ViTBackbone/{style}", ViTBackbone(**base_kwargs, fpn_adapter_style=style), style),
                (f"ViTCoPEBackbone/no_cls/{style}", ViTCoPEBackbone(**base_kwargs, use_cls_token=False, fpn_adapter_style=style), style),
                (f"ViTCoPEBackbone/cls/{style}", ViTCoPEBackbone(**base_kwargs, use_cls_token=True, fpn_adapter_style=style), style),
                (f"ViTSCoPEBackbone/no_cls/{style}", ViTSCoPEBackbone(**base_kwargs, use_cls_token=False, fpn_adapter_style=style), style),
                (f"ViTSCoPEBackbone/cls/{style}", ViTSCoPEBackbone(**base_kwargs, use_cls_token=True, fpn_adapter_style=style), style),
            ]
        )

    for name, model, style in cases:
        assert_backbone_outputs(name, model, expected_feature_shapes(style))
    return len(cases)


def main():
    seg_count, det_count = validate_config_builders()
    backbone_count = validate_backbones()
    print(f"Validated {seg_count} segmentation configs")
    print(f"Validated {det_count} detection configs")
    print(f"Validated {backbone_count} custom backbone forward cases")


if __name__ == "__main__":
    main()
