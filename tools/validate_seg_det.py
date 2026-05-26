#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight validation for segmentation and detection wiring.

The full training stack needs mmcv-full CUDA extensions. This smoke test stubs
the OpenMMLab entry points so config construction and custom backbone forwards
can be checked in a plain CPU Python environment.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Config(dict):
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


class DummyDataset:
    CLASSES = ("dummy",)
    PALETTE = [(0, 0, 0)]

    def evaluate(self, *args, **kwargs):
        return {"metric": 0.0}


class DummyModel(torch.nn.Module):
    CLASSES = ()

    def cuda(self, *args, **kwargs):
        return self


def _install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = lambda model, device_ids=None: model

    mmseg = types.ModuleType("mmseg")
    mmseg_apis = types.ModuleType("mmseg.apis")
    mmseg_apis.set_random_seed = lambda *args, **kwargs: None
    mmseg_apis.single_gpu_test = lambda *args, **kwargs: []
    mmseg_apis.train_segmentor = lambda *args, **kwargs: None
    mmseg_datasets = types.ModuleType("mmseg.datasets")
    mmseg_datasets.build_dataset = lambda *args, **kwargs: DummyDataset()
    mmseg_datasets.build_dataloader = lambda *args, **kwargs: []
    mmseg_models = types.ModuleType("mmseg.models")
    mmseg_models.build_segmentor = lambda *args, **kwargs: DummyModel()
    mmseg_models_builder = types.ModuleType("mmseg.models.builder")
    mmseg_models_builder.BACKBONES = Registry()

    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = lambda *args, **kwargs: None
    mmdet_apis.train_detector = lambda *args, **kwargs: None
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = lambda *args, **kwargs: DummyDataset()
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = lambda *args, **kwargs: DummyModel()
    mmdet_models_builder = types.ModuleType("mmdet.models.builder")
    mmdet_models_builder.BACKBONES = Registry()

    stubs = {
        "mmcv": mmcv,
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
    sys.modules.update(stubs)


def _load_cfg(cfg_path: Path):
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
        sys.argv = ["validate_seg_det.py", "--cfg", str(cfg_path)]
        args = load_cfg(parser)
    finally:
        sys.argv = old_argv

    args.nowandb = True
    return args


def _validate_segmentation_configs():
    from tasks.segmentation import SegmentationTask

    paths = sorted((REPO_ROOT / "configs").glob("seg_*.yaml"))
    assert paths, "No segmentation configs found"
    for cfg_path in paths:
        args = _load_cfg(cfg_path)
        assert args.task == "seg", f"{cfg_path} is not a segmentation config"
        assert getattr(args, "size", None) is not None, f"{cfg_path} missing size"
        assert getattr(args, "patch", None) is not None, f"{cfg_path} missing patch"

        task = SegmentationTask.__new__(SegmentationTask)
        task.args = args
        task.device = "cpu"
        task.run_name = task._build_run_name()
        cfg = task._build_mmseg_config()

        assert cfg.model["type"] == "EncoderDecoder"
        assert cfg.data["workers_per_gpu"] == 4
        assert cfg.data["samples_per_gpu"] == args.bs
        assert len(cfg.model["decode_head"]["in_channels"]) == 4
    print(f"validated {len(paths)} segmentation configs")


def _validate_detection_configs():
    from tasks.detection import DetectionTask

    paths = sorted((REPO_ROOT / "configs").glob("detection_*.yaml"))
    assert paths, "No detection configs found"
    for cfg_path in paths:
        args = _load_cfg(cfg_path)
        assert args.task == "det", f"{cfg_path} is not a detection config"
        assert getattr(args, "size", None) is not None, f"{cfg_path} missing size"
        assert getattr(args, "patch", None) is not None, f"{cfg_path} missing patch"

        task = DetectionTask.__new__(DetectionTask)
        task.args = args
        task.device = "cpu"
        task.run_name = task._build_run_name()
        cfg = task._build_mmdet_config()

        assert cfg.model["type"] == "MaskRCNN"
        assert cfg.data["workers_per_gpu"] == 4
        assert cfg.data["samples_per_gpu"] == args.bs
        assert len(cfg.model["neck"]["in_channels"]) == 4
    print(f"validated {len(paths)} detection configs")


def _assert_feature_shapes(name, outputs):
    expected_hw = [(8, 8), (4, 4), (2, 2), (1, 1)]
    assert len(outputs) == 4, f"{name}: expected 4 outputs, got {len(outputs)}"
    for idx, (output, hw) in enumerate(zip(outputs, expected_hw)):
        assert tuple(output.shape[:2]) == (1, 32), f"{name}[{idx}] bad batch/channels: {tuple(output.shape)}"
        assert tuple(output.shape[-2:]) == hw, f"{name}[{idx}] bad spatial shape: {tuple(output.shape)}"


def _validate_backbone_forwards():
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    torch.manual_seed(0)
    image = torch.randn(1, 3, 32, 32)
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
        cases.append((f"ViTBackbone/{style}", ViTBackbone(fpn_adapter_style=style, **common)))
        for use_cls_token in (False, True):
            cases.append((
                f"ViTCoPEBackbone/{style}/cls={use_cls_token}",
                ViTCoPEBackbone(fpn_adapter_style=style, use_cls_token=use_cls_token, **common),
            ))
            cases.append((
                f"ViTSCoPEBackbone/{style}/cls={use_cls_token}",
                ViTSCoPEBackbone(fpn_adapter_style=style, use_cls_token=use_cls_token, **common),
            ))

    for name, model in cases:
        model.eval()
        with torch.no_grad():
            _assert_feature_shapes(name, model(image))
    print(f"validated {len(cases)} backbone forward cases")


def main():
    _install_openmmlab_stubs()
    _validate_segmentation_configs()
    _validate_detection_configs()
    _validate_backbone_forwards()
    print("segmentation/detection validation passed")


if __name__ == "__main__":
    main()
