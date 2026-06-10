#!/usr/bin/env python3
"""Smoke-test segmentation/detection configs and custom dense backbones.

This intentionally stubs the small OpenMMLab API surface needed to build task
configs, so it can run in lightweight environments without mmcv-full C++ ops.
"""

import argparse
import glob
import os
import sys
import types


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class Config(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class Registry:
    def __init__(self):
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        def decorator(cls):
            self.module_dict[name or cls.__name__] = cls
            return cls

        if module is not None:
            self.module_dict[name or module.__name__] = module
            return module
        return decorator


def _install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    sys.modules["mmcv"] = mmcv

    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object
    sys.modules["mmcv.parallel"] = mmcv_parallel

    for root in ("mmdet", "mmseg"):
        sys.modules[root] = types.ModuleType(root)

        apis = types.ModuleType(f"{root}.apis")
        apis.set_random_seed = lambda *args, **kwargs: None
        if root == "mmdet":
            apis.train_detector = lambda *args, **kwargs: None
        else:
            apis.train_segmentor = lambda *args, **kwargs: None
            apis.single_gpu_test = lambda *args, **kwargs: []
        sys.modules[f"{root}.apis"] = apis

        datasets = types.ModuleType(f"{root}.datasets")
        datasets.build_dataset = lambda *args, **kwargs: None
        if root == "mmseg":
            datasets.build_dataloader = lambda *args, **kwargs: None
        sys.modules[f"{root}.datasets"] = datasets

        models = types.ModuleType(f"{root}.models")
        models.build_detector = lambda *args, **kwargs: None
        models.build_segmentor = lambda *args, **kwargs: None
        sys.modules[f"{root}.models"] = models

        builder = types.ModuleType(f"{root}.models.builder")
        builder.BACKBONES = Registry()
        sys.modules[f"{root}.models.builder"] = builder


def _parser_for_config(path):
    from utils.cfg import load_cfg

    old_argv = sys.argv[:]
    try:
        sys.argv = ["validate_seg_det", "--cfg", path]
        parser = argparse.ArgumentParser()
        parser.add_argument("--cfg", type=str, default="")
        parser.add_argument("--resume", type=str, default="")
        parser.add_argument("--workers_per_gpu", type=int, default=None)
        parser.add_argument("--model", type=str, default=None)
        parser.add_argument("--data_dir", type=str, default=None)
        parser.add_argument("--time_profile", action="store_true")
        parser.add_argument("--time_profile_interval", type=int, default=1000)
        return load_cfg(parser)
    finally:
        sys.argv = old_argv


def _validate_task_configs():
    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    checked = {"seg": 0, "det": 0}

    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "seg*.yaml"))):
        args = _parser_for_config(path)
        task = object.__new__(SegmentationTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmseg_config()
        if cfg.model["type"] != "EncoderDecoder":
            raise AssertionError(f"{path}: unexpected segmentation model {cfg.model['type']}")
        if int(cfg.data["workers_per_gpu"]) <= 0:
            raise AssertionError(f"{path}: workers_per_gpu should be positive")
        checked["seg"] += 1

    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "detection_*.yaml"))):
        args = _parser_for_config(path)
        task = object.__new__(DetectionTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmdet_config()
        if cfg.model["type"] != "MaskRCNN":
            raise AssertionError(f"{path}: unexpected detection model {cfg.model['type']}")
        if int(cfg.data["workers_per_gpu"]) <= 0:
            raise AssertionError(f"{path}: workers_per_gpu should be positive")
        checked["det"] += 1

    return checked


def _validate_backbone_forwards():
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
    expected_hw = [(8, 8), (4, 4), (2, 2), (1, 1)]
    cases = []
    for style in ("resize", "simple_fpn"):
        cases.extend(
            [
                ("ViTBackbone", ViTBackbone, dict(common, fpn_adapter_style=style)),
                ("ViTCoPEBackbone", ViTCoPEBackbone, dict(common, fpn_adapter_style=style, use_cls_token=False)),
                ("ViTCoPEBackbone+cls", ViTCoPEBackbone, dict(common, fpn_adapter_style=style, use_cls_token=True)),
                ("ViTSCoPEBackbone", ViTSCoPEBackbone, dict(common, fpn_adapter_style=style, use_cls_token=True)),
                ("ViTSCoPEBackbone-no-cls", ViTSCoPEBackbone, dict(common, fpn_adapter_style=style, use_cls_token=False)),
            ]
        )

    x = torch.randn(1, 3, 32, 32)
    for name, cls, kwargs in cases:
        model = cls(**kwargs).eval()
        with torch.no_grad():
            outs = model(x)
        if len(outs) != 4:
            raise AssertionError(f"{name}: expected 4 outputs, got {len(outs)}")
        for idx, (out, expected) in enumerate(zip(outs, expected_hw)):
            if tuple(out.shape) != (1, 32, expected[0], expected[1]):
                raise AssertionError(
                    f"{name} output {idx}: expected {(1, 32, *expected)}, got {tuple(out.shape)}"
                )

    return len(cases)


def main():
    _install_openmmlab_stubs()
    config_counts = _validate_task_configs()
    backbone_cases = _validate_backbone_forwards()
    print(
        "Validated "
        f"{config_counts['seg']} segmentation configs, "
        f"{config_counts['det']} detection configs, "
        f"and {backbone_cases} custom backbone CPU forward cases."
    )


if __name__ == "__main__":
    main()
