#!/usr/bin/env python3
"""Lightweight validation for segmentation and detection task wiring.

This script avoids full MMDetection/MMSegmentation trainer construction so it
can run in a CPU-only development environment. It verifies every dense-task YAML
can build its internal config and smoke-tests the custom ViT backbones.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Config(dict):
    """Minimal mmcv.Config stand-in for task config builders."""

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
            module_name = name or cls.__name__
            if force or module_name not in self.module_dict:
                self.module_dict[module_name] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


def _noop(*args, **kwargs):
    return None


def install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object

    mmseg = types.ModuleType("mmseg")
    mmseg.__path__ = []
    mmseg_apis = types.ModuleType("mmseg.apis")
    mmseg_apis.set_random_seed = _noop
    mmseg_apis.single_gpu_test = _noop
    mmseg_apis.train_segmentor = _noop
    mmseg_datasets = types.ModuleType("mmseg.datasets")
    mmseg_datasets.build_dataloader = _noop
    mmseg_datasets.build_dataset = _noop
    mmseg_models = types.ModuleType("mmseg.models")
    mmseg_models.build_segmentor = _noop
    mmseg_builder = types.ModuleType("mmseg.models.builder")
    mmseg_builder.BACKBONES = Registry()

    mmdet = types.ModuleType("mmdet")
    mmdet.__path__ = []
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = _noop
    mmdet_apis.train_detector = _noop
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = _noop
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = _noop
    mmdet_builder = types.ModuleType("mmdet.models.builder")
    mmdet_builder.BACKBONES = Registry()

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
    mmseg.apis = mmseg_apis
    mmseg.datasets = mmseg_datasets
    mmseg.models = mmseg_models
    mmseg_models.builder = mmseg_builder
    mmdet.apis = mmdet_apis
    mmdet.datasets = mmdet_datasets
    mmdet.models = mmdet_models
    mmdet_models.builder = mmdet_builder


def load_args(config_path: Path) -> SimpleNamespace:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    # Mirror argparse defaults that can otherwise mask None-handling issues.
    defaults = {
        "cfg": str(config_path),
        "resume": "",
        "workers_per_gpu": None,
        "log_interval": None,
        "eval_interval": None,
        "checkpoint_interval": None,
        "warmup_iters": None,
        "run_tag": None,
        "seed": None,
        "final_eval": None,
    }
    defaults.update(data)
    return SimpleNamespace(**defaults)


def build_task_config(task_cls, args, method_name):
    task = task_cls.__new__(task_cls)
    task.args = args
    task.run_name = task._build_run_name()
    return getattr(task, method_name)()


def validate_configs() -> int:
    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    checked = 0
    config_paths = sorted(ROOT.glob("configs/seg_*.yaml")) + sorted(ROOT.glob("configs/detection_*.yaml"))
    for config_path in config_paths:
        args = load_args(config_path)
        if args.task == "seg":
            cfg = build_task_config(SegmentationTask, args, "_build_mmseg_config")
            assert cfg.model["type"] == "EncoderDecoder"
            assert cfg.data["workers_per_gpu"] == 4
        elif args.task == "det":
            cfg = build_task_config(DetectionTask, args, "_build_mmdet_config")
            assert cfg.model["type"] == "MaskRCNN"
            assert cfg.data["workers_per_gpu"] == 4
        else:
            raise AssertionError(f"{config_path}: unexpected task {args.task!r}")
        checked += 1
    print(f"validated {checked} segmentation/detection configs")
    return checked


def validate_backbones() -> int:
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    cases = []
    for style in ("resize", "simple_fpn"):
        common = dict(
            image_size=32,
            patch_size=16,
            dim=32,
            depth=4,
            heads=2,
            mlp_dim=64,
            dim_head=16,
            out_indices=(0, 1, 2, 3),
            fpn_adapter_style=style,
        )
        cases.extend(
            [
                ("vit", ViTBackbone(**common)),
                ("vitcope_no_cls", ViTCoPEBackbone(**common, use_cls_token=False)),
                ("vitcope_cls", ViTCoPEBackbone(**common, use_cls_token=True)),
                ("vitscope_no_cls", ViTSCoPEBackbone(**common, use_cls_token=False)),
                ("vitscope_cls", ViTSCoPEBackbone(**common, use_cls_token=True)),
            ]
        )

    expected_hw = [(8, 8), (4, 4), (2, 2), (1, 1)]
    image = torch.randn(1, 3, 32, 32)
    checked = 0
    with torch.no_grad():
        for name, model in cases:
            model.eval()
            outputs = model(image)
            assert len(outputs) == 4, f"{name}: expected 4 outputs, got {len(outputs)}"
            for output, (height, width) in zip(outputs, expected_hw):
                assert output.shape == (1, 32, height, width), f"{name}: unexpected output shape {tuple(output.shape)}"
            checked += 1
    print(f"validated {checked} custom backbone forward cases")
    return checked


def main() -> int:
    install_openmmlab_stubs()
    config_count = validate_configs()
    backbone_count = validate_backbones()
    print(f"segmentation/detection validation passed ({config_count} configs, {backbone_count} backbones)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
