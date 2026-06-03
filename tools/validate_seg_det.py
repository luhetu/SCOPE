#!/usr/bin/env python3
"""Lightweight smoke validation for segmentation and detection code paths.

The full OpenMMLab training stack requires compiled mmcv extensions and real
datasets. This script stubs only the OpenMMLab APIs used during config building
so that repository-level segmentation/detection wiring can be checked quickly.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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

    def register_module(self, name=None, module=None, force=False, **kwargs):
        def _register(cls):
            key = name or cls.__name__
            if key in self.module_dict and not force:
                return cls
            self.module_dict[key] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


class DummyDataset:
    CLASSES = tuple(f"class_{idx}" for idx in range(3))
    PALETTE = [(0, 0, 0), (128, 0, 0), (0, 128, 0)]


class DummyModel:
    CLASSES = DummyDataset.CLASSES

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict, strict=False):
        return None

    def cuda(self, device=None):
        return self


def _install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = lambda model, device_ids=None: model

    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = lambda seed, deterministic=False: None
    mmdet_apis.train_detector = lambda *args, **kwargs: None
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = lambda *args, **kwargs: DummyDataset()
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_builder = types.ModuleType("mmdet.models.builder")
    mmdet_builder.BACKBONES = Registry()
    mmdet_builder.build_detector = lambda *args, **kwargs: DummyModel()
    mmdet_models.build_detector = mmdet_builder.build_detector
    mmdet_models.builder = mmdet_builder

    mmseg = types.ModuleType("mmseg")
    mmseg_apis = types.ModuleType("mmseg.apis")
    mmseg_apis.set_random_seed = lambda seed, deterministic=False: None
    mmseg_apis.single_gpu_test = lambda *args, **kwargs: []
    mmseg_apis.train_segmentor = lambda *args, **kwargs: None
    mmseg_datasets = types.ModuleType("mmseg.datasets")
    mmseg_datasets.build_dataset = lambda *args, **kwargs: DummyDataset()
    mmseg_datasets.build_dataloader = lambda *args, **kwargs: []
    mmseg_models = types.ModuleType("mmseg.models")
    mmseg_builder = types.ModuleType("mmseg.models.builder")
    mmseg_builder.BACKBONES = Registry()
    mmseg_builder.build_segmentor = lambda *args, **kwargs: DummyModel()
    mmseg_models.build_segmentor = mmseg_builder.build_segmentor
    mmseg_models.builder = mmseg_builder

    modules = {
        "mmcv": mmcv,
        "mmcv.parallel": mmcv_parallel,
        "mmdet": mmdet,
        "mmdet.apis": mmdet_apis,
        "mmdet.datasets": mmdet_datasets,
        "mmdet.models": mmdet_models,
        "mmdet.models.builder": mmdet_builder,
        "mmseg": mmseg,
        "mmseg.apis": mmseg_apis,
        "mmseg.datasets": mmseg_datasets,
        "mmseg.models": mmseg_models,
        "mmseg.models.builder": mmseg_builder,
    }
    sys.modules.update(modules)


INT_KEYS = {
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
    "seg_aux_in_index",
    "seg_neck_dim",
    "embed_dim",
    "window_size",
}
FLOAT_KEYS = {
    "lr",
    "min_lr",
    "warmup_epochs",
    "drop_path_rate",
    "weight_decay",
    "dropout",
    "emb_dropout",
    "layer_decay_rate",
}
BOOL_KEYS = {"amp", "aug", "nowandb", "use_cls_token", "final_eval"}
OPTIONAL_NONE_KEYS = {
    "workers_per_gpu",
    "log_interval",
    "checkpoint_interval",
    "eval_interval",
    "max_iters",
    "warmup_iters",
    "min_lr",
    "weight_decay",
    "layer_decay_rate",
    "final_eval",
    "seg_head_dim",
    "seg_aux_dim",
    "seg_aux_in_index",
    "seg_norm_type",
    "seg_neck_style",
    "seg_neck_dim",
    "backbone_size",
    "test_img_scale",
    "img_scale",
    "dim_head",
    "out_indices",
    "use_cls_token",
    "det_neck_type",
    "drop_path_rate",
    "run_tag",
}


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_cfg(raw):
    cfg = {}
    for key, value in raw.items():
        if value == "null":
            value = None
        elif value is not None and key in INT_KEYS:
            value = int(value)
        elif value is not None and key in FLOAT_KEYS:
            value = float(value)
        elif value is not None and key in BOOL_KEYS:
            value = _as_bool(value)
        cfg[key] = value
    return cfg


def _load_args(path):
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    values = {"cfg": str(path), **{key: None for key in OPTIONAL_NONE_KEYS}}
    values.update(_coerce_cfg(raw))
    return argparse.Namespace(**values)


def _validate_config_paths(label, paths, task_cls, method_name):
    failures = []
    for path in paths:
        args = _load_args(path)
        task = object.__new__(task_cls)
        task.args = args
        task.run_name = path.stem
        try:
            cfg = getattr(task, method_name)()
            if not cfg.get("model") or not cfg.get("data"):
                raise AssertionError("missing model or data config")
        except Exception as exc:  # pragma: no cover - surfaced in CLI output
            failures.append((path, exc))

    if failures:
        print(f"\n{label} config validation failed:")
        for path, exc in failures:
            print(f"  - {path.relative_to(ROOT)}: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
    print(f"{label}: validated {len(paths)} configs")


def _validate_backbones():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    expected = {
        "simple_fpn": [(1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1)],
        "resize": [(1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1)],
    }
    constructors = [
        ("vit", ViTBackbone, {}),
        ("vitcope_no_cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("vitcope_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("vitscope_no_cls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("vitscope_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]
    cases = 0
    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        for style, expected_shapes in expected.items():
            for name, constructor, extra_kwargs in constructors:
                model = constructor(
                    image_size=32,
                    patch_size=16,
                    dim=32,
                    depth=4,
                    heads=4,
                    mlp_dim=64,
                    dim_head=8,
                    out_indices=(0, 1, 2, 3),
                    fpn_adapter_style=style,
                    **extra_kwargs,
                )
                model.eval()
                outputs = model(image)
                shapes = [tuple(output.shape) for output in outputs]
                if shapes != expected_shapes:
                    raise AssertionError(f"{name}/{style} shapes {shapes} != {expected_shapes}")
                cases += 1
    print(f"backbones: validated {cases} CPU forward cases")


def main():
    _install_openmmlab_stubs()

    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    seg_paths = sorted((ROOT / "configs").glob("seg_*.yaml"))
    det_paths = sorted((ROOT / "configs").glob("detection_*.yaml"))
    _validate_config_paths("segmentation", seg_paths, SegmentationTask, "_build_mmseg_config")
    _validate_config_paths("detection", det_paths, DetectionTask, "_build_mmdet_config")
    _validate_backbones()
    print("segmentation/detection smoke validation passed")


if __name__ == "__main__":
    main()
