#!/usr/bin/env python3
"""Lightweight validation for segmentation and detection configuration paths.

The full training stack requires mmcv-full CUDA/C++ extensions.  This script
stubs only the registry/API pieces needed to exercise SCOPE's config builders,
then runs CPU smoke tests for the custom dense-prediction backbones.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


class Config(dict):
    """Small attribute-access dict compatible with what the tasks use."""

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


def _noop(*args, **kwargs):
    return None


def _install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object

    mmseg = types.ModuleType("mmseg")
    mmseg_apis = types.ModuleType("mmseg.apis")
    mmseg_apis.set_random_seed = _noop
    mmseg_apis.single_gpu_test = lambda *args, **kwargs: []
    mmseg_apis.train_segmentor = _noop
    mmseg_datasets = types.ModuleType("mmseg.datasets")
    mmseg_datasets.build_dataloader = _noop
    mmseg_datasets.build_dataset = _noop
    mmseg_models = types.ModuleType("mmseg.models")
    mmseg_models.build_segmentor = _noop
    mmseg_builder = types.ModuleType("mmseg.models.builder")
    mmseg_builder.BACKBONES = Registry()

    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = _noop
    mmdet_apis.train_detector = _noop
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = _noop
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = _noop
    mmdet_builder = types.ModuleType("mmdet.models.builder")
    mmdet_builder.BACKBONES = Registry()

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


OPTIONAL_DEFAULTS = {
    "workers_per_gpu": None,
    "checkpoint_interval": None,
    "eval_interval": None,
    "log_interval": None,
    "final_eval": None,
    "seg_head_dim": None,
    "seg_aux_dim": None,
    "seg_aux_in_index": None,
    "seg_norm_type": None,
    "seg_neck_style": None,
    "seg_neck_dim": None,
    "backbone_size": None,
    "det_neck_type": None,
    "warmup_iters": None,
    "warmup_epochs": None,
    "weight_decay": None,
    "layer_decay_rate": None,
    "drop_path_rate": None,
    "out_indices": None,
    "dim_head": None,
    "min_lr": None,
}


INT_FIELDS = {
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

FLOAT_FIELDS = {
    "lr",
    "min_lr",
    "warmup_epochs",
    "drop_path_rate",
    "weight_decay",
    "dropout",
    "emb_dropout",
    "layer_decay_rate",
}

BOOL_FIELDS = {"amp", "aug", "nowandb", "use_cls_token", "final_eval"}


def _normalize_value(key, value):
    if value is None:
        return None
    if key in INT_FIELDS:
        return int(value)
    if key in FLOAT_FIELDS:
        return float(value)
    if key in BOOL_FIELDS:
        return bool(value)
    return value


def _load_args(path):
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    values = {key: _normalize_value(key, value) for key, value in raw.items()}
    for key, value in OPTIONAL_DEFAULTS.items():
        values.setdefault(key, value)
    values.setdefault("cfg", str(path))
    values.setdefault("resume", "")
    values.setdefault("run_tag", None)
    return argparse.Namespace(**values)


def _config_paths(prefix):
    return sorted(CONFIG_DIR.glob(f"{prefix}_*.yaml"))


def _build_task_config(task_cls, args, build_method_name):
    task = object.__new__(task_cls)
    task.args = args
    task.device = "cpu"
    task.run_name = task._build_run_name()
    return getattr(task, build_method_name)()


def validate_configs():
    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    seg_paths = _config_paths("seg")
    det_paths = _config_paths("detection")
    failures = []

    for path in seg_paths:
        args = _load_args(path)
        try:
            cfg = _build_task_config(SegmentationTask, args, "_build_mmseg_config")
            assert cfg.model["type"] == "EncoderDecoder"
            assert cfg.data["workers_per_gpu"] == 4
        except Exception as exc:  # pragma: no cover - diagnostic path
            failures.append((path, exc))

    for path in det_paths:
        args = _load_args(path)
        try:
            cfg = _build_task_config(DetectionTask, args, "_build_mmdet_config")
            assert cfg.model["type"] == "MaskRCNN"
            assert cfg.data["workers_per_gpu"] == 4
        except Exception as exc:  # pragma: no cover - diagnostic path
            failures.append((path, exc))

    if failures:
        for path, exc in failures:
            print(f"FAILED config build: {path.relative_to(ROOT)}: {exc}")
        raise SystemExit(1)

    print(f"Validated {len(seg_paths)} segmentation configs and {len(det_paths)} detection configs.")


def _expected_feature_sizes():
    return [(8, 8), (4, 4), (2, 2), (1, 1)]


def _assert_feature_shapes(name, outputs, dim):
    assert len(outputs) == 4, f"{name}: expected 4 outputs, got {len(outputs)}"
    for idx, (output, hw) in enumerate(zip(outputs, _expected_feature_sizes())):
        expected = (1, dim, *hw)
        actual = tuple(output.shape)
        assert actual == expected, f"{name} output {idx}: expected {expected}, got {actual}"


def validate_backbones():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    dim = 24
    common = dict(
        image_size=32,
        patch_size=16,
        dim=dim,
        depth=4,
        heads=3,
        mlp_dim=48,
        dim_head=8,
        out_indices=(0, 1, 2, 3),
    )
    cases = []
    for style in ("resize", "simple_fpn"):
        cases.append((f"ViTBackbone[{style}]", ViTBackbone(fpn_adapter_style=style, **common)))
        for use_cls_token in (False, True):
            cases.append(
                (
                    f"ViTCoPEBackbone[{style},cls={use_cls_token}]",
                    ViTCoPEBackbone(fpn_adapter_style=style, use_cls_token=use_cls_token, **common),
                )
            )
            cases.append(
                (
                    f"ViTSCoPEBackbone[{style},cls={use_cls_token}]",
                    ViTSCoPEBackbone(fpn_adapter_style=style, use_cls_token=use_cls_token, **common),
                )
            )

    sample = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        for name, model in cases:
            model.eval()
            outputs = model(sample)
            _assert_feature_shapes(name, outputs, dim)

    print(f"Validated {len(cases)} custom backbone CPU forward cases.")


def main():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    _install_openmmlab_stubs()
    validate_configs()
    validate_backbones()


if __name__ == "__main__":
    main()
