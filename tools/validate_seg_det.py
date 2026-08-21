#!/usr/bin/env python3
"""Lightweight validation for segmentation and detection configuration code.

The full training stacks require mmcv-full, datasets, and checkpoints. This
script focuses on repo-local checks that can run in a plain CI/cloud shell:

* Stub the small OpenMMLab surface needed to import task modules.
* Build every configs/seg_*.yaml and configs/detection_*.yaml task config.
* Smoke-test custom dense-prediction backbones on CPU when torch deps exist.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class Config(dict):
    """Small mmcv.Config stand-in with attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class Registry:
    def __init__(self) -> None:
        self.module_dict: Dict[str, Any] = {}

    def register_module(self, name: str | None = None, module: Any | None = None, force: bool = False):
        def _register(cls):
            key = name or cls.__name__
            if key in self.module_dict and not force:
                raise KeyError(f"{key} is already registered")
            self.module_dict[key] = cls
            return cls

        return _register(module) if module is not None else _register


def _noop(*args, **kwargs):
    return None


def _install_openmmlab_stubs() -> None:
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


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _has_torch_backbone_deps() -> Tuple[bool, List[str]]:
    missing = [name for name in ("torch", "timm", "einops") if not _has_module(name)]
    return not missing, missing


def _install_backbone_stub() -> None:
    module = types.ModuleType("models.vit_backbone")

    class _Backbone:
        pass

    module.ViTBackbone = _Backbone
    module.ViTCoPEBackbone = _Backbone
    module.ViTSCoPEBackbone = _Backbone
    sys.modules["models.vit_backbone"] = module


def _install_torch_stub() -> None:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_nn = types.ModuleType("torch.nn")
    torch_nn_functional = types.ModuleType("torch.nn.functional")
    torch_nn_functional.interpolate = _noop
    torch.nn = torch_nn
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch_nn
    sys.modules["torch.nn.functional"] = torch_nn_functional


def _is_null_like(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in {"", "null", "none"})


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _coerce_value(key: str, value: Any) -> Any:
    if _is_null_like(value):
        return None
    if key in {"lr", "min_lr"}:
        return float(value)
    if key in {
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
        return int(value)
    if key in {"warmup_epochs", "drop_path_rate", "weight_decay", "dropout", "emb_dropout", "layer_decay_rate"}:
        return float(value)
    if key in {"amp", "aug", "nowandb", "use_cls_token"}:
        return _as_bool(value)
    return value


DEFAULT_ARGS: Dict[str, Any] = {
    "cfg": "",
    "run_tag": None,
    "seed": None,
    "pretrained": None,
    "nowandb": True,
    "amp": False,
    "task": None,
    "model": None,
    "data_dir": "/tmp/scope-data",
    "bs": 1,
    "size": 224,
    "n_epochs": 1,
    "max_iters": None,
    "warmup_epochs": 0,
    "warmup_iters": None,
    "checkpoint_interval": 5000,
    "eval_interval": 2001,
    "log_interval": 100,
    "lr": 1e-4,
    "min_lr": 0.0,
    "weight_decay": 0.05,
    "betas": None,
    "patch": 16,
    "dim": 192,
    "depth": 12,
    "heads": 3,
    "mlp_dim": 768,
    "dim_head": 64,
    "drop_path_rate": 0.0,
    "out_indices": (3, 5, 7, 11),
    "use_cls_token": None,
    "crop_size": 512,
    "img_scale": None,
    "test_img_scale": None,
    "backbone_size": None,
    "seg_head_dim": None,
    "seg_aux_dim": None,
    "seg_aux_in_index": 2,
    "seg_neck_dim": None,
    "seg_neck_style": "xcit_fpn",
    "seg_norm_type": "SyncBN",
    "det_neck_type": "fpn",
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
    "workers_per_gpu": None,
    "final_eval": False,
}


def _load_args(config_path: str) -> SimpleNamespace:
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle) or {}
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping, got {type(raw_cfg).__name__}")

    values = dict(DEFAULT_ARGS)
    values["cfg"] = config_path
    for key, value in raw_cfg.items():
        values[key] = _coerce_value(key, value)
    return SimpleNamespace(**values)


def _make_train_parser() -> argparse.ArgumentParser:
    """Match train.py argparse defaults, including workers_per_gpu=None."""
    parser = argparse.ArgumentParser(description="Unified ViT/CoPE/SCoPE Trainer")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--workers_per_gpu", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--time_profile", action="store_true")
    parser.add_argument("--time_profile_interval", type=int, default=1000)
    return parser


def _load_cfg_with_argv(argv: List[str]):
    from utils.cfg import load_cfg

    original_argv = sys.argv
    try:
        sys.argv = argv
        return load_cfg(_make_train_parser())
    finally:
        sys.argv = original_argv


def _validate_cli_overrides(config_path: str) -> None:
    expected = {
        "model": "cli-model",
        "data_dir": "/tmp/scope-cli-data",
        "workers_per_gpu": 2,
    }
    args = _load_cfg_with_argv(
        [
            "train.py",
            "--cfg",
            config_path,
            "--model",
            expected["model"],
            "--data_dir",
            expected["data_dir"],
            "--workers_per_gpu",
            str(expected["workers_per_gpu"]),
        ]
    )
    actual = {key: getattr(args, key) for key in expected}
    if actual != expected:
        raise AssertionError(f"CLI overrides were not preserved: {actual}, expected {expected}")
    print("OK config loader CLI overrides")


def _validate_real_parser_builds(seg_paths: Iterable[str], det_paths: Iterable[str]) -> int:
    """Build every config through train.py's real parser + load_cfg path."""
    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    count = 0
    for task_name, paths, builder_cls, build_method, expected_type in (
        ("seg", seg_paths, SegmentationTask, "_build_mmseg_config", "EncoderDecoder"),
        ("det", det_paths, DetectionTask, "_build_mmdet_config", "MaskRCNN"),
    ):
        for path in paths:
            args = _load_cfg_with_argv(["train.py", "--cfg", path])
            if args.task != task_name:
                raise AssertionError(f"{path}: expected task={task_name}, got {args.task!r}")
            if getattr(args, "workers_per_gpu", "missing") is not None:
                raise AssertionError(
                    f"{path}: expected train.py default workers_per_gpu=None, got {args.workers_per_gpu!r}"
                )
            task = object.__new__(builder_cls)
            task.args = args
            task.run_name = task._build_run_name()
            cfg = getattr(task, build_method)()
            _assert_config(task_name, path, cfg)
            if cfg.data["workers_per_gpu"] != 4:
                raise AssertionError(
                    f"{path}: expected workers_per_gpu fallback 4, got {cfg.data['workers_per_gpu']!r}"
                )
            if cfg.model["type"] != expected_type:
                raise AssertionError(f"{path}: expected {expected_type}, got {cfg.model['type']}")
            print(f"OK real-parser {task_name} config: {os.path.relpath(path, REPO_ROOT)}")
            count += 1
    return count


def _assert_config(task: str, path: str, cfg: Config) -> None:
    required = ("model", "data", "optimizer", "runner", "checkpoint_config", "evaluation", "work_dir")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise AssertionError(f"{path}: missing config keys: {missing}")
    if "backbone" not in cfg.model:
        raise AssertionError(f"{path}: model config has no backbone")
    if not cfg.data.get("train") or not cfg.data.get("val") or not cfg.data.get("test"):
        raise AssertionError(f"{path}: data splits are incomplete")
    if task == "seg" and cfg.model["type"] != "EncoderDecoder":
        raise AssertionError(f"{path}: expected EncoderDecoder model")
    if task == "det" and cfg.model["type"] != "MaskRCNN":
        raise AssertionError(f"{path}: expected MaskRCNN model")


def _validate_segmentation(paths: Iterable[str]) -> int:
    from tasks.segmentation import SegmentationTask

    count = 0
    for path in paths:
        args = _load_args(path)
        if args.task != "seg":
            raise AssertionError(f"{path}: expected task=seg, got {args.task!r}")
        task = object.__new__(SegmentationTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmseg_config()
        _assert_config("seg", path, cfg)
        print(f"OK seg config: {os.path.relpath(path, REPO_ROOT)}")
        count += 1
    return count


def _validate_detection(paths: Iterable[str]) -> int:
    from tasks.detection import DetectionTask

    count = 0
    for path in paths:
        args = _load_args(path)
        if args.task != "det":
            raise AssertionError(f"{path}: expected task=det, got {args.task!r}")
        task = object.__new__(DetectionTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmdet_config()
        _assert_config("det", path, cfg)
        print(f"OK det config: {os.path.relpath(path, REPO_ROOT)}")
        count += 1
    return count


def _run_backbone_smoke() -> None:
    import torch
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    cases = [
        ("ViTBackbone", ViTBackbone, {}),
        ("ViTCoPEBackbone(no-cls)", ViTCoPEBackbone, {"use_cls_token": False}),
        ("ViTCoPEBackbone(cls)", ViTCoPEBackbone, {"use_cls_token": True}),
        ("ViTSCoPEBackbone(no-cls)", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("ViTSCoPEBackbone(cls)", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]
    expected_shapes = ((1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1))
    sample = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
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
                outputs = model(sample)
                shapes = tuple(tuple(out.shape) for out in outputs)
                if shapes != expected_shapes:
                    raise AssertionError(f"{name} ({style}) shapes {shapes}, expected {expected_shapes}")
                print(f"OK backbone smoke: {name} ({style})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-backbone", action="store_true", help="Skip CPU backbone forward smoke tests.")
    args = parser.parse_args()

    seg_paths = sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "seg_*.yaml")))
    det_paths = sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "detection_*.yaml")))
    if not seg_paths:
        raise RuntimeError("No segmentation configs matched configs/seg_*.yaml")
    if not det_paths:
        raise RuntimeError("No detection configs matched configs/detection_*.yaml")

    has_backbone_deps, missing = _has_torch_backbone_deps()
    _install_openmmlab_stubs()
    if not has_backbone_deps:
        if "torch" in missing:
            _install_torch_stub()
        _install_backbone_stub()

    _validate_cli_overrides(seg_paths[0])
    seg_count = _validate_segmentation(seg_paths)
    det_count = _validate_detection(det_paths)
    parser_count = _validate_real_parser_builds(seg_paths, det_paths)
    print(
        f"Validated {seg_count} segmentation configs, {det_count} detection configs, "
        f"and {parser_count} real-parser builds."
    )

    if args.skip_backbone:
        print("Skipped backbone smoke tests by request.")
    elif has_backbone_deps:
        _run_backbone_smoke()
    else:
        print(f"Skipped backbone smoke tests; missing modules: {', '.join(missing)}")

    print("Segmentation/detection validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
