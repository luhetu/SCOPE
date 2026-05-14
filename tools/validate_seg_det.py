#!/usr/bin/env python3
"""Validate segmentation and detection configuration paths.

This is a lightweight smoke test for environments that do not have the full
mmcv-full/mmseg/mmdet stack installed. It stubs the OpenMMLab builders, imports
the project task classes, and verifies every seg/det YAML can build its config.
If torch/timm/einops are available, it also runs small CPU forward passes
through the custom dense-prediction ViT backbones.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import sys
import types
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Config(dict):
    """Small mmcv.Config stand-in for task config construction."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class Registry:
    """Minimal registry compatible with register_module usage in task files."""

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


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _install_task_stubs() -> None:
    """Install enough stubs for config builders to import without OpenMMLab."""

    if "torch" not in sys.modules and not _has_module("torch"):
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.load = lambda *args, **kwargs: {}
        sys.modules["torch"] = torch
        sys.modules["torch.nn"] = types.ModuleType("torch.nn")
        sys.modules["torch.nn.functional"] = types.ModuleType("torch.nn.functional")

    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    sys.modules["mmcv"] = mmcv

    parallel = types.ModuleType("mmcv.parallel")
    parallel.MMDataParallel = object
    sys.modules["mmcv.parallel"] = parallel

    for root in ("mmseg", "mmdet"):
        pkg = types.ModuleType(root)
        pkg.__path__ = []
        sys.modules[root] = pkg

        apis = types.ModuleType(f"{root}.apis")
        apis.set_random_seed = lambda *args, **kwargs: None
        if root == "mmseg":
            apis.single_gpu_test = lambda *args, **kwargs: []
            apis.train_segmentor = lambda *args, **kwargs: None
        else:
            apis.train_detector = lambda *args, **kwargs: None
        sys.modules[f"{root}.apis"] = apis

        datasets = types.ModuleType(f"{root}.datasets")
        datasets.build_dataset = lambda *args, **kwargs: types.SimpleNamespace(CLASSES=[])
        datasets.build_dataloader = lambda *args, **kwargs: None
        sys.modules[f"{root}.datasets"] = datasets

        models = types.ModuleType(f"{root}.models")
        if root == "mmseg":
            models.build_segmentor = lambda *args, **kwargs: types.SimpleNamespace(CLASSES=[])
        else:
            models.build_detector = lambda *args, **kwargs: types.SimpleNamespace(CLASSES=[])
        sys.modules[f"{root}.models"] = models

        builder = types.ModuleType(f"{root}.models.builder")
        builder.BACKBONES = Registry()
        sys.modules[f"{root}.models.builder"] = builder

    if "models.vit_backbone" not in sys.modules:
        vit_backbone = types.ModuleType("models.vit_backbone")
        for name in ("ViTBackbone", "ViTCoPEBackbone", "ViTSCoPEBackbone"):
            setattr(vit_backbone, name, type(name, (), {}))
        sys.modules["models.vit_backbone"] = vit_backbone


def _coerce_cfg_types(data: dict) -> dict:
    """Mirror the type coercions in utils.cfg.load_cfg for direct YAML loading."""

    coerced = dict(data)
    for key, value in list(coerced.items()):
        if key in ("lr", "min_lr"):
            coerced[key] = float(value)
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
            coerced[key] = int(value)
        elif key in {
            "warmup_epochs",
            "drop_path_rate",
            "weight_decay",
            "dropout",
            "emb_dropout",
            "layer_decay_rate",
        }:
            coerced[key] = float(value)
        elif key in ("amp", "aug", "nowandb", "use_cls_token"):
            coerced[key] = bool(value)
        elif key == "pretrained" and value == "null":
            coerced[key] = None
    return coerced


def _load_args(path: Path) -> argparse.Namespace:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = _coerce_cfg_types(data)
    return argparse.Namespace(cfg=str(path.relative_to(ROOT)), **data)


def _assert_shared_train_fields(args: argparse.Namespace, path: Path) -> None:
    """Check fields used by train.py before task-specific branching."""

    required = ("task", "model", "data_dir", "bs", "lr", "n_epochs", "size", "patch")
    missing = [name for name in required if not hasattr(args, name)]
    if missing:
        raise AssertionError(f"{path.relative_to(ROOT)} is missing shared fields: {missing}")


def validate_config_builders() -> list[str]:
    _install_task_stubs()

    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    checks = [
        ("seg", "seg_*.yaml", SegmentationTask, "_build_mmseg_config"),
        ("det", "detection_*.yaml", DetectionTask, "_build_mmdet_config"),
    ]
    summaries = []
    failures = []

    for kind, pattern, task_cls, build_method in checks:
        for path in sorted((ROOT / "configs").glob(pattern)):
            args = _load_args(path)
            try:
                _assert_shared_train_fields(args, path)
                task = object.__new__(task_cls)
                task.args = args
                task.device = "cpu"
                task.run_name = task._build_run_name()
                with contextlib.redirect_stdout(io.StringIO()):
                    cfg = getattr(task, build_method)()
                backbone = cfg.model["backbone"]["type"]
                summaries.append(f"config {kind}: {path.relative_to(ROOT)} -> {backbone}")
            except Exception as exc:  # noqa: BLE001 - report all validation failures.
                failures.append(f"{path.relative_to(ROOT)}: {type(exc).__name__}: {exc}")

    if failures:
        joined = "\n  - ".join(failures)
        raise RuntimeError(f"seg/det config validation failed:\n  - {joined}")
    return summaries


def validate_backbone_forwards() -> list[str]:
    missing = [name for name in ("torch", "timm", "einops") if not _has_module(name)]
    if missing:
        return [f"backbone forwards skipped; missing optional modules: {', '.join(missing)}"]

    import torch
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    cases = [
        ("ViTBackbone", ViTBackbone, {}),
        ("ViTCoPEBackbone-no-cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("ViTCoPEBackbone-cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("ViTSCoPEBackbone-no-cls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("ViTSCoPEBackbone-cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]
    expected_shapes = [(1, 16, 8, 8), (1, 16, 4, 4), (1, 16, 2, 2), (1, 16, 1, 1)]
    summaries = []

    for adapter_style in ("resize", "simple_fpn"):
        for name, cls, extra_kwargs in cases:
            model = cls(
                image_size=32,
                patch_size=16,
                dim=16,
                depth=4,
                heads=2,
                mlp_dim=32,
                dim_head=8,
                out_indices=(0, 1, 2, 3),
                fpn_adapter_style=adapter_style,
                **extra_kwargs,
            )
            model.eval()
            with torch.no_grad():
                outputs = model(torch.randn(1, 3, 32, 32))
            shapes = [tuple(output.shape) for output in outputs]
            if shapes != expected_shapes:
                raise AssertionError(
                    f"{name} ({adapter_style}) produced {shapes}, expected {expected_shapes}"
                )
            summaries.append(f"backbone {name} ({adapter_style}) -> {shapes}")

    return summaries


def main() -> int:
    results = []
    results.extend(validate_backbone_forwards())
    results.extend(validate_config_builders())

    print("Segmentation/detection validation passed:")
    for line in results:
        print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
