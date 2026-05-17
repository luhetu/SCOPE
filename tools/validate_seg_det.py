#!/usr/bin/env python3
"""Lightweight segmentation/detection validation for this repository.

The full training entrypoint needs mmcv-full compiled ops. This script validates
the repository-owned pieces that can run in a minimal CPU environment:

* all segmentation and detection YAMLs load into their MMSeg/MMDet configs
* Swin configs do not require ViT-only arguments
* custom dense-prediction backbones produce four feature maps with expected
  spatial sizes
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ConfigDict(dict):
    """Tiny attribute-access dict compatible with the config builders."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = self._wrap(value)

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            return cls({k: cls._wrap(v) for k, v in value.items()})
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cls._wrap(v) for v in value)
        return value


class Config(ConfigDict):
    pass


class Registry:
    def __init__(self):
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        def _register(cls):
            module_name = name or cls.__name__
            if module_name in self.module_dict and not force:
                raise KeyError(f"{module_name} is already registered")
            self.module_dict[module_name] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


def _noop(*args, **kwargs):
    return None


def install_openmmlab_stubs():
    """Install enough mmcv/mmseg/mmdet stubs for config construction."""

    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object
    mmcv.parallel = mmcv_parallel

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
    mmdet_models.builder = mmdet_builder
    mmdet.models = mmdet_models
    mmdet.apis = mmdet_apis
    mmdet.datasets = mmdet_datasets

    mmseg = types.ModuleType("mmseg")
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
    mmseg_models.builder = mmseg_builder
    mmseg.models = mmseg_models
    mmseg.apis = mmseg_apis
    mmseg.datasets = mmseg_datasets

    sys.modules.update(
        {
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
    )


def _coerce_cfg_value(key, value):
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
    if key in {
        "warmup_epochs",
        "drop_path_rate",
        "weight_decay",
        "dropout",
        "emb_dropout",
        "layer_decay_rate",
    }:
        return float(value)
    if key in {"amp", "aug", "nowandb", "use_cls_token"}:
        return bool(value)
    if key == "pretrained" and value == "null":
        return None
    return value


def load_args(cfg_path: Path) -> SimpleNamespace:
    args = SimpleNamespace(
        cfg=str(cfg_path),
        resume="",
        workers_per_gpu=None,
        model=None,
        data_dir=None,
        time_profile=False,
        time_profile_interval=1000,
    )
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    for key, value in cfg.items():
        setattr(args, key, _coerce_cfg_value(key, value))
    return args


def apply_train_schedule_defaults(args: SimpleNamespace) -> None:
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


def config_paths(selected):
    if selected:
        return [Path(path).resolve() for path in selected]
    return sorted((ROOT / "configs").glob("seg_*.yaml")) + sorted(
        (ROOT / "configs").glob("detection_*.yaml")
    )


def build_task_config(cfg_path: Path):
    args = load_args(cfg_path)
    apply_train_schedule_defaults(args)

    if args.task == "seg":
        from tasks.segmentation import SegmentationTask

        task = SegmentationTask.__new__(SegmentationTask)
        task.args = args
        task.run_name = task._build_run_name()
        return task._build_mmseg_config()

    if args.task == "det":
        from tasks.detection import DetectionTask

        task = DetectionTask.__new__(DetectionTask)
        task.args = args
        task.run_name = task._build_run_name()
        return task._build_mmdet_config()

    raise ValueError(f"Unsupported dense-prediction task: {args.task}")


def validate_configs(paths):
    failures = []
    for cfg_path in paths:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cfg = build_task_config(cfg_path)
            backbone_type = cfg.model.backbone.type
            print(f"PASS config {cfg_path.relative_to(ROOT)} ({backbone_type})")
        except Exception as exc:  # noqa: BLE001 - report all config failures.
            failures.append((cfg_path, exc))
            print(f"FAIL config {cfg_path.relative_to(ROOT)}: {exc}")
    return failures


def expected_feature_shapes(style: str, dim: int):
    if style in {"simple_fpn", "resize"}:
        return [
            (1, dim, 8, 8),
            (1, dim, 4, 4),
            (1, dim, 2, 2),
            (1, dim, 1, 1),
        ]
    return [(1, dim, 2, 2)] * 4


def validate_backbones():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    dim = 32
    image_size = 32
    patch_size = 16
    common = dict(
        image_size=image_size,
        patch_size=patch_size,
        dim=dim,
        depth=4,
        heads=4,
        mlp_dim=64,
        dim_head=8,
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.0,
    )
    cases = [
        ("vit", ViTBackbone, {}),
        ("vitcope_no_cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("vitcope_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("vitscope_no_cls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("vitscope_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]
    styles = ("simple_fpn", "resize", "identity")
    failures = []
    x = torch.randn(1, 3, image_size, image_size)

    with torch.no_grad():
        for style in styles:
            expected = expected_feature_shapes(style, dim)
            for name, cls, extra in cases:
                case_name = f"{name}/{style}"
                try:
                    model = cls(fpn_adapter_style=style, **common, **extra)
                    model.eval()
                    outputs = model(x)
                    shapes = [tuple(output.shape) for output in outputs]
                    if shapes != expected:
                        raise AssertionError(f"expected {expected}, got {shapes}")
                    print(f"PASS backbone {case_name}: {shapes}")
                except Exception as exc:  # noqa: BLE001 - report all smoke failures.
                    failures.append((case_name, exc))
                    print(f"FAIL backbone {case_name}: {exc}")
    return failures


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs",
        nargs="*",
        help="Optional config paths. Defaults to all configs/seg_*.yaml and configs/detection_*.yaml.",
    )
    parser.add_argument(
        "--skip-backbones",
        action="store_true",
        help="Only validate YAML-to-OpenMMLab config construction.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    install_openmmlab_stubs()

    paths = config_paths(args.configs)
    print(f"Validating {len(paths)} segmentation/detection config(s)")
    config_failures = validate_configs(paths)

    backbone_failures = []
    if not args.skip_backbones:
        print("\nValidating custom backbone CPU forward passes")
        backbone_failures = validate_backbones()

    failures = config_failures + backbone_failures
    if failures:
        print(f"\nValidation failed with {len(failures)} failure(s).")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
