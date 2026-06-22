#!/usr/bin/env python3
"""Lightweight validation for segmentation and detection wiring.

The full det/seg training paths need COCO/ADE20K, CUDA, and mmcv-full. This
script validates the parts that can be checked without datasets: YAML coverage,
MMDet/MMSeg config construction, default CLI handling, and (when torch is
installed) custom backbone forward shapes.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    "seg_neck_dim",
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


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError:
        return False


def _coerce_value(key, value):
    if value is None:
        return None
    if key in INT_KEYS:
        return int(value)
    if key in FLOAT_KEYS:
        return float(value)
    if key in BOOL_KEYS:
        return bool(value)
    if key == "pretrained" and value == "null":
        return None
    return value


def load_task_args(config_path: Path) -> Namespace:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    values = {
        "cfg": str(config_path),
        "resume": "",
        "workers_per_gpu": None,
        "model": None,
        "data_dir": None,
        "time_profile": False,
        "time_profile_interval": 1000,
        "run_tag": None,
        "seed": None,
    }
    for key, value in raw.items():
        values[key] = _coerce_value(key, value)
    return Namespace(**values)


class RegistryStub:
    def __init__(self):
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        def decorator(cls):
            module_name = name or cls.__name__
            if force or module_name not in self.module_dict:
                self.module_dict[module_name] = cls
            return cls

        if module is not None:
            return decorator(module)
        return decorator


class ConfigStub(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _install_torch_stub_if_needed() -> None:
    if _module_available("torch"):
        return
    torch_mod = types.ModuleType("torch")
    nn_mod = types.ModuleType("torch.nn")
    functional_mod = types.ModuleType("torch.nn.functional")

    class _Cuda:
        @staticmethod
        def is_available():
            return False

    torch_mod.cuda = _Cuda()
    torch_mod.nn = nn_mod
    nn_mod.functional = functional_mod
    sys.modules.setdefault("torch", torch_mod)
    sys.modules.setdefault("torch.nn", nn_mod)
    sys.modules.setdefault("torch.nn.functional", functional_mod)


def _install_backbone_stub_if_needed() -> None:
    if all(_module_available(name) for name in ("torch", "timm", "einops")):
        return
    backbone_mod = types.ModuleType("models.vit_backbone")

    class ViTBackbone:
        pass

    class ViTCoPEBackbone:
        pass

    class ViTSCoPEBackbone:
        pass

    backbone_mod.ViTBackbone = ViTBackbone
    backbone_mod.ViTCoPEBackbone = ViTCoPEBackbone
    backbone_mod.ViTSCoPEBackbone = ViTSCoPEBackbone
    sys.modules.setdefault("models.vit_backbone", backbone_mod)


def install_openmmlab_stubs() -> None:
    _install_torch_stub_if_needed()
    _install_backbone_stub_if_needed()

    mmcv_mod = types.ModuleType("mmcv")
    mmcv_mod.Config = ConfigStub
    mmcv_parallel_mod = types.ModuleType("mmcv.parallel")
    mmcv_parallel_mod.MMDataParallel = object

    mmdet_backbones = RegistryStub()
    mmdet_mod = types.ModuleType("mmdet")
    mmdet_apis_mod = types.ModuleType("mmdet.apis")
    mmdet_datasets_mod = types.ModuleType("mmdet.datasets")
    mmdet_models_mod = types.ModuleType("mmdet.models")
    mmdet_builder_mod = types.ModuleType("mmdet.models.builder")
    mmdet_apis_mod.set_random_seed = lambda *args, **kwargs: None
    mmdet_apis_mod.train_detector = lambda *args, **kwargs: None
    mmdet_datasets_mod.build_dataset = lambda *args, **kwargs: None
    mmdet_models_mod.build_detector = lambda *args, **kwargs: None
    mmdet_builder_mod.BACKBONES = mmdet_backbones

    mmseg_backbones = RegistryStub()
    mmseg_mod = types.ModuleType("mmseg")
    mmseg_apis_mod = types.ModuleType("mmseg.apis")
    mmseg_datasets_mod = types.ModuleType("mmseg.datasets")
    mmseg_models_mod = types.ModuleType("mmseg.models")
    mmseg_builder_mod = types.ModuleType("mmseg.models.builder")
    mmseg_apis_mod.set_random_seed = lambda *args, **kwargs: None
    mmseg_apis_mod.single_gpu_test = lambda *args, **kwargs: None
    mmseg_apis_mod.train_segmentor = lambda *args, **kwargs: None
    mmseg_datasets_mod.build_dataloader = lambda *args, **kwargs: None
    mmseg_datasets_mod.build_dataset = lambda *args, **kwargs: None
    mmseg_models_mod.build_segmentor = lambda *args, **kwargs: None
    mmseg_builder_mod.BACKBONES = mmseg_backbones

    modules = {
        "mmcv": mmcv_mod,
        "mmcv.parallel": mmcv_parallel_mod,
        "mmdet": mmdet_mod,
        "mmdet.apis": mmdet_apis_mod,
        "mmdet.datasets": mmdet_datasets_mod,
        "mmdet.models": mmdet_models_mod,
        "mmdet.models.builder": mmdet_builder_mod,
        "mmseg": mmseg_mod,
        "mmseg.apis": mmseg_apis_mod,
        "mmseg.datasets": mmseg_datasets_mod,
        "mmseg.models": mmseg_models_mod,
        "mmseg.models.builder": mmseg_builder_mod,
    }
    for name, module in modules.items():
        sys.modules[name] = module


def validate_config_builders(config_dir: Path) -> list[Path]:
    install_openmmlab_stubs()
    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    config_paths = sorted(config_dir.glob("detection_*.yaml")) + sorted(config_dir.glob("seg_*.yaml"))
    if not config_paths:
        raise AssertionError(f"No detection/segmentation configs found in {config_dir}")

    for config_path in config_paths:
        args = load_task_args(config_path)
        missing = [key for key in ("task", "model", "data_dir", "bs", "lr", "n_epochs", "size") if not hasattr(args, key)]
        if missing:
            raise AssertionError(f"{config_path}: missing required keys for train.py/task builders: {missing}")

        if args.task == "det":
            task = DetectionTask.__new__(DetectionTask)
            task.args = args
            task.run_name = task._build_run_name()
            cfg = task._build_mmdet_config()
            assert cfg.data["workers_per_gpu"] == 4, f"{config_path}: workers_per_gpu default should be 4"
            assert cfg.data["train"]["type"] == "CocoDataset", f"{config_path}: expected CocoDataset"
            assert cfg.model["type"] == "MaskRCNN", f"{config_path}: expected MaskRCNN"
        elif args.task == "seg":
            task = SegmentationTask.__new__(SegmentationTask)
            task.args = args
            task.run_name = task._build_run_name()
            cfg = task._build_mmseg_config()
            assert cfg.data["workers_per_gpu"] == 4, f"{config_path}: workers_per_gpu default should be 4"
            assert cfg.data["train"]["type"] == "ADE20KDataset", f"{config_path}: expected ADE20KDataset"
            assert cfg.model["type"] == "EncoderDecoder", f"{config_path}: expected EncoderDecoder"
        else:
            raise AssertionError(f"{config_path}: unexpected task {args.task!r}")

    return config_paths


def validate_backbones(require: bool = False) -> bool:
    missing = [name for name in ("torch", "timm", "einops") if not _module_available(name)]
    if missing:
        if require:
            raise RuntimeError(f"Missing dependencies for backbone smoke test: {missing}")
        print(f"SKIP backbone forward smoke test: missing dependencies {missing}")
        return False

    try:
        import torch
        from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone
    except Exception as exc:
        if require:
            raise
        print(f"SKIP backbone forward smoke test: {type(exc).__name__}: {exc}")
        return False

    expected_shapes = [(1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1)]
    cases = [
        ("ViTBackbone", ViTBackbone, {}),
        ("ViTCoPEBackbone_no_cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("ViTCoPEBackbone_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("ViTSCoPEBackbone_no_cls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("ViTSCoPEBackbone_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]
    for adapter_style in ("resize", "simple_fpn"):
        for name, cls, extra_kwargs in cases:
            model = cls(
                image_size=32,
                patch_size=16,
                dim=32,
                depth=4,
                heads=2,
                mlp_dim=64,
                dim_head=16,
                out_indices=(0, 1, 2, 3),
                fpn_adapter_style=adapter_style,
                **extra_kwargs,
            )
            model.eval()
            with torch.no_grad():
                outputs = model(torch.randn(1, 3, 32, 32))
            shapes = [tuple(output.shape) for output in outputs]
            assert shapes == expected_shapes, f"{name}/{adapter_style}: {shapes} != {expected_shapes}"
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate segmentation/detection wiring")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--skip-backbone", action="store_true", help="Skip real torch backbone smoke tests")
    parser.add_argument("--require-backbone", action="store_true", help="Fail if backbone smoke tests cannot run")
    args = parser.parse_args()

    config_paths = validate_config_builders(args.config_dir)
    print(f"OK config builders: {len(config_paths)} det/seg YAML files")

    if not args.skip_backbone:
        if validate_backbones(require=args.require_backbone):
            print("OK backbone forward smoke tests")

    print("OK segmentation/detection validation complete")


if __name__ == "__main__":
    main()
