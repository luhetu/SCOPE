#!/usr/bin/env python3
"""Smoke-check segmentation and detection task wiring without ML runtimes.

The full dense-prediction stack needs torch, mmcv-full CUDA ops, COCO/ADE20K,
and checkpoints. This checker intentionally validates the repo-owned wiring
that can break before any dataset is touched: YAML loading, task imports,
backbone registration, model config construction, and train/eval data config.
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


class _Config(dict):
    """Small stand-in for mmcv.Config used by task config builders."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _Registry:
    def __init__(self):
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        if module is None:
            def decorator(cls):
                self.module_dict[name or cls.__name__] = cls
                return cls

            return decorator

        if not force and name in self.module_dict:
            raise KeyError(f"{name} is already registered")
        self.module_dict[name] = module
        return module


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_task_stubs():
    """Install minimal modules so task files can be imported without torch/mmcv."""

    class _Cuda:
        @staticmethod
        def is_available():
            return False

    torch_mod = _module("torch", cuda=_Cuda())
    torch_nn = _module("torch.nn")
    torch_nn_functional = _module(
        "torch.nn.functional",
        interpolate=lambda *args, **kwargs: None,
    )
    torch_mod.nn = torch_nn
    torch_nn.functional = torch_nn_functional

    mmcv_mod = _module("mmcv", Config=_Config)
    mmcv_parallel = _module("mmcv.parallel", MMDataParallel=object)
    mmcv_mod.parallel = mmcv_parallel

    mmdet_backbones = _Registry()
    mmdet_builder = _module("mmdet.models.builder", BACKBONES=mmdet_backbones)
    mmdet_apis = _module(
        "mmdet.apis",
        set_random_seed=lambda *args, **kwargs: None,
        train_detector=lambda *args, **kwargs: None,
    )
    mmdet_datasets = _module("mmdet.datasets", build_dataset=lambda *args, **kwargs: None)
    mmdet_models = _module("mmdet.models", build_detector=lambda *args, **kwargs: None)
    mmdet_models.builder = mmdet_builder
    mmdet_mod = _module(
        "mmdet",
        apis=mmdet_apis,
        datasets=mmdet_datasets,
        models=mmdet_models,
    )

    mmseg_backbones = _Registry()
    mmseg_builder = _module("mmseg.models.builder", BACKBONES=mmseg_backbones)
    mmseg_apis = _module(
        "mmseg.apis",
        set_random_seed=lambda *args, **kwargs: None,
        single_gpu_test=lambda *args, **kwargs: None,
        train_segmentor=lambda *args, **kwargs: None,
    )
    mmseg_datasets = _module(
        "mmseg.datasets",
        build_dataloader=lambda *args, **kwargs: None,
        build_dataset=lambda *args, **kwargs: None,
    )
    mmseg_models = _module("mmseg.models", build_segmentor=lambda *args, **kwargs: None)
    mmseg_models.builder = mmseg_builder
    mmseg_mod = _module(
        "mmseg",
        apis=mmseg_apis,
        datasets=mmseg_datasets,
        models=mmseg_models,
    )

    class ViTBackbone:
        pass

    class ViTCoPEBackbone:
        pass

    class ViTSCoPEBackbone:
        pass

    backbone_mod = _module(
        "models.vit_backbone",
        ViTBackbone=ViTBackbone,
        ViTCoPEBackbone=ViTCoPEBackbone,
        ViTSCoPEBackbone=ViTSCoPEBackbone,
    )
    models_mod = _module("models", vit_backbone=backbone_mod)

    stubs = {
        "torch": torch_mod,
        "torch.nn": torch_nn,
        "torch.nn.functional": torch_nn_functional,
        "mmcv": mmcv_mod,
        "mmcv.parallel": mmcv_parallel,
        "mmdet": mmdet_mod,
        "mmdet.apis": mmdet_apis,
        "mmdet.datasets": mmdet_datasets,
        "mmdet.models": mmdet_models,
        "mmdet.models.builder": mmdet_builder,
        "mmseg": mmseg_mod,
        "mmseg.apis": mmseg_apis,
        "mmseg.datasets": mmseg_datasets,
        "mmseg.models": mmseg_models,
        "mmseg.models.builder": mmseg_builder,
        "models": models_mod,
        "models.vit_backbone": backbone_mod,
    }
    sys.modules.update(stubs)


def _load_yaml(path):
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data["cfg"] = str(path)
    return data


def _namespace(data):
    defaults = {
        "amp": False,
        "betas": None,
        "checkpoint_interval": 5000,
        "data_dir": "",
        "det_neck_type": "simple_fpn",
        "drop_path_rate": 0.0,
        "eval_interval": 2001,
        "final_eval": True,
        "img_scale": None,
        "layer_decay_rate": 1.0,
        "log_interval": 100,
        "min_lr": 0.0,
        "min_pretrained_match_rate": 80.0,
        "nowandb": True,
        "out_indices": (3, 5, 7, 11),
        "run_tag": None,
        "seed": None,
        "seg_aux_dim": 256,
        "seg_aux_in_index": 2,
        "seg_head_dim": 512,
        "seg_neck_style": "xcit_fpn",
        "seg_norm_type": "BN",
        "test_img_scale": None,
        "use_cls_token": None,
        "warmup_epochs": 0,
        "warmup_iters": None,
        "weight_decay": 0.01,
        "workers_per_gpu": 0,
    }
    merged = {**defaults, **data}
    return SimpleNamespace(**merged)


def _quiet_call(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def _iter_dense_configs(tasks):
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        data = _load_yaml(path)
        if data.get("task") in tasks:
            yield path, data


def _check_segmentation_config(path, data, errors, warnings):
    from tasks.segmentation import SegmentationTask

    args = _namespace(data)
    task = SegmentationTask.__new__(SegmentationTask)
    task.args = args
    task.run_name = task._build_run_name()
    cfg = _quiet_call(task._build_mmseg_config)

    model = cfg.model
    if model.get("type") != "EncoderDecoder":
        errors.append(f"{path}: expected EncoderDecoder, got {model.get('type')}")
    if model.get("decode_head", {}).get("type") != "UPerHead":
        errors.append(f"{path}: expected UPerHead decode head")
    if model.get("auxiliary_head", {}).get("type") != "FCNHead":
        errors.append(f"{path}: expected FCNHead auxiliary head")
    if cfg.data.train.get("type") != "ADE20KDataset":
        errors.append(f"{path}: train dataset is not ADE20KDataset")
    if cfg.data.val.get("ann_dir") != "annotations/validation":
        errors.append(f"{path}: validation annotations path is not ADE20K validation")
    if cfg.evaluation.get("metric") != "mIoU":
        errors.append(f"{path}: segmentation metric should be mIoU")

    backbone_type = model.get("backbone", {}).get("type")
    if args.model == "swin":
        expected = "SwinTransformer"
    else:
        expected = {
            "vit": "ViTBackbone",
            "vitcope": "ViTCoPEBackbone",
            "vitscope": "ViTSCoPEBackbone",
        }.get(args.model)
    if expected and backbone_type != expected:
        errors.append(f"{path}: expected backbone {expected}, got {backbone_type}")

    data_root = str(cfg.data.train.get("data_root", ""))
    if data_root.startswith("/home"):
        warnings.append(f"{path}: data_dir is machine-specific: {data_root}")


def _check_detection_config(path, data, errors, warnings):
    from tasks.detection import DetectionTask

    args = _namespace(data)
    task = DetectionTask.__new__(DetectionTask)
    task.args = args
    task.run_name = task._build_run_name()
    cfg = _quiet_call(task._build_mmdet_config)

    model = cfg.model
    if model.get("type") != "MaskRCNN":
        errors.append(f"{path}: expected MaskRCNN, got {model.get('type')}")
    if model.get("rpn_head", {}).get("type") != "RPNHead":
        errors.append(f"{path}: expected RPNHead")
    if model.get("roi_head", {}).get("mask_head", {}).get("type") != "FCNMaskHead":
        errors.append(f"{path}: expected FCNMaskHead mask head")
    if cfg.data.train.get("type") != "CocoDataset":
        errors.append(f"{path}: train dataset is not CocoDataset")
    if not str(cfg.data.val.get("ann_file", "")).endswith("instances_val2017.json"):
        errors.append(f"{path}: validation annotation file is not COCO val2017")
    if cfg.evaluation.get("metric") != ["bbox", "segm"]:
        errors.append(f"{path}: detection metrics should be ['bbox', 'segm']")

    neck_type = model.get("neck", {}).get("type")
    requested_neck = str(getattr(args, "det_neck_type", "simple_fpn")).lower()
    if requested_neck == "fpn" and neck_type != "FPN":
        errors.append(f"{path}: det_neck_type=fpn should build FPN, got {neck_type}")
    if requested_neck != "fpn" and neck_type != "SimpleFeaturePyramid":
        errors.append(f"{path}: expected SimpleFeaturePyramid, got {neck_type}")

    backbone = model.get("backbone", {})
    backbone_type = backbone.get("type")
    expected = {
        "vit": "ViTBackbone",
        "vitcope": "ViTCoPEBackbone",
        "vitscope": "ViTSCoPEBackbone",
        "swin": "SwinTransformer",
    }.get(args.model)
    if expected and backbone_type != expected:
        errors.append(f"{path}: expected backbone {expected}, got {backbone_type}")

    data_root = str(cfg.data.train.get("ann_file", ""))
    if data_root.startswith("/home"):
        warnings.append(f"{path}: data_dir is machine-specific: {args.data_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("seg", "det", "all"),
        default="all",
        help="Which dense task configs to verify.",
    )
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    _install_task_stubs()

    requested = {"seg", "det"} if args.task == "all" else {args.task}
    errors = []
    warnings = []
    counts = {"seg": 0, "det": 0}

    for path, data in _iter_dense_configs(requested):
        task = data.get("task")
        try:
            if task == "seg":
                _check_segmentation_config(path, data, errors, warnings)
                counts["seg"] += 1
            elif task == "det":
                _check_detection_config(path, data, errors, warnings)
                counts["det"] += 1
        except Exception as exc:  # noqa: BLE001 - report all config failures.
            errors.append(f"{path}: {type(exc).__name__}: {exc}")

    for warning in warnings:
        print(f"WARNING: {warning}")

    if errors:
        print("\nDense task smoke check FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1

    checked = ", ".join(f"{count} {task}" for task, count in counts.items() if task in requested)
    print(f"Dense task smoke check passed ({checked} configs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
