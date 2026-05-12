#!/usr/bin/env python3
"""Smoke-check segmentation/detection config builders and ViT backbones.

The full MMSegmentation/MMDetection trainers require datasets and compiled MMCV
ops. This script validates the repo-local task glue in a lightweight CPU path:
it stubs only the external OpenMMLab builders, exercises every seg/det YAML
config, and runs tiny forward passes through the custom dense-prediction
backbones.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml


class Config(dict):
    """Small stand-in for mmcv.Config used by task config builders."""

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
            module_name = name or cls.__name__
            if not force and module_name in self.module_dict:
                raise KeyError(f"{module_name} is already registered")
            self.module_dict[module_name] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


def install_openmmlab_stubs():
    """Install minimal modules needed to import task files."""

    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv.parallel = types.ModuleType("mmcv.parallel")
    mmcv.parallel.MMDataParallel = object
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.parallel"] = mmcv.parallel

    for root_name, build_model_name, train_name in (
        ("mmseg", "build_segmentor", "train_segmentor"),
        ("mmdet", "build_detector", "train_detector"),
    ):
        root = types.ModuleType(root_name)
        apis = types.ModuleType(f"{root_name}.apis")
        apis.set_random_seed = lambda *args, **kwargs: None
        setattr(apis, train_name, lambda *args, **kwargs: None)
        if root_name == "mmseg":
            apis.single_gpu_test = lambda *args, **kwargs: []

        datasets = types.ModuleType(f"{root_name}.datasets")
        datasets.build_dataset = lambda *args, **kwargs: None
        if root_name == "mmseg":
            datasets.build_dataloader = lambda *args, **kwargs: None

        models = types.ModuleType(f"{root_name}.models")
        setattr(models, build_model_name, lambda *args, **kwargs: None)
        builder = types.ModuleType(f"{root_name}.models.builder")
        builder.BACKBONES = Registry()
        models.builder = builder

        root.apis = apis
        root.datasets = datasets
        root.models = models
        sys.modules[root_name] = root
        sys.modules[f"{root_name}.apis"] = apis
        sys.modules[f"{root_name}.datasets"] = datasets
        sys.modules[f"{root_name}.models"] = models
        sys.modules[f"{root_name}.models.builder"] = builder


def coerce_cfg_types(cfg):
    int_keys = {
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
    float_keys = {
        "lr",
        "min_lr",
        "warmup_epochs",
        "drop_path_rate",
        "weight_decay",
        "dropout",
        "emb_dropout",
        "layer_decay_rate",
    }
    bool_keys = {"amp", "aug", "nowandb", "use_cls_token"}

    for key, value in list(cfg.items()):
        if key in int_keys and value is not None:
            cfg[key] = int(value)
        elif key in float_keys and value is not None:
            cfg[key] = float(value)
        elif key in bool_keys and value is not None:
            cfg[key] = bool(value)
        elif key == "pretrained" and value == "null":
            cfg[key] = None
    return cfg


def args_from_yaml(path):
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg = coerce_cfg_types(cfg)
    cfg.setdefault("cfg", str(path))
    cfg.setdefault("resume", "")
    cfg.setdefault("workers_per_gpu", None)
    cfg.setdefault("run_tag", None)
    cfg.setdefault("data_dir", "")
    return SimpleNamespace(**cfg)


def instantiate_for_config(task_cls, args, build_method_name):
    task = object.__new__(task_cls)
    task.args = args
    task.device = "cpu"
    task.run_name = task._build_run_name()
    return getattr(task, build_method_name)()


def validate_task_configs(paths):
    segmentation = importlib.import_module("tasks.segmentation")
    detection = importlib.import_module("tasks.detection")

    checked = []
    for path in paths:
        args = args_from_yaml(path)
        if args.task == "seg":
            cfg = instantiate_for_config(
                segmentation.SegmentationTask,
                args,
                "_build_mmseg_config",
            )
            assert cfg.model["type"] == "EncoderDecoder"
            assert cfg.data["train"]["type"] == "ADE20KDataset"
            assert cfg.model["decode_head"]["num_classes"] == 150
        elif args.task == "det":
            cfg = instantiate_for_config(
                detection.DetectionTask,
                args,
                "_build_mmdet_config",
            )
            assert cfg.model["type"] == "MaskRCNN"
            assert cfg.data["train"]["type"] == "CocoDataset"
            assert cfg.model["roi_head"]["bbox_head"]["num_classes"] == 80
        else:
            raise AssertionError(f"{path}: unsupported task {args.task!r}")
        checked.append(path)
    return checked


def validate_backbone_forward():
    import torch

    vit_backbone = importlib.import_module("models.vit_backbone")
    tiny_kwargs = dict(
        image_size=32,
        patch_size=16,
        dim=32,
        depth=4,
        heads=2,
        mlp_dim=64,
        dim_head=16,
        out_indices=(0, 1, 2, 3),
    )
    expected_shapes = {
        "identity": [(1, 32, 2, 2)] * 4,
        "resize": [(1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1)],
        "simple_fpn": [(1, 32, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2), (1, 32, 1, 1)],
    }

    cases = [
        ("ViTBackbone", vit_backbone.ViTBackbone, {}),
        ("ViTCoPEBackbone", vit_backbone.ViTCoPEBackbone, {"use_cls_token": False}),
        ("ViTCoPEBackbone_cls", vit_backbone.ViTCoPEBackbone, {"use_cls_token": True}),
        ("ViTSCoPEBackbone", vit_backbone.ViTSCoPEBackbone, {}),
    ]
    image = torch.randn(1, 3, 32, 32)
    checked = []

    with torch.no_grad():
        for style, shapes in expected_shapes.items():
            for name, cls, extra_kwargs in cases:
                model = cls(**tiny_kwargs, fpn_adapter_style=style, **extra_kwargs)
                model.eval()
                outputs = model(image)
                actual = [tuple(output.shape) for output in outputs]
                if actual != shapes:
                    raise AssertionError(
                        f"{name} style={style}: expected {shapes}, got {actual}"
                    )
                checked.append(f"{name}:{style}")
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="*",
        type=Path,
        default=None,
        help="Specific seg/det YAML configs to validate.",
    )
    parser.add_argument(
        "--skip-backbone-forward",
        action="store_true",
        help="Only validate task config construction.",
    )
    args = parser.parse_args()

    install_openmmlab_stubs()
    if args.configs:
        paths = args.configs
    else:
        paths = sorted(Path("configs").glob("seg_*.yaml"))
        paths.extend(sorted(Path("configs").glob("detection_*.yaml")))

    config_results = validate_task_configs(paths)
    print(f"Validated {len(config_results)} segmentation/detection configs.")

    if not args.skip_backbone_forward:
        backbone_results = validate_backbone_forward()
        print(f"Validated {len(backbone_results)} custom backbone forward cases.")

    print("Segmentation/detection smoke validation passed.")


if __name__ == "__main__":
    main()
