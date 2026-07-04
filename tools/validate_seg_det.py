#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight validation for segmentation/detection configuration code.

This script avoids the compiled mmcv-full extension by installing tiny stubs for
the APIs that task config builders import. It validates the Python configuration
paths and CPU forwards for the custom dense-prediction backbones.
"""

import argparse
import glob
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
                raise KeyError(f"{key} already registered")
            self.module_dict[key] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


def _install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.parallel"] = mmcv_parallel

    mmseg_backbones = Registry()
    mmseg = types.ModuleType("mmseg")
    mmseg_apis = types.ModuleType("mmseg.apis")
    mmseg_apis.set_random_seed = lambda *args, **kwargs: None
    mmseg_apis.single_gpu_test = lambda *args, **kwargs: []
    mmseg_apis.train_segmentor = lambda *args, **kwargs: None
    mmseg_datasets = types.ModuleType("mmseg.datasets")
    mmseg_datasets.build_dataloader = lambda *args, **kwargs: None
    mmseg_datasets.build_dataset = lambda *args, **kwargs: None
    mmseg_models = types.ModuleType("mmseg.models")
    mmseg_models.build_segmentor = lambda *args, **kwargs: None
    mmseg_builder = types.ModuleType("mmseg.models.builder")
    mmseg_builder.BACKBONES = mmseg_backbones
    mmseg_models.builder = mmseg_builder
    sys.modules["mmseg"] = mmseg
    sys.modules["mmseg.apis"] = mmseg_apis
    sys.modules["mmseg.datasets"] = mmseg_datasets
    sys.modules["mmseg.models"] = mmseg_models
    sys.modules["mmseg.models.builder"] = mmseg_builder

    mmdet_backbones = Registry()
    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = lambda *args, **kwargs: None
    mmdet_apis.train_detector = lambda *args, **kwargs: None
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = lambda *args, **kwargs: None
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = lambda *args, **kwargs: None
    mmdet_builder = types.ModuleType("mmdet.models.builder")
    mmdet_builder.BACKBONES = mmdet_backbones
    mmdet_models.builder = mmdet_builder
    sys.modules["mmdet"] = mmdet
    sys.modules["mmdet.apis"] = mmdet_apis
    sys.modules["mmdet.datasets"] = mmdet_datasets
    sys.modules["mmdet.models"] = mmdet_models
    sys.modules["mmdet.models.builder"] = mmdet_builder


DEFAULTS = dict(
    cfg="",
    task="",
    model="vit",
    data_dir="/tmp/data",
    bs=1,
    workers_per_gpu=None,
    size=224,
    patch=16,
    dim=192,
    depth=12,
    heads=3,
    mlp_dim=768,
    dim_head=64,
    out_indices=(3, 5, 7, 11),
    embed_dim=96,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    window_size=7,
    n_epochs=1,
    max_iters=None,
    warmup_epochs=0,
    warmup_iters=None,
    lr=1e-4,
    min_lr=0.0,
    weight_decay=0.01,
    layer_decay_rate=1.0,
    drop_path_rate=0.0,
    amp=False,
    nowandb=True,
    pretrained=None,
    seed=None,
    run_tag=None,
    crop_size=512,
    img_scale=(2048, 512),
    test_img_scale=None,
    seg_head_dim=None,
    seg_aux_dim=None,
    seg_aux_in_index=2,
    seg_norm_type="SyncBN",
    seg_neck_style="xcit_fpn",
    det_neck_type="fpn",
    checkpoint_interval=5000,
    eval_interval=2001,
    log_interval=100,
    final_eval=True,
    use_cls_token=False,
)


def _load_args(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    merged = dict(DEFAULTS)
    merged.update(data)
    merged["cfg"] = str(path)
    return SimpleNamespace(**merged)


def _validate_segmentation_configs(paths):
    from tasks.segmentation import SegmentationTask

    for path in paths:
        args = _load_args(path)
        task = object.__new__(SegmentationTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmseg_config()

        assert cfg.model["type"] == "EncoderDecoder", path
        assert cfg.data["workers_per_gpu"] == 4, path
        assert cfg.data["train"]["pipeline"], path
        assert cfg.data["val"]["pipeline"], path
        if args.model == "swin":
            assert cfg.model["backbone"]["type"] == "SwinTransformer", path
            assert cfg.model["decode_head"]["in_channels"] == [96, 192, 384, 768], path
        else:
            assert cfg.model["backbone"]["type"] in {"ViTBackbone", "ViTCoPEBackbone", "ViTSCoPEBackbone"}, path
            assert cfg.model["decode_head"]["in_channels"] == [args.dim] * 4, path


def _validate_detection_configs(paths):
    from tasks.detection import DetectionTask

    for path in paths:
        args = _load_args(path)
        task = object.__new__(DetectionTask)
        task.args = args
        task.run_name = task._build_run_name()
        cfg = task._build_mmdet_config()

        assert cfg.model["type"] == "MaskRCNN", path
        assert cfg.data["workers_per_gpu"] == 4, path
        assert cfg.data["train"]["pipeline"], path
        assert cfg.data["val"]["pipeline"], path
        if args.model == "swin":
            assert cfg.model["backbone"]["type"] == "SwinTransformer", path
            assert cfg.model["neck"]["in_channels"] == [96, 192, 384, 768], path
        else:
            assert cfg.model["backbone"]["type"] in {"ViTBackbone", "ViTCoPEBackbone", "ViTSCoPEBackbone"}, path
            assert cfg.model["neck"]["in_channels"] == [args.dim] * 4, path


def _validate_backbone_forward():
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
    cases = [
        ("vit", ViTBackbone, {}),
        ("vitcope_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("vitcope_nocls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("vitscope_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
        ("vitscope_nocls", ViTSCoPEBackbone, {"use_cls_token": False}),
    ]
    expected_hw = [(8, 8), (4, 4), (2, 2), (1, 1)]

    torch.manual_seed(0)
    sample = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        for style in ("resize", "simple_fpn"):
            for name, cls, extra in cases:
                model = cls(**common, fpn_adapter_style=style, **extra).eval()
                outputs = model(sample)
                assert len(outputs) == 4, (name, style)
                for output, hw in zip(outputs, expected_hw):
                    assert tuple(output.shape) == (1, 32, hw[0], hw[1]), (name, style, output.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-dir", default=str(ROOT / "configs"))
    args = parser.parse_args()

    _install_openmmlab_stubs()
    configs_dir = Path(args.configs_dir)
    seg_configs = sorted(glob.glob(str(configs_dir / "seg_*.yaml")))
    det_configs = sorted(glob.glob(str(configs_dir / "detection_*.yaml")))
    if not seg_configs or not det_configs:
        raise RuntimeError("Expected both segmentation and detection configs")

    _validate_segmentation_configs(seg_configs)
    print(f"validated {len(seg_configs)} segmentation configs")
    _validate_detection_configs(det_configs)
    print(f"validated {len(det_configs)} detection configs")
    _validate_backbone_forward()
    print("validated custom backbone CPU forwards")


if __name__ == "__main__":
    main()
