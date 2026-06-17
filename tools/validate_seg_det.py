#!/usr/bin/env python3
"""Smoke-test segmentation and detection config builders without mmcv-full."""

import argparse
import sys
import types
from pathlib import Path


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
            module_name = name or cls.__name__
            if force or module_name not in self.module_dict:
                self.module_dict[module_name] = cls
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
    mmcv.parallel = mmcv_parallel

    mmseg_backbones = Registry()
    mmseg_builder = types.ModuleType("mmseg.models.builder")
    mmseg_builder.BACKBONES = mmseg_backbones

    mmseg_models = types.ModuleType("mmseg.models")
    mmseg_models.builder = mmseg_builder
    mmseg_models.build_segmentor = _noop

    mmseg_apis = types.ModuleType("mmseg.apis")
    mmseg_apis.set_random_seed = _noop
    mmseg_apis.single_gpu_test = _noop
    mmseg_apis.train_segmentor = _noop

    mmseg_datasets = types.ModuleType("mmseg.datasets")
    mmseg_datasets.build_dataloader = _noop
    mmseg_datasets.build_dataset = _noop

    mmseg = types.ModuleType("mmseg")
    mmseg.apis = mmseg_apis
    mmseg.datasets = mmseg_datasets
    mmseg.models = mmseg_models

    mmdet_backbones = Registry()
    mmdet_builder = types.ModuleType("mmdet.models.builder")
    mmdet_builder.BACKBONES = mmdet_backbones

    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.builder = mmdet_builder
    mmdet_models.build_detector = _noop

    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = _noop
    mmdet_apis.train_detector = _noop

    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = _noop

    mmdet = types.ModuleType("mmdet")
    mmdet.apis = mmdet_apis
    mmdet.datasets = mmdet_datasets
    mmdet.models = mmdet_models

    sys.modules.update({
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
    })


def _make_parser():
    parser = argparse.ArgumentParser(description="seg/det validation parser")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--workers_per_gpu", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--time_profile", action="store_true")
    parser.add_argument("--time_profile_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_interval", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--final_eval", default=None)
    parser.add_argument("--seg_head_dim", type=int, default=None)
    parser.add_argument("--seg_aux_dim", type=int, default=None)
    parser.add_argument("--seg_aux_in_index", type=int, default=None)
    parser.add_argument("--seg_neck_dim", type=int, default=None)
    parser.add_argument("--seg_neck_style", type=str, default=None)
    parser.add_argument("--seg_norm_type", type=str, default=None)
    parser.add_argument("--det_neck_type", type=str, default=None)
    parser.add_argument("--backbone_size", default=None)
    parser.add_argument("--crop_size", default=None)
    parser.add_argument("--img_scale", default=None)
    parser.add_argument("--test_img_scale", default=None)
    parser.add_argument("--warmup_iters", type=int, default=None)
    parser.add_argument("--warmup_epochs", type=float, default=None)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--layer_decay_rate", type=float, default=None)
    parser.add_argument("--drop_path_rate", type=float, default=None)
    parser.add_argument("--dim_head", type=int, default=None)
    parser.add_argument("--out_indices", default=None)
    parser.add_argument("--use_cls_token", default=None)
    parser.add_argument("--betas", default=None)
    return parser


def _load_args(cfg_path):
    from utils.cfg import load_cfg

    old_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], "--cfg", str(cfg_path)]
        args = load_cfg(_make_parser())
    finally:
        sys.argv = old_argv
    return args


def _get(args, name, default=None):
    value = getattr(args, name, default)
    return default if value is None else value


def _normalize_train_schedule(args):
    if args.task == "seg":
        iters_per_epoch = 20210 // int(args.bs)
        max_iters = _get(args, "max_iters")
        args.max_iters = int(max_iters) if max_iters is not None else int(args.n_epochs * iters_per_epoch)
        warmup_iters = _get(args, "warmup_iters")
        warmup_epochs = _get(args, "warmup_epochs", 0)
        if warmup_iters is not None:
            args.warmup_iters = int(warmup_iters)
        elif warmup_epochs > 0:
            args.warmup_iters = int(warmup_epochs * iters_per_epoch)
        else:
            args.warmup_iters = 1500
        args.iters_per_epoch = iters_per_epoch
    elif args.task == "det":
        iters_per_epoch = 118287 // int(args.bs)
        warmup_iters = _get(args, "warmup_iters")
        warmup_epochs = _get(args, "warmup_epochs", 0)
        if warmup_iters is not None:
            args.warmup_iters = int(warmup_iters)
        elif warmup_epochs > 0:
            args.warmup_iters = int(warmup_epochs * iters_per_epoch)
        else:
            args.warmup_iters = 500
        args.iters_per_epoch = iters_per_epoch


def _assert_train_metadata(args, cfg_path):
    required = ("task", "model", "data_dir", "bs", "lr", "n_epochs", "size", "patch")
    missing = [name for name in required if not hasattr(args, name) or getattr(args, name) is None]
    if missing:
        raise AssertionError(f"{cfg_path} missing train.py metadata fields: {missing}")


def _build_task_config(task_cls, args, build_method):
    task = task_cls.__new__(task_cls)
    task.args = args
    task.device = "cpu"
    task.run_name = task._build_run_name()
    return getattr(task, build_method)()


def _validate_segmentation_configs(paths):
    from tasks.segmentation import SegmentationTask

    for path in paths:
        args = _load_args(path)
        _assert_train_metadata(args, path)
        _normalize_train_schedule(args)
        cfg = _build_task_config(SegmentationTask, args, "_build_mmseg_config")
        assert cfg.model["type"] == "EncoderDecoder"
        assert cfg.data["workers_per_gpu"] == 4
        assert cfg.model["backbone"]["type"] in {"ViTBackbone", "ViTCoPEBackbone", "ViTSCoPEBackbone", "SwinTransformer"}
    print(f"Validated {len(paths)} segmentation configs")


def _validate_detection_configs(paths):
    from tasks.detection import DetectionTask

    for path in paths:
        args = _load_args(path)
        _assert_train_metadata(args, path)
        _normalize_train_schedule(args)
        cfg = _build_task_config(DetectionTask, args, "_build_mmdet_config")
        assert cfg.model["type"] == "MaskRCNN"
        assert cfg.data["workers_per_gpu"] == 4
        assert cfg.model["backbone"]["type"] in {"ViTBackbone", "ViTCoPEBackbone", "ViTSCoPEBackbone", "SwinTransformer"}
    print(f"Validated {len(paths)} detection configs")


def _validate_backbone_forwards():
    import torch
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    torch.manual_seed(0)
    x = torch.randn(1, 3, 32, 32)
    cases = [
        ("ViTBackbone", ViTBackbone, {}),
        ("ViTCoPEBackbone", ViTCoPEBackbone, {"use_cls_token": False}),
        ("ViTCoPEBackbone_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("ViTSCoPEBackbone", ViTSCoPEBackbone, {"use_cls_token": True}),
        ("ViTSCoPEBackbone_no_cls", ViTSCoPEBackbone, {"use_cls_token": False}),
    ]
    expected_hw = [(8, 8), (4, 4), (2, 2), (1, 1)]

    checked = 0
    for style in ("resize", "simple_fpn"):
        for name, cls, extra in cases:
            model = cls(
                image_size=32,
                patch_size=16,
                dim=32,
                depth=4,
                heads=4,
                mlp_dim=64,
                dim_head=8,
                out_indices=(0, 1, 2, 3),
                fpn_adapter_style=style,
                **extra,
            )
            model.eval()
            with torch.no_grad():
                outputs = model(x)
            assert len(outputs) == 4, f"{name}/{style} returned {len(outputs)} outputs"
            for idx, (out, hw) in enumerate(zip(outputs, expected_hw)):
                expected = (1, 32, hw[0], hw[1])
                assert tuple(out.shape) == expected, f"{name}/{style} output {idx}: {tuple(out.shape)} != {expected}"
            checked += 1
    print(f"Validated {checked} custom backbone CPU forward cases")


def main():
    _install_openmmlab_stubs()
    seg_configs = sorted((ROOT / "configs").glob("seg*.yaml"))
    det_configs = sorted((ROOT / "configs").glob("detection_*.yaml"))
    if not seg_configs or not det_configs:
        raise SystemExit("No segmentation or detection configs found")
    _validate_segmentation_configs(seg_configs)
    _validate_detection_configs(det_configs)
    _validate_backbone_forwards()
    print("Segmentation/detection smoke validation passed")


if __name__ == "__main__":
    main()
