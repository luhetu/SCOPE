#!/usr/bin/env python3
"""Smoke-test segmentation and detection backbone wiring.

This validator is intentionally dataset-free.  It exercises the custom ViT,
ViT-CoPE, and ViT-SCoPE backbones with the feature-pyramid adapter styles used
by the segmentation and detection task builders.
"""

import argparse

import torch

from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone


BACKBONES = {
    "vit": (ViTBackbone, {}),
    "vitcope": (ViTCoPEBackbone, {"use_cls_token": False}),
    "vitscope": (ViTSCoPEBackbone, {}),
}

TASK_ADAPTERS = {
    "seg": "resize",
    "det": "simple_fpn",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run dataset-free segmentation/detection smoke checks."
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASK_ADAPTERS),
        default=None,
        help="Limit validation to one task. Defaults to both.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(BACKBONES),
        default=None,
        help="Limit validation to one backbone. Defaults to all.",
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--mlp-dim", type=int, default=96)
    parser.add_argument("--dim-head", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def expected_shapes(batch_size, dim, image_size, patch_size):
    base = image_size // patch_size
    return [
        (batch_size, dim, base * 4, base * 4),
        (batch_size, dim, base * 2, base * 2),
        (batch_size, dim, base, base),
        (batch_size, dim, base // 2, base // 2),
    ]


def validate_one(task, model_name, args):
    adapter_style = TASK_ADAPTERS[task]
    backbone_cls, extra_kwargs = BACKBONES[model_name]

    model = backbone_cls(
        image_size=args.image_size,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dim_head=args.dim_head,
        out_indices=(0, 1, 2, 3),
        fpn_adapter_style=adapter_style,
        **extra_kwargs,
    ).eval()

    image = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
    )
    with torch.no_grad():
        outputs = model(image)

    actual = [tuple(output.shape) for output in outputs]
    expected = expected_shapes(
        args.batch_size,
        args.dim,
        args.image_size,
        args.patch_size,
    )
    if actual != expected:
        raise AssertionError(
            f"{task}/{model_name} produced {actual}, expected {expected}"
        )

    print(f"[ok] {task}/{model_name}: {actual}")


def main():
    args = parse_args()

    if args.image_size % args.patch_size != 0:
        raise ValueError("--image-size must be divisible by --patch-size")

    tasks = [args.task] if args.task else sorted(TASK_ADAPTERS)
    models = [args.model] if args.model else sorted(BACKBONES)

    for task in tasks:
        for model_name in models:
            validate_one(task, model_name, args)

    print("Segmentation/detection smoke validation passed.")


if __name__ == "__main__":
    main()
