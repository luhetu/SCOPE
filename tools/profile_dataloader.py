#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataLoader bottleneck profiler for NFS environments.

Breaks down where time is actually spent during data loading:
  1. Raw NFS read speed  (open + read bytes, no decode)
  2. JPEG decode speed   (PIL open, no transform)
  3. Transform speed     (crop/flip/normalize, no IO)
  4. DataLoader throughput sweep across different num_workers

Usage:
    python tools/profile_dataloader.py \
        --data_dir /home3/dnrx52/vitcope \
        --split train \
        --bs 384 \
        --size 224 \
        --n_batches 20

    # Quick diagnosis (5 batches, fewer worker configs):
    python tools/profile_dataloader.py \
        --data_dir /home3/dnrx52/vitcope --quick
"""
import argparse
import os
import sys
import time
import random
import io
import statistics

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
from PIL import Image


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _find_split_dir(data_dir, split):
    """Locate train/val directory under data_dir."""
    candidates = [
        os.path.join(data_dir, split),
        os.path.join(data_dir, f"{split}1"),
        os.path.join(data_dir, f"{split}.X1"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    # fallback: first subdir
    subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))]
    if subdirs:
        print(f"[WARN] could not find '{split}' dir, using {subdirs[0]}")
        return subdirs[0]
    raise FileNotFoundError(f"No split dir found in {data_dir}")


def _collect_samples(split_dir, n=2000):
    """Collect up to n image paths from an ImageFolder-style split dir."""
    paths = []
    for cls in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                paths.append(os.path.join(cls_dir, fname))
        if len(paths) >= n:
            break
    random.shuffle(paths)
    return paths[:n]


def fmt(label, val, unit='ms', width=28):
    return f"  {label:<{width}} {val:>8.2f} {unit}"


# ------------------------------------------------------------------ #
# Stage 1: Raw NFS read (no decode, no transform)
# ------------------------------------------------------------------ #

def bench_raw_read(paths, n=500):
    """Measure raw NFS file open+read time per image."""
    paths = paths[:n]
    times = []
    total_bytes = 0
    for p in paths:
        t0 = time.perf_counter()
        with open(p, 'rb') as f:
            data = f.read()
        times.append((time.perf_counter() - t0) * 1000)
        total_bytes += len(data)

    median_ms = statistics.median(times)
    p95_ms    = sorted(times)[int(len(times) * 0.95)]
    mb_per_s  = (total_bytes / 1e6) / (sum(times) / 1000)
    avg_kb    = total_bytes / len(paths) / 1024

    print(f"\n{'='*55}")
    print(f"  Stage 1 — Raw NFS read  (n={len(paths)})")
    print(f"{'='*55}")
    print(fmt("median per file", median_ms))
    print(fmt("p95 per file",    p95_ms))
    print(fmt("avg file size",   avg_kb, 'KB'))
    print(fmt("throughput",      mb_per_s, 'MB/s'))
    print(f"  {'NFS bottleneck?':<28} {'YES – NFS latency is the root cause' if median_ms > 5 else 'No – NFS reads are fast'}")
    return median_ms


# ------------------------------------------------------------------ #
# Stage 2: JPEG decode only (file already read into bytes)
# ------------------------------------------------------------------ #

def bench_decode(paths, n=500):
    """Measure PIL JPEG decode time (data already in memory)."""
    paths = paths[:n]
    # Pre-read all files into memory
    raw_list = []
    for p in paths:
        with open(p, 'rb') as f:
            raw_list.append(f.read())

    times = []
    for raw in raw_list:
        t0 = time.perf_counter()
        img = Image.open(io.BytesIO(raw)).convert('RGB')
        _ = img.size  # force decode
        times.append((time.perf_counter() - t0) * 1000)

    median_ms = statistics.median(times)
    p95_ms    = sorted(times)[int(len(times) * 0.95)]
    imgs_per_s = len(paths) / (sum(times) / 1000)

    print(f"\n{'='*55}")
    print(f"  Stage 2 — JPEG decode only  (n={len(paths)})")
    print(f"{'='*55}")
    print(fmt("median per image", median_ms))
    print(fmt("p95 per image",    p95_ms))
    print(fmt("throughput",       imgs_per_s, 'img/s'))
    print(f"  {'Decode bottleneck?':<28} {'YES' if median_ms > 3 else 'No – decode is fast'}")
    return median_ms


# ------------------------------------------------------------------ #
# Stage 3: Transform only (image already decoded)
# ------------------------------------------------------------------ #

def bench_transform(paths, size, n=500):
    """Measure torchvision transform time (image in memory, no IO)."""
    paths = paths[:n]
    tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.08, 1.0), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Pre-decode
    imgs = []
    for p in paths:
        with open(p, 'rb') as f:
            imgs.append(Image.open(io.BytesIO(f.read())).convert('RGB'))

    times = []
    for img in imgs:
        t0 = time.perf_counter()
        _ = tf(img)
        times.append((time.perf_counter() - t0) * 1000)

    median_ms = statistics.median(times)
    p95_ms    = sorted(times)[int(len(times) * 0.95)]
    imgs_per_s = len(imgs) / (sum(times) / 1000)

    print(f"\n{'='*55}")
    print(f"  Stage 3 — Transform only  (n={len(paths)})")
    print(f"{'='*55}")
    print(fmt("median per image", median_ms))
    print(fmt("p95 per image",    p95_ms))
    print(fmt("throughput",       imgs_per_s, 'img/s'))
    return median_ms


# ------------------------------------------------------------------ #
# Stage 4: Full DataLoader throughput sweep (different num_workers)
# ------------------------------------------------------------------ #

def bench_dataloader(split_dir, size, bs, worker_list, n_batches):
    """Measure DataLoader batches/sec and images/sec for each worker count."""
    tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.08, 1.0), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolder(split_dir, transform=tf)

    print(f"\n{'='*55}")
    print(f"  Stage 4 — DataLoader throughput sweep")
    print(f"  dataset: {len(dataset)} images | bs={bs} | n_batches={n_batches}")
    print(f"{'='*55}")
    print(f"  {'workers':<10} {'img/s':>10} {'ms/batch':>12} {'gpu_stall_est':>16}")
    print(f"  {'-'*50}")

    results = {}
    for nw in worker_list:
        loader = DataLoader(
            dataset, batch_size=bs, shuffle=True,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=(nw > 0),
            prefetch_factor=4 if nw > 0 else None,
        )
        it = iter(loader)

        # Warmup 2 batches
        for _ in range(min(2, n_batches)):
            try:
                next(it)
            except StopIteration:
                break

        # Measure
        batch_times = []
        t_start = time.perf_counter()
        for _ in range(n_batches):
            t0 = time.perf_counter()
            try:
                imgs, _ = next(it)
            except StopIteration:
                break
            batch_times.append((time.perf_counter() - t0) * 1000)

        if not batch_times:
            continue

        avg_ms     = statistics.mean(batch_times)
        median_ms  = statistics.median(batch_times)
        imgs_per_s = bs / (avg_ms / 1000)

        # GPU stall estimate: if GPU compute ~200ms/batch,
        # how long does loader stall it per batch?
        gpu_compute_ms = 200.0
        stall_ms = max(0, avg_ms - gpu_compute_ms)

        print(f"  {nw:<10} {imgs_per_s:>10.0f} {avg_ms:>12.1f} {stall_ms:>14.1f}ms")
        results[nw] = avg_ms

        del loader

    return results


# ------------------------------------------------------------------ #
# Stage 5: Diagnose worker count vs CPU cores
# ------------------------------------------------------------------ #

def print_system_info(data_dir):
    import platform
    print(f"\n{'='*55}")
    print(f"  System Info")
    print(f"{'='*55}")
    print(f"  hostname        : {platform.node()}")
    try:
        import multiprocessing
        print(f"  CPU cores       : {multiprocessing.cpu_count()}")
    except Exception:
        pass
    print(f"  data_dir        : {data_dir}")

    # Check if data_dir is on NFS
    try:
        result = os.popen(f"df -T {data_dir} 2>/dev/null | tail -1").read().strip()
        print(f"  df -T output    : {result}")
        if 'nfs' in result.lower():
            print(f"  ⚠️  Filesystem   : NFS — I/O bottleneck likely")
        elif 'ext4' in result.lower() or 'xfs' in result.lower() or 'nvme' in result.lower():
            print(f"  ✅ Filesystem   : Local disk")
        else:
            print(f"  Filesystem      : {result.split()[1] if len(result.split()) > 1 else 'unknown'}")
    except Exception:
        pass

    try:
        result = os.popen(f"df -h {data_dir} 2>/dev/null | tail -1").read().strip()
        print(f"  disk usage      : {result}")
    except Exception:
        pass


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="DataLoader bottleneck profiler")
    parser.add_argument("--data_dir", type=str, default="/home3/dnrx52/vitcope")
    parser.add_argument("--split",    type=str, default="train",
                        choices=["train", "val"])
    parser.add_argument("--bs",       type=int, default=384)
    parser.add_argument("--size",     type=int, default=224)
    parser.add_argument("--n_batches",type=int, default=20,
                        help="Batches to time in loader sweep")
    parser.add_argument("--n_files",  type=int, default=500,
                        help="Files to sample for stage 1-3")
    parser.add_argument("--workers",  type=str, default="0,2,4,8,12,16",
                        help="Comma-separated worker counts to test")
    parser.add_argument("--quick",    action="store_true",
                        help="Quick mode: fewer samples, fewer worker configs")
    args = parser.parse_args()

    if args.quick:
        args.n_files   = 200
        args.n_batches = 5
        args.workers   = "0,4,8"

    worker_list = [int(w) for w in args.workers.split(",")]

    print_system_info(args.data_dir)

    split_dir = _find_split_dir(args.data_dir, args.split)
    print(f"\n  split_dir: {split_dir}")

    paths = _collect_samples(split_dir, n=args.n_files)
    print(f"  sampled {len(paths)} image paths")

    t_read   = bench_raw_read(paths,        n=args.n_files)
    t_decode = bench_decode(paths,          n=args.n_files)
    t_tf     = bench_transform(paths, args.size, n=min(200, args.n_files))
    results  = bench_dataloader(split_dir, args.size, args.bs, worker_list, args.n_batches)

    # Summary
    print(f"\n{'='*55}")
    print(f"  SUMMARY — bottleneck diagnosis")
    print(f"{'='*55}")

    total_serial_ms = t_read + t_decode + t_tf
    print(f"  Per-image cost (single-threaded):")
    print(f"    NFS read   : {t_read:.1f} ms  ({100*t_read/total_serial_ms:.0f}%)")
    print(f"    JPEG decode: {t_decode:.1f} ms  ({100*t_decode/total_serial_ms:.0f}%)")
    print(f"    Transform  : {t_tf:.1f} ms  ({100*t_tf/total_serial_ms:.0f}%)")

    if t_read > t_decode * 3:
        dominant = "NFS read latency"
        fix = "→ rsync to local disk ($TMPDIR) before training"
    elif t_decode > t_tf * 2:
        dominant = "JPEG decode (CPU-bound)"
        fix = "→ increase num_workers, or use pre-decoded cache"
    else:
        dominant = "Transform / augmentation (CPU-bound)"
        fix = "→ increase num_workers or simplify augmentation"

    print(f"\n  Dominant bottleneck : {dominant}")
    print(f"  Recommended fix     : {fix}")

    if results:
        best_nw = min(results, key=results.get)
        print(f"\n  Best num_workers in this test: {best_nw} → {results[best_nw]:.1f} ms/batch")

    print()


if __name__ == "__main__":
    main()
