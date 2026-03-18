#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert ImageNet-style directory (train/val) to LMDB format.

Usage:
    python tools/make_lmdb.py \
        --src /home3/dnrx52/vitcope/train \
        --dst /home3/dnrx52/vitcope_lmdb/train \
        --num_workers 8

    python tools/make_lmdb.py \
        --src /home3/dnrx52/vitcope/val \
        --dst /home3/dnrx52/vitcope_lmdb/val \
        --num_workers 8
"""
import argparse
import os
import sys
import lmdb
import pickle
import numpy as np
from multiprocessing import Pool
from torchvision.datasets import ImageFolder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True,
                        help='Source ImageFolder directory (e.g. .../train)')
    parser.add_argument('--dst', type=str, required=True,
                        help='Output LMDB directory')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--map_size_gb', type=float, default=200.0,
                        help='Max LMDB map size in GB (default 200GB, safe for ImageNet-1K train)')
    return parser.parse_args()


def read_image_bytes(path):
    """Read raw bytes of one image file."""
    with open(path, 'rb') as f:
        return f.read()


def worker(item):
    """Worker: return (idx_str, label, raw_bytes)."""
    idx, (path, label) = item
    try:
        raw = read_image_bytes(path)
        return idx, label, raw
    except Exception as e:
        print(f'[WARN] skip {path}: {e}', flush=True)
        return idx, label, None


def main():
    args = parse_args()
    os.makedirs(args.dst, exist_ok=True)

    print(f'[make_lmdb] Scanning: {args.src}')
    dataset = ImageFolder(args.src)
    samples = dataset.samples          # list of (path, label)
    n = len(samples)
    print(f'[make_lmdb] Total samples: {n}')
    print(f'[make_lmdb] Classes: {len(dataset.classes)}')

    # Save class_to_idx mapping alongside lmdb
    meta = {
        'classes': dataset.classes,
        'class_to_idx': dataset.class_to_idx,
        'n': n,
    }
    meta_path = os.path.join(args.dst, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f'[make_lmdb] Saved meta -> {meta_path}')

    map_size = int(args.map_size_gb * 1024 ** 3)
    env = lmdb.open(args.dst, map_size=map_size, subdir=True, readonly=False,
                    meminit=False, map_async=True)

    chunk = 1000
    with Pool(args.num_workers) as pool:
        items = list(enumerate(samples))
        done = 0
        for start in range(0, n, chunk):
            batch = items[start:start + chunk]
            results = pool.map(worker, batch)

            with env.begin(write=True) as txn:
                for idx, label, raw in results:
                    if raw is None:
                        continue
                    key = f'{idx:08d}'.encode()
                    value = pickle.dumps({'label': label, 'raw': raw})
                    txn.put(key, value)

            done += len(batch)
            if done % 10000 == 0 or done >= n:
                print(f'[make_lmdb] {done}/{n} ({100*done/n:.1f}%)', flush=True)

    env.sync()
    env.close()
    print(f'[make_lmdb] Done! LMDB saved to: {args.dst}')


if __name__ == '__main__':
    main()
