#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LMDB-backed ImageNet dataset.
Drop-in replacement for torchvision.datasets.ImageFolder.
"""
import io
import os
import pickle
import lmdb
import torch
from torch.utils.data import Dataset
from PIL import Image


class LMDBImageDataset(Dataset):
    """
    Read images from an LMDB file created by tools/make_lmdb.py.

    Args:
        lmdb_dir (str): Path to LMDB directory (contains data.mdb + meta.pkl)
        transform: torchvision transforms to apply
    """
    def __init__(self, lmdb_dir, transform=None):
        self.lmdb_dir = lmdb_dir
        self.transform = transform

        # Load metadata
        meta_path = os.path.join(lmdb_dir, 'meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        self.classes = meta['classes']
        self.class_to_idx = meta['class_to_idx']
        self.n = meta['n']

        # Open LMDB (read-only, lock=False for multi-worker)
        self._env = None  # lazy open per-worker

    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_dir,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        return self._env

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        env = self._get_env()
        key = f'{idx:08d}'.encode()
        with env.begin(write=False) as txn:
            value = txn.get(key)
        if value is None:
            raise KeyError(f'Key {idx} not found in LMDB: {self.lmdb_dir}')
        item = pickle.loads(value)
        label = item['label']
        img = Image.open(io.BytesIO(item['raw'])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
