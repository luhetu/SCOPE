#!/usr/bin/env python3
"""检查数据加载速度"""
import time
import os
from datasets.classification import build_imagenet_loader

data_dir = "/home2/dnrx52/vitcope"

print("=" * 60)
print("测试不同配置的数据加载速度")
print("=" * 60)

configs = [
    ("bs=256, workers=4, aug=False", 256, 4, False),
    ("bs=256, workers=4, aug=True", 256, 4, True),
    ("bs=512, workers=4, aug=False", 512, 4, False),
    ("bs=768, workers=4, aug=False", 768, 4, False),
]

for name, bs, workers, aug in configs:
    print(f"\n测试配置: {name}")
    try:
        trainloader, _ = build_imagenet_loader(data_dir, 224, bs, workers, aug)
        
        # 测试50个batch
        start = time.time()
        for i, (images, labels) in enumerate(trainloader):
            if i >= 50:
                break
            if i % 10 == 0:
                elapsed = time.time() - start
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  Batch {i+1}/50: {speed:.2f} batches/sec")
        
        total_time = time.time() - start
        avg_speed = 50 / total_time
        print(f"  ✅ 平均速度: {avg_speed:.2f} batches/sec ({1000/avg_speed:.1f}ms/batch)")
        
    except Exception as e:
        print(f"  ❌ 错误: {e}")

print("\n" + "=" * 60)



