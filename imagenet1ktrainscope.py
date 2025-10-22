# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv
import time
import math
import random

from models import *
from utils import progress_bar
from randomaug import RandAugment
from models.vitscope442 import ViTcope  # 保留你的自定义模型路径
from models.convmixer import ConvMixer
from models.mobilevit import mobilevit_xxs

# ---------------- Arguments ---------------- #
parser = argparse.ArgumentParser(description='PyTorch ImageNet-1k Training (scope)')
parser.add_argument('--lr', default=3e-4, type=float)                # ↑ 默认 3e-4
parser.add_argument('--min_lr', default=1e-5, type=float, help='cosine min lr')
parser.add_argument('--warmup_epochs', type=int, default=2, help='linear warmup epochs')  # ↑ 2
parser.add_argument('--opt', default="adamw", choices=["adam", "adamw", "sgd"])          # ↑ 默认 adamw
parser.add_argument('--resume', '-r', action='store_true')
parser.add_argument('--aug', action='store_true', help='use RandAugment')
parser.add_argument('--amp', action='store_true', help='enable mixed precision')
parser.add_argument('--nowandb', action='store_true')
parser.add_argument('--mixup', action='store_true')   # 传入则启用 Mixup(α=0.2)
parser.add_argument('--net', default='vit', choices=['vit', 'res18'])
parser.add_argument('--dp', action='store_true')
parser.add_argument('--bs', default='256')
parser.add_argument('--size', default='256')
parser.add_argument('--n_epochs', type=int, default=100)            # 
parser.add_argument('--patch', default=32, type=int)                # 32
parser.add_argument('--dimhead', default=384, type=int)             # ↑ 384 (ViT-S)
parser.add_argument('--heads', default=6, type=int)                 # ↑ 6
parser.add_argument('--mlp_dim', default=1536, type=int)            # ↑ 1536
args = parser.parse_args()

# ---------------- Env ---------------- #
usewandb = not args.nowandb
if usewandb:
    import wandb
    watermark = "{}scopeS32_lr{}".format(args.net, args.lr)
    wandb.init(project="imagenet-1k-challenge", name=watermark)
    wandb.config.update(vars(args))

bs = int(args.bs)
imsize = int(args.size)
use_amp = args.amp

# RandAugment 默认关闭；主增强改为 RandomResizedCrop
aug = args.aug  # 若传 --noaug 会置 False；保持和你原逻辑一致

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.0
start_epoch = 0

# ---------------- Data ---------------- #
print('==> Preparing data..')
size = imsize  # 一律 256x256

# 关键：使用 RandomResizedCrop(256)
from torchvision.transforms import InterpolationMode
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.08, 1.0), interpolation=InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
transform_test = transforms.Compose([
    transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 可选再叠加 RandAugment（默认不开）
if aug:
    N, M = 2, 14
    # 插在 RRC 之后、ToTensor 之前
    transform_train.transforms.insert(1, RandAugment(N, M))

data_dir = "/home/hetu/ImageNet"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)

valset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform_test)
valloader = torch.utils.data.DataLoader(valset, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

# ---------------- Model ---------------- #
print('==> Building model..')
if args.net == 'res18':
    net = ResNet18()
elif args.net == "vit":
    net = ViTcope(
        image_size=size,
        patch_size=args.patch,        # 32
        num_classes=1000,
        dim=int(args.dimhead),        # 384
        depth=12,
        heads=int(args.heads),        # 6
        mlp_dim=int(args.mlp_dim),    # 1536
        dropout=0.1,
        emb_dropout=0.1
    )
else:
    raise ValueError(f"'{args.net}' is not a valid model")

# Multi-GPU
if device == 'cuda':
    if args.dp:
        print("using data parallel")
        net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# ---------------- Resume ---------------- #
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net), map_location='cpu')
    state_dict = checkpoint.get('net', checkpoint.get('model'))
    net.load_state_dict(state_dict)
    best_acc = checkpoint.get('acc', 0.0)
    start_epoch = checkpoint.get('epoch', 0)

# ---------------- Mixup helper ---------------- #
def mixup_data(x, y, alpha=0.2):
    if alpha <= 0.0: return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b), lam

def mixup_criterion(criterion, pred, targets, lam):
    y_a, y_b = targets
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ---------------- Optim / Sched ---------------- #
# Label Smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

if args.opt == "adamw":
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.999))
elif args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
else:
    raise ValueError(f"Unknown optimizer: {args.opt}")

# Warmup + Cosine 到 min_lr
base_lr = args.lr
min_lr = args.min_lr
warmup_epochs = max(1, int(args.warmup_epochs))
total_epochs = int(args.n_epochs)

def lr_lambda(current_epoch: int):
    e = current_epoch
    if e < warmup_epochs:
        return float(e + 1) / float(warmup_epochs)  # 0->1 线性
    progress = (e - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (min_lr / base_lr) + (1.0 - (min_lr / base_lr)) * cosine

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# ---------------- Train / Eval ---------------- #
import numpy as np

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        use_mix = bool(args.mixup)
        if use_mix:
            inputs, mix_targets, lam = mixup_data(inputs, targets, alpha=0.2)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            if use_mix:
                loss = mixup_criterion(criterion, outputs, mix_targets, lam)
            else:
                loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if not use_mix:
            correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx, len(trainloader),
            'Loss: %.3f | Acc: %.3f%% (%d/%d) | LR: %.6f' % (
                train_loss/(batch_idx+1),
                100.*correct/total if not use_mix else 0.0,
                correct,
                total,
                optimizer.param_groups[0]['lr']
            )
        )
    return train_loss/(batch_idx+1)

def test(epoch):
    global best_acc
    net.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx, len(valloader),
                'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    test_loss/(batch_idx+1),
                    100.*correct/total,
                    correct,
                    total
                )
            )
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "acc": acc,
            "epoch": epoch
        }
        os.makedirs('checkpoint', exist_ok=True)
        torch.save(state, './checkpoint/'+args.net+'-{}-imagenet-1k-scope-ckpt.t7'.format(args.patch))
        best_acc = acc

    os.makedirs("log", exist_ok=True)
    content = time.ctime() + " " + f"Epoch {epoch}, lr: {optimizer.param_groups[0]['lr']:.7f}, val loss: {test_loss:.5f}, acc: {acc:.5f}"
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss, list_acc = [], []

if usewandb:
    wandb.watch(net)

net.to(device)
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)

    # 每个 epoch 结束后再 step
    scheduler.step()

    list_loss.append(val_loss)
    list_acc.append(acc)

    if usewandb:
        wandb.log({
            'epoch': epoch,
            'train_loss': trainloss,
            'val_loss': val_loss,
            'val_acc': acc,
            'lr': optimizer.param_groups[0]["lr"],
            "epoch_time": time.time()-start
        })

    os.makedirs('log', exist_ok=True)
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss)
        writer.writerow(list_acc)

if usewandb:
    wandb.save("wandb_{}.h5".format(args.net))
