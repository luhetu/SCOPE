# -*- coding: utf-8 -*-
import os, time, torch
import torch.nn as nn
from utils.optim import build_optimizer, build_scheduler
from utils.progress_bar import progress_bar
from datasets.classification import build_imagenet_loader, build_cifar10_loader, build_cifar100_loader
import wandb


class ClassificationTask:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 根据数据集类型确定类别数
        dataset_name = getattr(args, 'dataset', 'imagenet')
        if dataset_name == 'cifar10':
            self.num_classes = 10
        elif dataset_name == 'cifar100':
            self.num_classes = 100
        else:  # imagenet or default
            self.num_classes = 1000

        # ------------------ 模型选择 ------------------ #
        if args.model == 'vit':
            from models.vit import ViT
            self.net = ViT(
                image_size=args.size,
                patch_size=args.patch,
                num_classes=self.num_classes,
                dim=args.dim,
                depth=args.depth,
                heads=args.heads,
                mlp_dim=args.mlp_dim
            )
        elif args.model == 'vitcope':
            from models.vitcope import ViTcope
            # 使用 dim_head 参数（如果有），否则默认为 64
            dim_head = getattr(args, 'dim_head', 64)
            self.net = ViTcope(
                image_size=args.size,
                patch_size=args.patch,
                num_classes=self.num_classes,
                dim=args.dim,
                depth=args.depth,
                heads=args.heads,
                mlp_dim=args.mlp_dim,
                dim_head=dim_head,
                pool='mean',
                dropout=0.1,
                emb_dropout=0.1
            )
        elif args.model == 'vitscope':
            from models.vitscope_embed import ViTScope
            # 使用 dim_head 参数（如果有），否则默认为 64
            dim_head = getattr(args, 'dim_head', 64)
            self.net = ViTScope(
                image_size=args.size,
                patch_size=args.patch,
                num_classes=self.num_classes,
                dim=args.dim,
                depth=args.depth,
                heads=args.heads,
                mlp_dim=args.mlp_dim,
                dim_head=dim_head,
                pool='mean'
            )
        elif args.model == 'vitscope_old':
            from models.vitscope import ViTScope
            self.net = ViTScope(
                image_size=args.size,
                patch_size=args.patch,
                num_classes=self.num_classes,
                dim=args.dim,
                depth=args.depth,
                heads=args.heads,
                mlp_dim=args.mlp_dim
            )
        elif args.model == 'swin':
            from models.swin_transformer import SwinTransformer
            self.net = SwinTransformer(
                img_size=args.size,
                patch_size=args.patch,
                num_classes=self.num_classes,
                embed_dim=args.embed_dim,
                depths=args.depths,
                num_heads=args.num_heads,
                window_size=args.window_size,
                drop_path_rate=args.drop_path_rate,
                patch_norm=True
            )
        else:
            raise ValueError(f"❌ Unknown model: {args.model}")

        self.net = self.net.to(self.device)
        
        # 根据数据集类型加载数据
        dataset_name = getattr(args, 'dataset', 'imagenet')
        if dataset_name == 'cifar10':
            self.trainloader, self.valloader = build_cifar10_loader(
                args.data_dir, args.size, args.bs, 4, args.aug
            )
        elif dataset_name == 'cifar100':
            self.trainloader, self.valloader = build_cifar100_loader(
                args.data_dir, args.size, args.bs, 4, args.aug
            )
        else:  # imagenet or default
            self.trainloader, self.valloader = build_imagenet_loader(
                args.data_dir, args.size, args.bs, 4, args.aug
            )

        # ------------------ 优化器与损失函数 ------------------ #
        # Label smoothing 需要 PyTorch >= 1.10
        pytorch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
        if pytorch_version >= (1, 10):
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = build_optimizer(self.net, args)
        self.scheduler = build_scheduler(self.optimizer, args)
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

        # ------------------ 训练状态 ------------------ #
        self.best_acc = 0.0
        self.start_epoch = 0
        self.start_time = time.time()
        
        # ------------------ 断点续训 ------------------ #
        if hasattr(args, 'resume') and args.resume:
            self.load_checkpoint(args.resume)

        # ------------------ WandB ------------------ #
        self.use_wandb = not args.nowandb
        if self.use_wandb:
            # 根据数据集统一项目名
            dataset_name = getattr(args, 'dataset', 'imagenet')
            project_name = f"{dataset_name}-experiments"
            
            # 运行名称包含模型、数据集和关键参数
            watermark = f"{args.model}_{dataset_name}_size{args.size}_patch{args.patch}_dim{args.dim}_depth{args.depth}_lr{args.lr}"
            
            wandb.init(project=project_name, name=watermark)
            wandb.config.update(vars(args))
            wandb.watch(self.net, log="all", log_freq=100)

    # ------------------------------------------------------- #
    def train_one_epoch(self, epoch):
        self.net.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            # 混合精度训练
            with torch.amp.autocast('cuda', enabled=self.args.amp):
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(self.trainloader),
                f"Loss:{total_loss/(batch_idx+1):.3f} | Acc:{100.*correct/total:.2f}%"
            )

        train_acc = 100. * correct / total
        return total_loss / (batch_idx + 1), train_acc

    # ------------------------------------------------------- #
    def validate(self, epoch):
        self.net.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.valloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                progress_bar(
                    batch_idx,
                    len(self.valloader),
                    f"Loss:{total_loss/(batch_idx+1):.3f} | Acc:{100.*correct/total:.2f}%"
                )
        
        acc = 100. * correct / total
        return total_loss / (batch_idx + 1), acc

    # ------------------------------------------------------- #
    def save_checkpoint(self, acc, epoch, best=False):
        os.makedirs('checkpoint', exist_ok=True)
        filename = f"checkpoint/{self.args.model}_{'best' if best else 'last'}.pth"
        state = {
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'epoch': epoch,
            'acc': acc,
            'best_acc': self.best_acc,
        }
        torch.save(state, filename)
        print(f"💾 Saved: {filename} (acc={acc:.2f}%)")
    
    def load_checkpoint(self, checkpoint_path):
        """从 checkpoint 恢复训练"""
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            return
        
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.net.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        if 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint.get('best_acc', checkpoint['acc'])
        
        print(f"✅ Resumed from epoch {checkpoint['epoch']} (acc={checkpoint['acc']:.2f}%, best={self.best_acc:.2f}%)")

    # ------------------------------------------------------- #
    def train(self):
        dataset_name = getattr(self.args, 'dataset', 'imagenet').upper()
        if self.start_epoch > 0:
            print(f"🔄 Resuming training from epoch {self.start_epoch} to {self.args.n_epochs} on {dataset_name}\n")
        else:
            print(f"🚀 Start training {self.args.model} for {self.args.n_epochs} epochs on {dataset_name}\n")
        
        for epoch in range(self.start_epoch, self.args.n_epochs):
            t0 = time.time()
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            self.scheduler.step()

            # 打印日志
            print(f"[Epoch {epoch:03d}] "
                  f"TrainAcc={train_acc:.2f}% | ValAcc={val_acc:.2f}% | "
                  f"LR={self.optimizer.param_groups[0]['lr']:.6f} | "
                  f"Time={(time.time()-t0)/60:.2f} min")

            # 记录 WandB
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'lr': self.optimizer.param_groups[0]["lr"],
                    'epoch_time_min': (time.time()-t0)/60
                })

            # 保存 best/last 模型
            self.save_checkpoint(val_acc, epoch, best=False)
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(val_acc, epoch, best=True)

        print(f"\n✅ Training finished. Best Acc={self.best_acc:.2f}% "
              f"| Total time={(time.time()-self.start_time)/3600:.2f}h\n")

        if self.use_wandb:
            wandb.run.summary["best_acc"] = self.best_acc
            wandb.finish()
