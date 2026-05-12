# -*- coding: utf-8 -*-
import os
import time
import json
import inspect
import torch
import torch.nn as nn

from utils.optim import build_optimizer, build_scheduler
from utils.progress_bar import progress_bar
from datasets.classification import (
    build_imagenet_loader,
    build_imagenet100_loader,
    build_cifar10_loader,
    build_cifar100_loader,
)

import wandb
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy


class ClassificationTask:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ------------------ Dataset / Classes ------------------ #
        dataset_name = getattr(args, "dataset", "imagenet")
        data_dir = getattr(args, "data_dir", None)
        print(f"✅ [dataset] {dataset_name} | data_dir={data_dir}")

        if dataset_name == "cifar10":
            self.num_classes = 10
        elif dataset_name == "cifar100":
            self.num_classes = 100
        else:
            self.num_classes = 1000

        # Recipe flags debug
        self._debug_log(
            hypothesis_id="H1",
            location="tasks/classification.py:init_recipe_flags",
            message="Recipe flags loaded from config",
            data={
                "model": getattr(args, "model", None),
                "mixup": getattr(args, "mixup", None),
                "cutmix": getattr(args, "cutmix", None),
                "random_erasing": getattr(args, "random_erasing", None),
                "repeated_augmentations": getattr(args, "repeated_augmentations", None),
                "label_smoothing": getattr(args, "label_smoothing", None),
                "stochastic_depth": getattr(args, "stochastic_depth", None),
                "drop_path_rate": getattr(args, "drop_path_rate", None),
                "aug": getattr(args, "aug", None),
            },
        )

        # ------------------ DropPath Setting ------------------ #
        default_drop_path = 0.1 if bool(getattr(args, "stochastic_depth", False)) else 0.0
        drop_path_rate = float(getattr(args, "drop_path_rate", default_drop_path))

        # ------------------ Model Selection ------------------ #
        if args.model == "vit":
            from models.vit import ViT

            model_kwargs = dict(
                image_size=args.size,
                patch_size=args.patch,
                num_classes=self.num_classes,
                dim=args.dim,
                depth=args.depth,
                heads=args.heads,
                mlp_dim=args.mlp_dim,
                dim_head=getattr(args, "dim_head", 64),
                pool=getattr(args, "pool", "cls"),
                dropout=float(getattr(args, "dropout", 0.0)),
                emb_dropout=float(getattr(args, "emb_dropout", 0.0)),
            )
            model_kwargs = self._maybe_add_drop_path(ViT, model_kwargs, drop_path_rate)
            self.net = ViT(**model_kwargs)

        elif args.model == "vitcope":
            from models.vitcope import ViTCoPE

            mlp_dim = args.mlp_dim if hasattr(args, "mlp_dim") and args.mlp_dim else args.dim * 4
            dim_head = getattr(args, "dim_head", args.dim // args.heads)
            use_cls_token = bool(getattr(args, "use_cls_token", False))
            pool = getattr(args, "pool", "cls" if use_cls_token else "mean")

            model_kwargs = dict(
                image_size=args.size,
                patch_size=args.patch,
                num_classes=self.num_classes,
                dim=args.dim,
                depth=args.depth,
                heads=args.heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                use_cls_token=use_cls_token,
                pool=pool,
                dropout=0.1,
                emb_dropout=0.1,
            )
            model_kwargs = self._maybe_add_drop_path(ViTCoPE, model_kwargs, drop_path_rate)
            self.net = ViTCoPE(**model_kwargs)

        elif args.model == "vitscope":
            from models.vitscope import ViTScope

            dim_head = getattr(args, "dim_head", 64)
            model_kwargs = dict(
                image_size=args.size,
                patch_size=args.patch,
                num_classes=self.num_classes,
                dim=args.dim,
                depth=args.depth,
                heads=args.heads,
                mlp_dim=args.mlp_dim,
                dim_head=dim_head,
                dropout=float(getattr(args, "dropout", 0.0)),
                emb_dropout=float(getattr(args, "emb_dropout", 0.0)),
                use_cls_token=bool(getattr(args, "use_cls_token", True)),
                pool=getattr(args, "pool", "cls"),
            )
            model_kwargs = self._maybe_add_drop_path(ViTScope, model_kwargs, drop_path_rate)
            self.net = ViTScope(**model_kwargs)

        elif args.model == "vitscope_nocls":
            from models.vitscope_nocls import ViTScope_NoCLS

            dim_head = getattr(args, "dim_head", 64)
            model_kwargs = dict(
                image_size=args.size,
                patch_size=args.patch,
                num_classes=self.num_classes,
                dim=args.dim,
                depth=args.depth,
                heads=args.heads,
                mlp_dim=args.mlp_dim,
                dim_head=dim_head,
                dropout=float(getattr(args, "dropout", 0.0)),
                emb_dropout=float(getattr(args, "emb_dropout", 0.0)),
                use_cls_token=bool(getattr(args, "use_cls_token", True)),
                pool=getattr(args, "pool", "cls"),
            )
            model_kwargs = self._maybe_add_drop_path(ViTScope_NoCLS, model_kwargs, drop_path_rate)
            self.net = ViTScope_NoCLS(**model_kwargs)

        elif args.model == "vitpool":
            from models.vitpool import ViTPool

            dim_head = getattr(args, "dim_head", 32)
            model_kwargs = dict(
                image_size=args.size,
                patch_size=args.patch,
                num_classes=self.num_classes,
                dim=args.dim,
                depth=args.depth,
                heads=args.heads,
                mlp_dim=args.mlp_dim,
                dim_head=dim_head,
            )
            model_kwargs = self._maybe_add_drop_path(ViTPool, model_kwargs, drop_path_rate)
            self.net = ViTPool(**model_kwargs)

        elif args.model == "vitpool_nocls":
            from models.vitpool_nocls import ViTPool_NoCLS

            dim_head = getattr(args, "dim_head", 32)
            model_kwargs = dict(
                image_size=args.size,
                patch_size=args.patch,
                num_classes=self.num_classes,
                dim=args.dim,
                depth=args.depth,
                heads=args.heads,
                mlp_dim=args.mlp_dim,
                dim_head=dim_head,
            )
            model_kwargs = self._maybe_add_drop_path(ViTPool_NoCLS, model_kwargs, drop_path_rate)
            self.net = ViTPool_NoCLS(**model_kwargs)

        elif args.model == "vitcpe":
            from models.vitcpe import ViT_CPE

            dim_head = getattr(args, "dim_head", 64)
            model_kwargs = dict(
                image_size=args.size,
                patch_size=args.patch,
                num_classes=self.num_classes,
                dim=args.dim,
                depth=args.depth,
                heads=args.heads,
                mlp_dim=args.mlp_dim,
                dim_head=dim_head,
                dropout=0.1,
                emb_dropout=0.1,
            )
            model_kwargs = self._maybe_add_drop_path(ViT_CPE, model_kwargs, drop_path_rate)
            self.net = ViT_CPE(**model_kwargs)

        elif args.model == "vitscope_old":
            from models.vitscope import ViTScope

            model_kwargs = dict(
                image_size=args.size,
                patch_size=args.patch,
                num_classes=self.num_classes,
                dim=args.dim,
                depth=args.depth,
                heads=args.heads,
                mlp_dim=args.mlp_dim,
            )
            model_kwargs = self._maybe_add_drop_path(ViTScope, model_kwargs, drop_path_rate)
            self.net = ViTScope(**model_kwargs)

        elif args.model == "swin":
            from models.swin_transformer import SwinTransformer

            self.net = SwinTransformer(
                img_size=args.size,
                patch_size=args.patch,
                num_classes=self.num_classes,
                embed_dim=args.embed_dim,
                depths=args.depths,
                num_heads=args.num_heads,
                window_size=args.window_size,
                drop_path_rate=drop_path_rate,
                patch_norm=True,
            )

        else:
            raise ValueError(f"❌ Unknown model: {args.model}")

        self.net = self.net.to(self.device)

        # DropPath debug
        drop_path_modules = 0
        for module in self.net.modules():
            if module.__class__.__name__.lower() == "droppath":
                drop_path_modules += 1

        self._debug_log(
            hypothesis_id="H2",
            location="tasks/classification.py:model_drop_path_probe",
            message="Model drop-path module count",
            data={
                "model": getattr(args, "model", None),
                "drop_path_rate": drop_path_rate,
                "drop_path_modules": int(drop_path_modules),
            },
        )

        print(f"✅ DropPath rate: {drop_path_rate}")
        print(f"✅ DropPath modules: {drop_path_modules}")

        # ------------------ Dataloader ------------------ #
        num_workers = 6 if "SLURM_JOB_ID" in os.environ else 4
        if hasattr(args, "workers_per_gpu") and args.workers_per_gpu is not None:
            num_workers = int(args.workers_per_gpu)

        dataset_name = getattr(args, "dataset", "imagenet")

        if dataset_name == "cifar10":
            self.trainloader, self.valloader = build_cifar10_loader(
                args.data_dir,
                args.size,
                args.bs,
                num_workers,
                args.aug,
            )

        elif dataset_name == "cifar100":
            self.trainloader, self.valloader = build_cifar100_loader(
                args.data_dir,
                args.size,
                args.bs,
                num_workers,
                args.aug,
            )

        else:
            if dataset_name in ["imagenet100", "im100"]:
                self.trainloader, self.valloader = build_imagenet100_loader(
                    args.data_dir,
                    args.size,
                    args.bs,
                    num_workers,
                    args.aug,
                    random_erasing=bool(getattr(args, "random_erasing", False)),
                    repeated_augmentations=bool(getattr(args, "repeated_augmentations", False)),
                )
            else:
                self.trainloader, self.valloader =build_imagenet_loader(
    args.data_dir,
    args.size,
    args.bs,
    num_workers,
    args.aug,
    random_erasing=bool(getattr(args, "random_erasing", False)),
    repeated_augmentations=bool(getattr(args, "repeated_augmentations", False)),
)
        train_transform_names = []
        train_dataset_name = None
        sampler_name = None

        dataset_obj = getattr(self.trainloader, "dataset", None)
        if dataset_obj is not None:
            train_dataset_name = dataset_obj.__class__.__name__

            if hasattr(dataset_obj, "transform") and hasattr(dataset_obj.transform, "transforms"):
                train_transform_names = [t.__class__.__name__ for t in dataset_obj.transform.transforms]

            elif hasattr(dataset_obj, "datasets") and len(dataset_obj.datasets) > 0:
                first_ds = dataset_obj.datasets[0]
                if hasattr(first_ds, "transform") and hasattr(first_ds.transform, "transforms"):
                    train_transform_names = [t.__class__.__name__ for t in first_ds.transform.transforms]

        if getattr(self.trainloader, "sampler", None) is not None:
            sampler_name = self.trainloader.sampler.__class__.__name__

        self._debug_log(
            hypothesis_id="H3",
            location="tasks/classification.py:dataloader_probe",
            message="Dataloader transform/sampler summary",
            data={
                "dataset": train_dataset_name,
                "sampler": sampler_name,
                "transforms": train_transform_names,
                "aug": getattr(args, "aug", None),
                "random_erasing": getattr(args, "random_erasing", None),
                "repeated_augmentations": getattr(args, "repeated_augmentations", None),
            },
        )

        print(f"✅ Train sampler: {sampler_name}")
        print(f"✅ Train transforms: {train_transform_names}")

        # ------------------ Loss / Mixup / CutMix ------------------ #
        self.mixup_fn = None
        mixup_active = bool(getattr(args, "mixup", False)) or bool(getattr(args, "cutmix", False))

        if mixup_active:
            self.mixup_fn = Mixup(
                mixup_alpha=0.8 if bool(getattr(args, "mixup", False)) else 0.0,
                cutmix_alpha=1.0 if bool(getattr(args, "cutmix", False)) else 0.0,
                cutmix_minmax=None,
                prob=1.0,
                switch_prob=0.5,
                mode="batch",
                label_smoothing=0.1 if bool(getattr(args, "label_smoothing", True)) else 0.0,
                num_classes=self.num_classes,
            )
            self.train_criterion = SoftTargetCrossEntropy()
        else:
            smoothing = 0.1 if bool(getattr(args, "label_smoothing", True)) else 0.0
            self.train_criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)

        self.val_criterion = nn.CrossEntropyLoss()

        self._debug_log(
            hypothesis_id="H4",
            location="tasks/classification.py:criterion_probe",
            message="Criterion and label-smoothing summary",
            data={
                "mixup_active": bool(self.mixup_fn is not None),
                "train_criterion_class": self.train_criterion.__class__.__name__,
                "val_criterion_class": self.val_criterion.__class__.__name__,
                "cfg_label_smoothing": getattr(args, "label_smoothing", None),
            },
        )

        print(f"✅ Mixup/CutMix active: {self.mixup_fn is not None}")
        print(f"✅ Train criterion: {self.train_criterion.__class__.__name__}")
        print(f"✅ Val criterion: {self.val_criterion.__class__.__name__}")

        # ------------------ Optimizer / Scheduler ------------------ #
        self.optimizer = build_optimizer(self.net, args)
        self.scheduler = build_scheduler(self.optimizer, args)
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

        # ------------------ Training State ------------------ #
        self.best_acc = 0.0
        self.start_epoch = 0
        self.start_time = time.time()
        self.run_tag = getattr(args, "run_tag", time.strftime("%m%d_%H%M%S"))

        # ------------------ Resume ------------------ #
        if hasattr(args, "resume") and args.resume:
            self.load_checkpoint(args.resume)

        # ------------------ Profiling ------------------ #
        self.time_profile = bool(getattr(args, "time_profile", False))
        self.time_profile_interval = int(getattr(args, "time_profile_interval", 1000))
        self.time_profile = self.time_profile and self.time_profile_interval > 0

        self.log_interval = int(getattr(args, "log_interval", 50))
        if self.log_interval <= 0:
            self.log_interval = 50

        # ------------------ WandB ------------------ #
        self.use_wandb = not args.nowandb

        if self.use_wandb:
            project_name = f"{dataset_name}-experiments"
            watermark = (
                f"{args.model}_{dataset_name}_size{args.size}_patch{args.patch}_"
                f"dim{args.dim}_depth{args.depth}_lr{args.lr}_{self.run_tag}"
            )

            wandb.init(project=project_name, name=watermark)
            wandb.config.update(vars(args))
            wandb.watch(self.net, log="gradients", log_freq=1000)

    # ------------------------------------------------------- #
    def train_one_epoch(self, epoch):
        self.net.train()

        if hasattr(self.trainloader.sampler, "set_epoch"):
            self.trainloader.sampler.set_epoch(epoch)

        total_loss, correct, total = 0.0, 0, 0
        prev_end = time.time() if self.time_profile else None

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            batch_start = time.time() if self.time_profile else None
            wait_ms = (batch_start - prev_end) * 1000.0 if self.time_profile else None

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            hard_targets = targets

            if epoch == self.start_epoch and batch_idx == 0:
                self._debug_log(
                    hypothesis_id="H5",
                    location="tasks/classification.py:first_batch_before_mixup",
                    message="First batch target tensor before mixup",
                    data={
                        "targets_dtype": str(targets.dtype),
                        "targets_ndim": int(targets.ndim),
                        "targets_shape": list(targets.shape),
                    },
                )

            if self.mixup_fn is not None:
                inputs, targets = self.mixup_fn(inputs, targets)

            if epoch == self.start_epoch and batch_idx == 0:
                self._debug_log(
                    hypothesis_id="H6",
                    location="tasks/classification.py:first_batch_after_mixup",
                    message="First batch target tensor after mixup",
                    data={
                        "targets_dtype": str(targets.dtype),
                        "targets_ndim": int(targets.ndim),
                        "targets_shape": list(targets.shape),
                    },
                )

            after_data = time.time() if self.time_profile else None

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.args.amp):
                outputs = self.net(inputs)
                loss = self.train_criterion(outputs, targets)

            after_forward = time.time() if self.time_profile else None

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            after_step = time.time() if self.time_profile else None
            if self.time_profile:
                prev_end = after_step

            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += hard_targets.size(0)
            correct += predicted.eq(hard_targets).sum().item()

            if self.time_profile and (batch_idx % self.time_profile_interval == 0):
                data_ms = (after_data - batch_start) * 1000.0
                fwd_ms = (after_forward - after_data) * 1000.0
                step_ms = (after_step - after_forward) * 1000.0
                total_ms = (after_step - batch_start) * 1000.0
                gpu_util = 100.0 * total_ms / (total_ms + wait_ms) if wait_ms > 0 else 100.0

                msg = (
                    f"[PROFILE] batch={batch_idx} "
                    f"wait={wait_ms:.1f}ms data={data_ms:.1f}ms "
                    f"fwd_launch={fwd_ms:.1f}ms bwd+opt(GPU)={step_ms:.1f}ms "
                    f"compute={total_ms:.1f}ms gpu_util={gpu_util:.1f}% "
                    f"| Loss:{total_loss / (batch_idx + 1):.3f} "
                    f"| Acc:{100. * correct / total:.2f}%"
                )
                print(msg, flush=True)

        train_acc = 100.0 * correct / total
        return total_loss / (batch_idx + 1), train_acc

    # ------------------------------------------------------- #
    def validate(self, epoch):
        self.net.eval()

        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.valloader):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.net(inputs)
                loss = self.val_criterion(outputs, targets)

                total_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100.0 * correct / total
        return total_loss / (batch_idx + 1), acc

    # ------------------------------------------------------- #
    def save_checkpoint(self, acc, epoch, best=False):
        os.makedirs("checkpoint", exist_ok=True)

        task = self.args.task or "cls"
        size = getattr(self.args, "size", "na")
        patch = getattr(self.args, "patch", "na")
        dim = getattr(self.args, "dim", "na")
        depth = getattr(self.args, "depth", "na")
        tag = "best" if best else "last"

        filename = (
            f"checkpoint/{self.args.model}_{task}_"
            f"size{size}_patch{patch}_dim{dim}_depth{depth}_"
            f"{self.run_tag}_{tag}.pth"
        )

        state = {
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "acc": acc,
            "best_acc": self.best_acc,
        }

        torch.save(state, filename)
        print(f"💾 Saved: {filename} (acc={acc:.2f}%)")

    # ------------------------------------------------------- #
    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            return

        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.net.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_acc = checkpoint.get("best_acc", checkpoint["acc"])

        print(
            f"✅ Resumed from epoch {checkpoint['epoch']} "
            f"(acc={checkpoint['acc']:.2f}%, best={self.best_acc:.2f}%)"
        )

    # ------------------------------------------------------- #
    def train(self):
        dataset_name = getattr(self.args, "dataset", "imagenet").upper()

        if self.start_epoch > 0:
            print(
                f"🔄 Resuming training from epoch {self.start_epoch} "
                f"to {self.args.n_epochs} on {dataset_name}\n"
            )
        else:
            print(
                f"🚀 Start training {self.args.model} "
                f"for {self.args.n_epochs} epochs on {dataset_name}\n"
            )

        for epoch in range(self.start_epoch, self.args.n_epochs):
            t0 = time.time()

            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            self.scheduler.step()

            print(
                f"[Epoch {epoch:03d}] "
                f"TrainAcc={train_acc:.2f}% | ValAcc={val_acc:.2f}% | "
                f"LR={self.optimizer.param_groups[0]['lr']:.6f} | "
                f"Time={(time.time() - t0) / 60:.2f} min"
            )

            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "epoch_time_min": (time.time() - t0) / 60,
                    }
                )

            self.save_checkpoint(val_acc, epoch, best=False)

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(val_acc, epoch, best=True)

        print(
            f"\n✅ Training finished. Best Acc={self.best_acc:.2f}% "
            f"| Total time={(time.time() - self.start_time) / 3600:.2f}h\n"
        )

        if self.use_wandb:
            wandb.run.summary["best_acc"] = self.best_acc
            wandb.finish()

    # ------------------------------------------------------- #
    def _maybe_add_drop_path(self, model_cls, kwargs, drop_path_rate):
        """
        如果模型 __init__ 支持 drop_path_rate，就自动传入。
        如果模型还没改 DropPath，这里不会报错，但 DropPath modules 会是 0。
        """
        try:
            sig = inspect.signature(model_cls.__init__)
            if "drop_path_rate" in sig.parameters:
                kwargs["drop_path_rate"] = drop_path_rate
        except Exception:
            pass
        return kwargs

    # ------------------------------------------------------- #
    def _debug_log(self, hypothesis_id, location, message, data):
        payload = {
            "sessionId": "a67f17",
            "runId": os.environ.get("DEBUG_RUN_ID", "pre-fix"),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }

        try:
            os.makedirs("/home3/dnrx52/SCOPE/.cursor", exist_ok=True)
            with open(
                "/home3/dnrx52/SCOPE/.cursor/debug-a67f17.log",
                "a",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass