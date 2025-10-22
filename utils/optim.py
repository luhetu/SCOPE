import math, torch

def build_optimizer(model, args):
    if args.opt == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.999))
    elif args.opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {args.opt}")

def build_scheduler(opt, args):
    base_lr, min_lr, warmup_epochs, total_epochs = args.lr, args.min_lr, args.warmup_epochs, args.n_epochs

    def lr_lambda(e):
        if e < warmup_epochs:
            return float(e + 1) / float(warmup_epochs)
        progress = (e - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr / base_lr) + (1.0 - (min_lr / base_lr)) * cosine
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
