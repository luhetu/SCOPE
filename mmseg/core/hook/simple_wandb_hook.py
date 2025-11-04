# -*- coding: utf-8 -*-
"""
简单的 WandB Hook，用于记录评估指标
"""
from mmcv.runner import Hook, HOOKS

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@HOOKS.register_module()
class SimpleWandBHook(Hook):
    """简单的 WandB Hook，记录训练和评估指标"""
    
    def __init__(self, use_wandb=True, log_interval=50):
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.log_interval = log_interval
    
    def after_train_iter(self, runner):
        """训练迭代后记录指标"""
        if not self.use_wandb:
            return
        
        # 每 log_interval 次记录一次
        if self.every_n_iters(runner, self.log_interval):
            log_dict = {}
            for key, val in runner.log_buffer.output.items():
                # 记录所有训练指标（loss, acc, lr等）
                if isinstance(val, (int, float)):
                    log_dict[f'train/{key}'] = val
            
            if log_dict and runner.rank == 0:  # 只在主进程记录
                wandb.log(log_dict, step=runner.iter)
    
    def after_val_epoch(self, runner):
        """验证后记录评估指标"""
        if not self.use_wandb:
            return
        
        if runner.rank == 0:  # 只在主进程记录
            log_dict = {}
            for key, val in runner.log_buffer.output.items():
                # 记录所有评估指标
                if any(metric in key for metric in ['mIoU', 'mAcc', 'aAcc', 'IoU', 'Acc']):
                    log_dict[f'val/{key}'] = val
            
            if log_dict:
                wandb.log(log_dict, step=runner.iter)

