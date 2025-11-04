# -*- coding: utf-8 -*-
"""
WandB Logger Hook for Detection
记录 COCO 评估指标：APb, APb50, APb75, APm, APm50, APm75
"""
from mmcv.runner import HOOKS, Hook

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@HOOKS.register_module()
class WandBLoggerHook(Hook):
    """WandB 日志记录 Hook，记录检测训练和评估指标"""
    
    def __init__(self, interval=100, use_wandb=True):
        self.interval = interval
        self.use_wandb = use_wandb and WANDB_AVAILABLE
    
    def after_train_iter(self, runner):
        """训练迭代后记录损失"""
        if not self.use_wandb:
            return
        
        # 每 interval 次记录一次训练指标
        if self.every_n_iters(runner, self.interval) and runner.rank == 0:
            log_dict = {}
            for key, val in runner.log_buffer.output.items():
                # 记录所有数值型指标（loss, acc, lr等）
                if isinstance(val, (int, float)):
                    log_dict[f'train/{key}'] = val
            
            if log_dict:
                wandb.log(log_dict, step=runner.iter)
    
    def after_val_epoch(self, runner):
        """验证后记录 COCO 评估指标"""
        if not self.use_wandb or runner.rank != 0:
            return
        
        log_dict = {}
        # 直接从 log_buffer.output 读取所有指标
        for key, val in runner.log_buffer.output.items():
            # 记录所有包含 mAP, AP 的指标
            if isinstance(val, (int, float)) and any(metric in key for metric in ['mAP', 'AP', 'bbox', 'segm']):
                log_dict[f'val/{key}'] = val
        
        if log_dict:
            wandb.log(log_dict, step=runner.iter)

