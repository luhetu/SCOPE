# -*- coding: utf-8 -*-
"""
进度条 Hook for Segmentation，用于显示训练进度
"""
from mmcv.runner import HOOKS, Hook
from mmcv.utils import ProgressBar
import sys


@HOOKS.register_module()
class SegProgressBarHook(Hook):
    """在训练过程中显示进度条（分割任务专用）"""
    
    def __init__(self, num_iters_per_epoch=1000):
        self.num_iters_per_epoch = num_iters_per_epoch
        self.current_epoch = 0
    
    def before_train_iter(self, runner):
        """每次迭代前检查是否需要初始化新的进度条"""
        current_epoch = runner.iter // self.num_iters_per_epoch
        
        if current_epoch != self.current_epoch:
            self.current_epoch = current_epoch
            max_epochs = runner.max_iters // self.num_iters_per_epoch
            # 打印 epoch 开始信息
            print(f"\nEpoch {self.current_epoch + 1}/{max_epochs}")
            self.prog_bar = ProgressBar(self.num_iters_per_epoch)
    
    def after_train_iter(self, runner):
        """每次迭代后更新进度条"""
        iter_in_epoch = runner.iter % self.num_iters_per_epoch
        if iter_in_epoch == 0 and runner.iter > 0:
            # Epoch 结束，打印损失信息
            sys.stdout.write('\n')
            
            # 收集所有损失
            losses = {}
            for key, val in runner.log_buffer.output.get('log_vars', {}).items():
                if isinstance(val, (int, float)):
                    losses[key] = val
            
            # 打印损失信息
            if losses:
                loss_str = ' | '.join([f'{k}: {v:.4f}' for k, v in losses.items() if 'loss' in k or 'acc' in k])
                if loss_str:
                    print(f'  Metrics: {loss_str}')
            
            # 打印学习率
            lr = runner.current_lr()
            if isinstance(lr, list):
                lr = lr[0]
            print(f'  LR: {lr:.6f}')
            sys.stdout.flush()
        else:
            self.prog_bar.update()

