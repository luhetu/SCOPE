# -*- coding: utf-8 -*-
"""
进度条 Hook，用于显示训练进度（类似分类任务）
"""
from mmcv.runner import HOOKS, Hook
from mmcv.utils import ProgressBar
import sys


@HOOKS.register_module()
class ProgressBarHook(Hook):
    """在训练过程中显示进度条"""
    
    def before_train_epoch(self, runner):
        """在每个 epoch 开始时初始化进度条"""
        self.prog_bar = ProgressBar(len(runner.data_loader))
        # 打印 epoch 开始信息
        print(f"\nEpoch {runner.epoch + 1}/{runner.max_epochs}")
    
    def after_train_iter(self, runner):
        """每次迭代后更新进度条"""
        self.prog_bar.update()
    
    def after_train_epoch(self, runner):
        """每个 epoch 结束时打印损失信息"""
        sys.stdout.write('\n')
        
        # 收集所有损失
        losses = {}
        for key, val in runner.log_buffer.output.get('log_vars', {}).items():
            if isinstance(val, (int, float)):
                losses[key] = val
        
        # 打印损失信息
        if losses:
            loss_str = ' | '.join([f'{k}: {v:.4f}' for k, v in losses.items() if 'loss' in k])
            if loss_str:
                print(f'  Losses: {loss_str}')
        
        # 打印学习率
        lr = runner.current_lr()
        if isinstance(lr, list):
            lr = lr[0]
        print(f'  LR: {lr:.6f}')
        sys.stdout.flush()

