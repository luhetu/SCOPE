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
    """WandB 日志记录 Hook，专门记录检测评估指标"""
    
    def __init__(self, interval=1, use_wandb=True):
        self.interval = interval
        self.use_wandb = use_wandb and WANDB_AVAILABLE
    
    def after_train_iter(self, runner):
        """训练迭代后记录损失"""
        if not self.use_wandb:
            return
        
        # 每 interval 次记录一次训练损失
        if self.every_n_iters(runner, self.interval * 50):
            log_dict = {}
            for key, val in runner.log_buffer.output.get('log_vars', {}).items():
                if isinstance(val, (int, float)):
                    log_dict[f'train/{key}'] = val
            
            if log_dict:
                wandb.log(log_dict, step=runner.iter)
    
    def after_val_epoch(self, runner):
        """验证后记录 COCO 评估指标"""
        if not self.use_wandb:
            return
        
        # 获取评估结果
        eval_results = runner.log_buffer.output.get('eval_results', {})
        
        if eval_results:
            log_dict = {}
            
            # COCO bbox metrics
            if 'bbox_mAP' in eval_results:
                log_dict['val/APb'] = eval_results['bbox_mAP']
            if 'bbox_mAP_50' in eval_results:
                log_dict['val/APb50'] = eval_results['bbox_mAP_50']
            if 'bbox_mAP_75' in eval_results:
                log_dict['val/APb75'] = eval_results['bbox_mAP_75']
            
            # COCO segm metrics
            if 'segm_mAP' in eval_results:
                log_dict['val/APm'] = eval_results['segm_mAP']
            if 'segm_mAP_50' in eval_results:
                log_dict['val/APm50'] = eval_results['segm_mAP_50']
            if 'segm_mAP_75' in eval_results:
                log_dict['val/APm75'] = eval_results['segm_mAP_75']
            
            # 额外的指标
            if 'bbox_mAP_s' in eval_results:
                log_dict['val/APb_small'] = eval_results['bbox_mAP_s']
            if 'bbox_mAP_m' in eval_results:
                log_dict['val/APb_medium'] = eval_results['bbox_mAP_m']
            if 'bbox_mAP_l' in eval_results:
                log_dict['val/APb_large'] = eval_results['bbox_mAP_l']
            
            if 'segm_mAP_s' in eval_results:
                log_dict['val/APm_small'] = eval_results['segm_mAP_s']
            if 'segm_mAP_m' in eval_results:
                log_dict['val/APm_medium'] = eval_results['segm_mAP_m']
            if 'segm_mAP_l' in eval_results:
                log_dict['val/APm_large'] = eval_results['segm_mAP_l']
            
            if log_dict:
                wandb.log(log_dict, step=runner.iter)
                
                # 打印关键指标
                print(f"\n📊 Validation Metrics:")
                if 'val/APb' in log_dict:
                    print(f"   APb: {log_dict['val/APb']:.4f}, APb50: {log_dict.get('val/APb50', 0):.4f}, APb75: {log_dict.get('val/APb75', 0):.4f}")
                if 'val/APm' in log_dict:
                    print(f"   APm: {log_dict['val/APm']:.4f}, APm50: {log_dict.get('val/APm50', 0):.4f}, APm75: {log_dict.get('val/APm75', 0):.4f}")

