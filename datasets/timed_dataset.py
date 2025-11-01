"""带计时功能的ImageFolder数据集"""
import time
import os
from PIL import Image
import torch
from torchvision.datasets import ImageFolder

class TimedImageFolder(ImageFolder):
    """带详细计时的ImageFolder，用于诊断数据加载瓶颈"""
    
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)
        self.timers = {
            'io_read': [],      # 文件I/O读取
            'decode': [],       # JPEG解码
            'transform': [],    # Transform操作
            'total': []         # 总时间
        }
        self.sample_count = 0
        self.report_interval = 100  # 每100个样本报告一次
        
    def __getitem__(self, index):
        """重写__getitem__添加计时"""
        t_start = time.time()
        
        # 1. 读取图片路径
        path, target = self.samples[index]
        
        # 2. I/O读取 + JPEG解码（合并测量）
        t_io = time.time()
        img = Image.open(path).convert('RGB')
        decode_time = time.time() - t_io
        io_time = decode_time  # I/O和解码在PIL中是一起的
        
        # 4. Transform操作
        t_transform = time.time()
        if self.transform is not None:
            img = self.transform(img)
        transform_time = time.time() - t_transform
        
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        total_time = time.time() - t_start
        
        # 记录时间
        self.timers['io_read'].append(io_time * 1000)  # 转换为ms
        self.timers['decode'].append(decode_time * 1000)
        self.timers['transform'].append(transform_time * 1000)
        self.timers['total'].append(total_time * 1000)
        
        self.sample_count += 1
        
        # 每100个样本报告一次
        if self.sample_count % self.report_interval == 0:
            self.print_stats()
            
        return img, target
    
    def print_stats(self):
        """打印统计信息"""
        n = len(self.timers['total'])
        if n == 0:
            return
            
        print(f"\n{'='*70}")
        print(f"📊 DataLoader内部性能分析 (最近{self.report_interval}个样本)")
        print(f"{'='*70}")
        print(f"{'环节':<15} {'平均时间':>12} {'占比':>8} {'最大时间':>12}")
        print(f"{'-'*70}")
        
        # 计算最近100个样本的平均值
        recent = min(self.report_interval, n)
        
        io_avg = sum(self.timers['io_read'][-recent:]) / recent
        decode_avg = sum(self.timers['decode'][-recent:]) / recent
        transform_avg = sum(self.timers['transform'][-recent:]) / recent
        total_avg = sum(self.timers['total'][-recent:]) / recent
        
        io_max = max(self.timers['io_read'][-recent:])
        decode_max = max(self.timers['decode'][-recent:])
        transform_max = max(self.timers['transform'][-recent:])
        total_max = max(self.timers['total'][-recent:])
        
        print(f"{'I/O+解码':<15} {decode_avg:>10.1f}ms {decode_avg/total_avg*100:>7.1f}% {decode_max:>10.1f}ms")
        print(f"{'Transform':<15} {transform_avg:>10.1f}ms {transform_avg/total_avg*100:>7.1f}% {transform_max:>10.1f}ms")
        print(f"{'-'*70}")
        print(f"{'总计':<15} {total_avg:>10.1f}ms/样本  最大: {total_max:.1f}ms")
        
        # 瓶颈分析
        max_component = max([
            ('I/O+解码', decode_avg/total_avg),
            ('Transform', transform_avg/total_avg)
        ], key=lambda x: x[1])
        
        print(f"\n🔍 瓶颈: {max_component[0]} 占比{max_component[1]*100:.1f}%")
        
        if io_avg / total_avg > 0.5:
            print("   💡 建议: I/O是瓶颈，数据在网络存储上，考虑复制到本地SSD")
        elif transform_avg / total_avg > 0.5:
            print("   💡 建议: Transform是瓶颈，考虑关闭数据增强或简化transform")
        
        print(f"{'='*70}\n")
        
        # 清空记录，避免内存占用
        if len(self.timers['total']) > 1000:
            for key in self.timers:
                self.timers[key] = self.timers[key][-100:]

