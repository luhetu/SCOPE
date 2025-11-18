"""ImageFolder dataset with timing functionality"""
import time
import os
from PIL import Image
import torch
from torchvision.datasets import ImageFolder

class TimedImageFolder(ImageFolder):
    """ImageFolder with detailed timing for diagnosing data loading bottlenecks"""
    
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)
        self.timers = {
            'io_read': [],      # File I/O reading
            'decode': [],       # JPEG decoding
            'transform': [],    # Transform operations
            'total': []         # Total time
        }
        self.sample_count = 0
        self.report_interval = 100  # Report every 100 samples
        
    def __getitem__(self, index):
        """Override __getitem__ to add timing"""
        t_start = time.time()
        
        # 1. Read image path
        path, target = self.samples[index]
        
        # 2. I/O read + JPEG decode (measured together)
        t_io = time.time()
        img = Image.open(path).convert('RGB')
        decode_time = time.time() - t_io
        io_time = decode_time  # I/O and decode are together in PIL
        
        # 4. Transform operations
        t_transform = time.time()
        if self.transform is not None:
            img = self.transform(img)
        transform_time = time.time() - t_transform
        
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        total_time = time.time() - t_start
        
        # Record time
        self.timers['io_read'].append(io_time * 1000)  # Convert to ms
        self.timers['decode'].append(decode_time * 1000)
        self.timers['transform'].append(transform_time * 1000)
        self.timers['total'].append(total_time * 1000)
        
        self.sample_count += 1
        
        # Report every 100 samples
        if self.sample_count % self.report_interval == 0:
            self.print_stats()
            
        return img, target
    
    def print_stats(self):
        """Print statistics"""
        n = len(self.timers['total'])
        if n == 0:
            return
            
        print(f"\n{'='*70}")
        print(f"📊 DataLoader Performance Analysis (last {self.report_interval} samples)")
        print(f"{'='*70}")
        print(f"{'Stage':<15} {'Avg Time':>12} {'Ratio':>8} {'Max Time':>12}")
        print(f"{'-'*70}")
        
        # Calculate average of recent 100 samples
        recent = min(self.report_interval, n)
        
        io_avg = sum(self.timers['io_read'][-recent:]) / recent
        decode_avg = sum(self.timers['decode'][-recent:]) / recent
        transform_avg = sum(self.timers['transform'][-recent:]) / recent
        total_avg = sum(self.timers['total'][-recent:]) / recent
        
        io_max = max(self.timers['io_read'][-recent:])
        decode_max = max(self.timers['decode'][-recent:])
        transform_max = max(self.timers['transform'][-recent:])
        total_max = max(self.timers['total'][-recent:])
        
        print(f"{'I/O+Decode':<15} {decode_avg:>10.1f}ms {decode_avg/total_avg*100:>7.1f}% {decode_max:>10.1f}ms")
        print(f"{'Transform':<15} {transform_avg:>10.1f}ms {transform_avg/total_avg*100:>7.1f}% {transform_max:>10.1f}ms")
        print(f"{'-'*70}")
        print(f"{'Total':<15} {total_avg:>10.1f}ms/sample  Max: {total_max:.1f}ms")
        
        # Bottleneck analysis
        max_component = max([
            ('I/O+Decode', decode_avg/total_avg),
            ('Transform', transform_avg/total_avg)
        ], key=lambda x: x[1])
        
        print(f"\n🔍 Bottleneck: {max_component[0]} ratio {max_component[1]*100:.1f}%")
        
        if io_avg / total_avg > 0.5:
            print("   💡 Suggestion: I/O is bottleneck, data on network storage, consider copying to local SSD")
        elif transform_avg / total_avg > 0.5:
            print("   💡 Suggestion: Transform is bottleneck, consider disabling data augmentation or simplifying transform")
        
        print(f"{'='*70}\n")
        
        # Clear records to avoid memory usage
        if len(self.timers['total']) > 1000:
            for key in self.timers:
                self.timers[key] = self.timers[key][-100:]

