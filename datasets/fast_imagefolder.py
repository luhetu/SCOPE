"""使用OpenCV替代PIL的快速ImageFolder"""
import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import make_dataset, IMG_EXTENSIONS
import cv2
try:
    # 禁用 OpenCV 内部线程，避免与 DataLoader workers 竞争
    cv2.setNumThreads(0)
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
except Exception:
    pass
import numpy as np

class FastImageFolder(Dataset):
    """使用OpenCV替代PIL的ImageFolder，在网络存储上更快"""
    
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        # 扫描所有类别
        classes, class_to_idx = self._find_classes(root)
        self.classes = classes
        self.class_to_idx = class_to_idx
        
        # 扫描所有图片
        self.samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        self.targets = [s[1] for s in self.samples]
        
    def _find_classes(self, dir):
        """查找所有类别文件夹"""
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """使用OpenCV加载图片（比PIL快2-3倍）"""
        path, target = self.samples[index]
        
        # 使用OpenCV读取（BGR格式）
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        
        if img is None:
            raise RuntimeError(f"Failed to load image: {path}")
        
        # 转换为RGB（torchvision需要RGB）
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL格式（因为transform需要PIL）
        from PIL import Image
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target


class UltraFastImageFolder(Dataset):
    """完全绕过PIL，直接用OpenCV+numpy，最快但需要调整transform"""
    
    def __init__(self, root, transform=None, size=224):
        self.root = root
        self.size = size
        self.transform = transform
        
        classes, class_to_idx = self._find_classes(root)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        self.targets = [s[1] for s in self.samples]
        
        # ImageNet均值和标准差
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """完全使用OpenCV+numpy处理，避免PIL"""
        path, target = self.samples[index]
        
        # OpenCV读取
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to load image: {path}")
        
        # 转RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # RandomResizedCrop using OpenCV
        h, w = img.shape[:2]
        scale = np.random.uniform(0.08, 1.0)
        aspect = np.random.uniform(3./4., 4./3.)
        target_area = h * w * scale
        target_w = int(np.sqrt(target_area * aspect))
        target_h = int(np.sqrt(target_area / aspect))
        
        if target_w <= w and target_h <= h:
            x = np.random.randint(0, w - target_w + 1)
            y = np.random.randint(0, h - target_h + 1)
            img = img[y:y+target_h, x:x+target_w]
        
        # Resize to target size
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)
        
        # 转换为float并normalize
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        
        # 转tensor
        img = torch.from_numpy(img)
        
        return img, target

