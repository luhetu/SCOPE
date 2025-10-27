# 🚀 检测/分割环境安装指南

本指南帮助你在新平台上快速复现检测和分割任务的环境。

---

## 📋 系统要求

- **Python**: 3.7
- **CUDA**: 11.1
- **GPU**: 建议 >= 8GB 显存
- **系统**: Linux (推荐 Ubuntu 18.04+)

---

## 🔧 快速安装（推荐）

### 1. 创建虚拟环境
```bash
# 使用 Python 3.7
python3.7 -m venv venv_detection
source venv_detection/bin/activate
```

### 2. 升级 pip
```bash
pip install --upgrade pip setuptools wheel
```

### 3. 安装 PyTorch (CUDA 11.1)
```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

### 4. 安装 MMCV-Full (预编译版本)
```bash
pip install mmcv-full==1.3.17 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

### 5. 安装其他依赖
```bash
pip install -r requirements_detection.txt
```

---

## 🐛 常见问题

### 问题 1: `mmcv-full` 安装失败
**原因**: MMCV 需要 CUDA 编译支持

**解决方案**:
```bash
# 安装编译依赖
pip install cython numpy==1.21.6

# 使用预编译轮子（推荐）
pip install mmcv-full==1.3.17 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

### 问题 2: `pycocotools` 安装失败
**原因**: 缺少 Cython 或编译工具

**解决方案**:
```bash
# 先安装 Cython
pip install cython

# 再安装 pycocotools
pip install pycocotools
```

### 问题 3: NumPy 版本冲突
**原因**: NumPy 2.x 与 MMCV 1.3.17 不兼容

**解决方案**:
```bash
pip install "numpy<2"
```

### 问题 4: 找不到 Python 3.7
**Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install python3.7 python3.7-venv python3.7-dev
```

**CentOS/RHEL**:
```bash
sudo yum install python37 python37-devel
```

---

## 🌏 国内镜像加速

### 使用清华镜像
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements_detection.txt
```

### MMCV 使用国内源
```bash
pip install mmcv-full==1.3.17 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

---

## ✅ 验证安装

```bash
# 激活环境
source venv_detection/bin/activate

# 验证 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 验证 MMCV
python -c "from mmcv import _ext; print('MMCV-Full installed correctly')"

# 验证 MMDetection
python -c "from mmdet.apis import train_detector; print('MMDetection OK')"

# 验证 MMSegmentation
python -c "from mmseg.apis import train_segmentor; print('MMSegmentation OK')"
```

---

## 📦 环境切换

### 分类任务（基础环境）
```bash
# 使用系统默认 Python 环境
python train.py --cfg configs/vit.yaml
```

### 检测/分割任务（检测环境）
```bash
# 激活检测环境
source venv_detection/bin/activate
python train.py --cfg configs/detection_swin.yaml
```

---

## 🔄 从旧环境导出

如果你已有工作环境，可以导出依赖：

```bash
# 激活旧环境
source venv_swin_det/bin/activate

# 导出依赖
pip freeze > requirements_exported.txt
```

---

## 📚 相关链接

- **PyTorch**: https://pytorch.org/get-started/previous-versions/
- **MMCV**: https://github.com/open-mmlab/mmcv
- **MMDetection**: https://github.com/open-mmlab/mmdetection
- **MMSegmentation**: https://github.com/open-mmlab/mmsegmentation

---

## 💡 提示

1. **推荐使用虚拟环境**，避免与系统 Python 冲突
2. **GPU 驱动**: 确保安装了与 CUDA 11.1 兼容的驱动（>= 455.23）
3. **显存要求**: 
   - Swin 检测: ~12GB
   - ViT 检测 (Tiny + 512x512): ~4GB
   - 分割任务: ~8-12GB

---

## ⚙️ Docker 支持（可选）

```dockerfile
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

RUN apt-get update && apt-get install -y \
    python3.7 python3.7-dev python3.7-venv \
    git wget

WORKDIR /workspace
COPY requirements_detection.txt .
RUN python3.7 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
        -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install mmcv-full==1.3.17 \
        -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html && \
    pip install -r requirements_detection.txt
```

---

**最后更新**: 2025-10-26

