# 🚀 在 NCC 上安装检测环境 - 快速修复

## ❌ 问题

运行 `pip install -r requirements_detection.txt` 时报错：
```
ERROR: Could not find a version that satisfies the requirement torch==1.9.0+cu111
```

## ✅ 解决方案

**不要直接用 requirements_detection.txt！** 需要分步安装。

---

## 📦 方法 1: 使用自动化脚本（推荐）

```bash
# 在 SCOPE 目录下
cd ~/SCOPE

# 确保已激活环境
conda activate vitseg

# 运行安装脚本
bash scripts/install_detection_env.sh
```

脚本会自动完成：
1. ✅ 安装 PyTorch 1.9.0+cu111（使用正确的源）
2. ✅ 安装 MMCV-Full 1.3.17（预编译版本）
3. ✅ 安装其他依赖
4. ✅ 验证所有包

---

## 📦 方法 2: 手动分步安装

```bash
# 激活环境
conda activate vitseg

# ==================== 步骤 1: 安装 PyTorch ====================
# 使用 -f 参数指定 PyTorch 的特殊索引
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 验证 PyTorch
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"

# ==================== 步骤 2: 安装 MMCV-Full ====================
# 使用预编译的轮子（更快，避免编译错误）
pip install mmcv-full==1.3.17 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# 验证 MMCV
python -c "from mmcv import _ext; print('MMCV OK')"

# ==================== 步骤 3: 安装其他依赖 ====================
# 安装剩余的包
pip install opencv-python Pillow matplotlib
pip install pycocotools terminaltables
pip install timm==0.6.12 einops
pip install wandb tensorboard
pip install numpy scipy scikit-learn
pip install tqdm pyyaml
```

---

## 🔍 验证安装

```bash
python << 'PYEOF'
import torch
import torchvision
import mmcv
import cv2
import timm
import einops

print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ TorchVision: {torchvision.__version__}")
print(f"✅ MMCV: {mmcv.__version__}")
print(f"✅ CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
PYEOF
```

---

## �� 为什么会出现这个问题？

`requirements_detection.txt` 中包含了 `torch==1.9.0+cu111`，这是一个特殊的 PyTorch 版本，带有 CUDA 后缀。

**标准的 pip 源** (pypi.org) 只有通用版本：
- ✅ `torch==1.9.0` (通用版，不包含 CUDA)
- ❌ `torch==1.9.0+cu111` (找不到)

**PyTorch 官方源** 才有 CUDA 特定版本：
- ✅ `torch==1.9.0+cu111` (CUDA 11.1)
- ✅ `torch==1.9.0+cu102` (CUDA 10.2)

所以需要用 `-f` 参数指定 PyTorch 的官方索引。

---

## 🎯 安装完成后

```bash
# 修改配置文件的数据路径
vim configs/detection_swin.yaml
# 改为: data_dir: /your/path/to/coco

# 运行检测训练
python train.py --cfg configs/detection_swin.yaml

# 或提交到 SLURM
sbatch scripts/submit_slurm_det.sh configs/detection_swin.yaml
```

---

## ⚠️ 常见问题

### 问题 1: MMCV 安装失败

```bash
# 如果预编译版本不可用，可以从源码编译（慢）
pip install mmcv-full==1.3.17
# 需要确保有 gcc, g++, nvcc
```

### 问题 2: CUDA 版本不匹配

```bash
# 检查 NCC 的 CUDA 版本
module load cuda
nvcc --version

# 根据 CUDA 版本选择对应的 PyTorch
# CUDA 11.1: torch==1.9.0+cu111
# CUDA 10.2: torch==1.9.0+cu102
```

### 问题 3: 网络问题

```bash
# 使用清华镜像
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -f https://download.pytorch.org/whl/torch_stable.html
```

---

**问题解决了吗？** 如有其他问题，查看 `NCC_TUTORIAL.md` 的第十一节"常见问题"。
