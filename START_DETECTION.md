# 🎯 开始检测训练 - 完整指南

## 📋 前置条件检查

### ✅ 你需要的东西
- [ ] COCO 数据集
- [ ] 检测环境（Python 3.7 + PyTorch 1.9 + MMCV）
- [ ] 足够的 GPU 显存（至少 8GB）

---

## 🚀 方法 A: 本地训练

### 步骤 1: 切换到检测环境

```bash
# 激活检测环境
source venv_swin_det/bin/activate

# 或者如果还没创建，先创建：
python3.7 -m venv venv_swin_det
source venv_swin_det/bin/activate

# 安装依赖（如果还没装）
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.17 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install opencv-python Pillow pycocotools timm einops wandb
```

### 步骤 2: 准备 COCO 数据集

COCO 数据集结构应该是：
```
datasets/coco/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
│   ├── 000000000009.jpg
│   └── ...
└── val2017/
    ├── 000000000139.jpg
    └── ...
```

**如果还没下载**：
```bash
# 下载 COCO 数据集（约 20GB）
cd datasets/
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```

### 步骤 3: 修改配置文件

```bash
# 编辑检测配置
vim configs/detection_swin.yaml
```

**修改 `data_dir`**：
```yaml
data_dir: /home/hetu/ImageNet  # 改为 COCO 路径
# 改成：
data_dir: /home/hetu/MY project/SCOPE/datasets/coco
```

### 步骤 4: 开始训练

```bash
# 推荐：Swin + Mask R-CNN（显存占用适中）
python train.py --cfg configs/detection_swin.yaml

# 或者：ViT Tiny + Mask R-CNN（显存占用小，但需要小图）
python train.py --cfg configs/detection_vit.yaml
```

---

## ☁️ 方法 B: NCC 集群训练（推荐）

### 步骤 1: 在 NCC 上拉取最新代码

```bash
# 登录 NCC
ssh your_username@ncc.your_school.edu

# 进入项目目录
cd ~/SCOPE

# 拉取最新代码
git pull origin main
```

### 步骤 2: 创建检测环境

```bash
# 创建 conda 环境（Python 3.7）
conda create -n scope_det python=3.7 -y
conda activate scope_det

# 加载 CUDA 模块
module load cuda/11.1
module load cudnn/8.0

# 安装依赖
bash scripts/install_detection_env.sh
```

### 步骤 3: 准备数据集

**选项 A: 上传数据**
```bash
# 在本地压缩
tar -czf coco.tar.gz datasets/coco/

# 上传到 NCC
scp coco.tar.gz your_username@ncc:/scratch/your_username/

# 在 NCC 上解压
cd /scratch/your_username/
tar -xzf coco.tar.gz
```

**选项 B: 使用共享数据**
```bash
# 检查是否有共享 COCO
ls /data/datasets/COCO

# 创建软链接
ln -s /data/datasets/COCO ~/SCOPE/datasets/coco
```

### 步骤 4: 修改配置文件

```bash
cd ~/SCOPE
vim configs/detection_swin.yaml
```

修改数据路径：
```yaml
data_dir: /scratch/your_username/coco
# 或
data_dir: /data/datasets/COCO
```

### 步骤 5: 提交训练任务

```bash
# 提交 Swin 检测任务
sbatch scripts/submit_slurm_det.sh configs/detection_swin.yaml

# 查看任务状态
squeue -u your_username

# 查看日志
tail -f logs/scope_det_*.out
```

---

## 🎯 推荐的模型配置

### 1. Swin + Mask R-CNN（推荐首选）

**配置文件**: `configs/detection_swin.yaml`

**优势**:
- ✅ 性能最好（APb ~42, APm ~38）
- ✅ 显存占用适中（12GB）
- ✅ 训练稳定

**适合**:
- 有 12GB+ GPU
- 追求最佳性能
- 标准 COCO 训练

```bash
python train.py --cfg configs/detection_swin.yaml
```

---

### 2. ViT Tiny + Mask R-CNN（资源受限）

**配置文件**: `configs/detection_vit.yaml`

**优势**:
- ✅ 显存占用小（4-8GB）
- ✅ 训练快
- ✅ 可在小 GPU 上跑

**劣势**:
- ❌ 性能较低（需要小分辨率图像）
- ❌ 不适合大图

```bash
python train.py --cfg configs/detection_vit.yaml
```

---

## 📊 配置文件对比

| 模型 | 配置文件 | Batch Size | 图像尺寸 | 显存需求 | 预期 APb |
|------|---------|-----------|----------|---------|---------|
| Swin | `detection_swin.yaml` | 2 | 1333×800 | 12GB | ~42 |
| ViT Tiny | `detection_vit.yaml` | 1 | 512×512 | 4GB | ~30 |
| CoPE | `detection_vitcope.yaml` | 1 | 512×512 | 4GB | ~30 |
| SCoPE | `detection_vitscope.yaml` | 1 | 512×512 | 4GB | ~30 |

---

## 🔍 训练监控

### 查看训练进度

```bash
# 实时查看日志
tail -f logs/scope_det_*.out

# 或者在训练脚本中会看到：
# Epoch [1/12] [100/500] Loss: 1.234
```

### WandB 监控（如果配置了）

```bash
# 登录 WandB（首次使用）
wandb login YOUR_API_KEY

# 训练会自动上传到：
# https://wandb.ai/your_username/scope-detection
```

### 检查点位置

```bash
# 检测训练的检查点保存在：
ls work_dirs/mask_rcnn_*/

# 最佳模型：
work_dirs/mask_rcnn_*/best_bbox_mAP_epoch_*.pth
```

---

## ⚠️ 常见问题

### 问题 1: 环境错误

```bash
# 错误：ModuleNotFoundError: No module named 'mmcv._ext'
# 解决：确认你在检测环境中
source venv_swin_det/bin/activate
# 或
conda activate scope_det
```

### 问题 2: CUDA OOM（显存不足）

```bash
# 解决方案：
# 1. 减小 batch size
vim configs/detection_swin.yaml
# bs: 2 -> bs: 1

# 2. 减小图像尺寸
# img_scale: [1333, 800] -> [800, 600]

# 3. 使用 ViT Tiny
python train.py --cfg configs/detection_vit.yaml
```

### 问题 3: 数据集路径错误

```bash
# 错误：FileNotFoundError: COCO dataset not found
# 解决：检查配置文件中的 data_dir
vim configs/detection_swin.yaml

# 确认路径正确：
ls /path/to/coco/annotations/instances_train2017.json
```

---

## 🎯 快速开始命令总结

### 本地训练
```bash
# 1. 切换环境
source venv_swin_det/bin/activate

# 2. 修改数据路径
vim configs/detection_swin.yaml

# 3. 开始训练
python train.py --cfg configs/detection_swin.yaml
```

### NCC 集群
```bash
# 1. 准备环境
cd ~/SCOPE
git pull
conda activate scope_det

# 2. 修改配置
vim configs/detection_swin.yaml

# 3. 提交任务
sbatch scripts/submit_slurm_det.sh configs/detection_swin.yaml

# 4. 监控
squeue -u $USER
tail -f logs/scope_det_*.out
```

---

## 📚 参考文档

- 环境安装：`INSTALL_DETECTION_ON_NCC.md`
- 集群部署：`NCC_TUTORIAL.md`
- 快速修复：`QUICK_FIX_NCC.sh`

---

**下一步推荐**：
1. 先用 Swin 验证检测流程是否正常
2. 成功后再尝试 ViT/CoPE/SCoPE 变体
3. 对比不同 backbone 的检测性能

祝训练顺利！🚀

