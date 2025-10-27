# 🎓 NCC 集群部署 SCOPE - 完整教程

> **非常好！** 🎉 我看到你上传了 `vitseg` 相关的 7 个核心文件（`upernet.py`, `vit.py`, `traindet.py`, `vitscope442.py`），目前这些脚本还在本地合并中——**一起以最顺在 NCC（集群）上训练这些ViT/UPerNet/ViTCoPE 模型（ADE20K / ImageNet1K）** 📌

> 你现在想在 NCC 上配置一个新的训练环境。我给你一步一步写清楚，保证干净、稳定、运行标准化。

---

## 🌟 一、环境目标

**任务**：在 ImageNet1K 或 ADE20K 上训练 **ViT / UPerNet / ViTcope** 模型

**模型**：`ViT-Tiny` 或自定义 `ViTcope`

**代码依赖**：
- PyTorch + torchvision + tqdm + wandb + einops + Pillow + numpy + matplotlib
- （检测/分割）mmcv-full + timm 等

**环境名**：`vitseg`

---

## 📦 二、在 NCC 上新建 conda 环境

### 步骤 1：登录 NCC 并进入项目目录

```bash
# 登录到 NCC 集群（根据你的集群地址修改）
ssh your_username@ncc.your_school.edu

# 进入你的工作目录，例如：
cd /home2/dnr452/vitseg/vit-pytorch-main/
```

### 步骤 2：创建 conda 环境

```bash
# 创建新环境
conda create -n vitseg python=3.10 -y

# 激活环境
conda activate vitseg
```

---

## 📚 三、安装核心包（与 `traindet.py` 完全兼容）

### 方案 A：自动安装（推荐）

```bash
# 激活环境
conda activate vitseg

# 如果 NCC 提供了 pytorch，可用官方源（推荐）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他核心依赖
pip install tqdm einops Pillow numpy matplotlib wandb
```

### 方案 B：手动安装（适配特定 CUDA 版本）

```bash
# PyTorch（这里以使用 NCC 提供的 CUDA 12.x 镜像为例）
# 若 NCC 还装有 pytorch，可用官方办法：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 其他基本依赖
pip install tqdm einops Pillow numpy matplotlib wandb
```

### ✅ 如果你还需要用 torch.compile，确保 PyTorch >= 2.1

**NCC 常常有 GPU < A100，可以跳过 compile；改为 `--enable_flash False` 或 `--enable_jit False`。**

---

## 📦 四、安装检测/分割依赖（可选，如果需要 UPerNet/MMSegmentation）

### 步骤 1：安装 MMCV-Full

```bash
# 激活环境
conda activate vitseg

# 安装 mmcv-full（根据你的 PyTorch 和 CUDA 版本选择）
# 示例：PyTorch 2.1 + CUDA 12.1
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html

# 或者如果是 PyTorch 1.9 + CUDA 11.1（用于检测）
pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

### 步骤 2：安装其他工具包

```bash
pip install timm opencv-python terminaltables cityscapesscripts
```

---

## 🗂️ 五、上传代码和数据到 NCC

### 方案 1：使用 Git（推荐）

```bash
# 在 NCC 上克隆你的项目
cd /home2/dnr452/  # 你的工作目录
git clone https://github.com/luhetu/SCOPE.git
cd SCOPE
```

### 方案 2：使用 scp 上传

```bash
# 在你的本地电脑执行
# 上传整个 SCOPE 文件夹
scp -r /home/hetu/MY\ project/SCOPE your_username@ncc.your_school.edu:/home2/dnr452/

# 或者打包后上传
cd "/home/hetu/MY project/"
tar -czf SCOPE.tar.gz SCOPE/
scp SCOPE.tar.gz your_username@ncc.your_school.edu:/home2/dnr452/

# 在 NCC 上解压
ssh your_username@ncc.your_school.edu
cd /home2/dnr452/
tar -xzf SCOPE.tar.gz
```

---

## 📂 六、上传/配置数据集

### ImageNet 数据集

```bash
# 方法 1：从本地上传（如果数据在本地）
# 本地执行：
tar -czf imagenet.tar.gz /path/to/ImageNet/
scp imagenet.tar.gz your_username@ncc.your_school.edu:/scratch/dnr452/

# NCC 上解压：
cd /scratch/dnr452/
tar -xzf imagenet.tar.gz

# 方法 2：使用集群共享数据（如果 NCC 有）
ls /data/datasets/ImageNet  # 检查是否存在
# 如果存在，创建软链接：
ln -s /data/datasets/ImageNet ~/SCOPE/datasets/ImageNet
```

### COCO 数据集（用于检测）

```bash
# 同样方法上传或链接
ln -s /data/datasets/COCO ~/SCOPE/datasets/coco
```

### ADE20K 数据集（用于分割）

```bash
# 上传或链接 ADE20K
ln -s /data/datasets/ADE20K ~/SCOPE/datasets/ADE20K
```

---

## ⚙️ 七、修改配置文件

### 1. 分类任务配置（`configs/vit.yaml`）

```bash
cd ~/SCOPE
vim configs/vit.yaml
```

**修改数据路径：**

```yaml
# ImageNet 配置
data_dir: /scratch/dnr452/ImageNet  # 改为你的数据路径
# 或
data_dir: /data/datasets/ImageNet    # 如果用共享数据

# 其他参数根据需要调整
bs: 256        # batch size，根据 GPU 显存调整
n_epochs: 100
lr: 0.001
```

### 2. 检测任务配置（`configs/detection_swin.yaml`）

```bash
vim configs/detection_swin.yaml
```

```yaml
# COCO 路径
data_dir: /scratch/dnr452/coco  # 改为你的路径

# 根据显存调整
bs: 2  # 检测通常用小 batch size
img_scale: [1333, 800]
```

### 3. 分割任务配置（`configs/seg_vit.yaml`）

```bash
vim configs/seg_vit.yaml
```

```yaml
# ADE20K 路径
data_dir: /scratch/dnr452/ADE20K  # 改为你的路径

bs: 2
n_epochs: 80
```

---

## 🚀 八、创建 SLURM 提交脚本

### 1. 分类任务脚本（`submit_vit.sh`）

```bash
cd ~/SCOPE
vim submit_vit.sh
```

**内容：**

```bash
#!/bin/bash
#SBATCH --job-name=vit_cls          # 任务名
#SBATCH --partition=gpu             # 分区名（根据你的 NCC 修改）
#SBATCH --nodes=1                   # 节点数
#SBATCH --ntasks=1                  # 任务数
#SBATCH --cpus-per-task=8           # CPU 核心数
#SBATCH --gres=gpu:1                # GPU 数量
#SBATCH --mem=32G                   # 内存
#SBATCH --time=48:00:00             # 最长运行时间
#SBATCH --output=logs/vit_%j.out    # 输出日志
#SBATCH --error=logs/vit_%j.err     # 错误日志

# 加载模块（根据 NCC 的模块系统修改）
module purge
module load cuda/12.1
module load cudnn/8.9

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh  # 或 ~/anaconda3/etc/profile.d/conda.sh
conda activate vitseg

# 显示环境信息
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "=========================================="

# 进入项目目录
cd ~/SCOPE

# 创建日志目录
mkdir -p logs

# 运行训练
python train.py --cfg configs/vit.yaml

echo "End: $(date)"
```

**保存并添加执行权限：**

```bash
chmod +x submit_vit.sh
```

### 2. 检测任务脚本（`submit_detection.sh`）

```bash
vim submit_detection.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=swin_det
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G                   # 检测需要更多内存
#SBATCH --time=72:00:00
#SBATCH --output=logs/det_%j.out

# 加载模块（检测环境需要 Python 3.7）
module purge
module load python/3.7
module load cuda/11.1
module load cudnn/8.0

# 激活检测环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scope_det  # 注意：检测用单独环境

# 进入项目
cd ~/SCOPE
mkdir -p logs

# 运行检测
python train.py --cfg configs/detection_swin.yaml

echo "End: $(date)"
```

```bash
chmod +x submit_detection.sh
```

---

## 🎯 九、提交作业并监控

### 1. 提交分类任务

```bash
cd ~/SCOPE

# 提交任务
sbatch submit_vit.sh

# 查看任务状态
squeue -u your_username

# 实时查看日志
tail -f logs/vit_12345.out  # 替换为你的 job ID
```

### 2. 提交检测任务

```bash
sbatch submit_detection.sh

# 查看任务队列
squeue -u your_username
```

### 3. 批量提交所有模型

```bash
# ViT
sbatch --job-name=vit_cls submit_vit.sh

# CoPE
sbatch submit_vit.sh --cfg configs/vitcope.yaml

# SCoPE  
sbatch submit_vit.sh --cfg configs/vitscope.yaml

# Swin
sbatch submit_vit.sh --cfg configs/swin.yaml
```

---

## 📊 十、监控训练进度

### 方法 1：查看日志文件

```bash
# 实时查看输出
tail -f logs/vit_12345.out

# 搜索关键信息
grep "Acc:" logs/vit_12345.out
grep "Loss:" logs/vit_12345.out
```

### 方法 2：WandB 在线监控

```bash
# 首次使用需要登录 WandB
conda activate vitseg
wandb login YOUR_API_KEY

# 之后训练会自动上传到 WandB
# 访问 https://wandb.ai/your_username 查看
```

### 方法 3：登录计算节点查看 GPU

```bash
# 查看你的作业在哪个节点
squeue -j 12345

# SSH 到该节点
ssh node_name

# 查看 GPU 使用情况
nvidia-smi -l 1  # 每秒刷新
```

---

## 🔧 十一、常见问题和解决方案

### 问题 1：模块加载失败

```bash
# 查看可用模块
module avail

# 搜索特定模块
module spider cuda
module spider python

# 加载模块
module load cuda/12.1
```

### 问题 2：内存不足（OOM）

**解决方案：**

1. 减少 batch size：
```bash
vim configs/vit.yaml
# bs: 256 -> bs: 128
```

2. 增加内存请求：
```bash
vim submit_vit.sh
# #SBATCH --mem=32G -> #SBATCH --mem=64G
```

### 问题 3：任务一直排队

```bash
# 查看队列状态
squeue

# 查看你的任务优先级
sprio -j your_job_id

# 选择空闲的分区
sinfo  # 查看分区状态
```

### 问题 4：数据加载慢

**优化方案：**

```bash
# 使用 /scratch 而不是 /home（scratch 通常更快）
# 在配置文件中设置：
data_dir: /scratch/your_username/ImageNet

# 或者增加 dataloader workers
vim configs/vit.yaml
# num_workers: 4 -> num_workers: 8
```

### 问题 5：CUDA 版本不匹配

```bash
# 检查当前 CUDA
module list
nvcc --version

# 重新安装匹配的 PyTorch
# 例如 CUDA 11.1：
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

---

## 📋 十二、快速检查清单

**在提交任务前，确保：**

- [ ] ✅ 已登录 NCC 集群
- [ ] ✅ 已创建 conda 环境并安装依赖
- [ ] ✅ 已上传/链接数据集
- [ ] ✅ 已修改配置文件中的数据路径
- [ ] ✅ 已创建 SLURM 提交脚本
- [ ] ✅ 已创建 `logs/` 目录
- [ ] ✅ 已测试环境（`python -c "import torch; print(torch.cuda.is_available())"`）
- [ ] ✅ 已配置 WandB（可选）
- [ ] ✅ 提交脚本有执行权限（`chmod +x`）

---

## 🎉 十三、快速命令总结

```bash
# === 登录和环境 ===
ssh your_username@ncc.your_school.edu
cd ~/SCOPE
conda activate vitseg

# === 提交训练 ===
sbatch submit_vit.sh                    # 提交 ViT 分类
sbatch submit_detection.sh              # 提交检测任务

# === 查看状态 ===
squeue -u your_username                 # 查看你的任务
squeue -j job_id                        # 查看特定任务
tail -f logs/vit_12345.out              # 实时查看日志

# === 取消任务 ===
scancel job_id                          # 取消特定任务
scancel -u your_username                # 取消所有任务

# === 下载结果 ===
# 在本地执行：
scp -r your_username@ncc:~/SCOPE/checkpoint ./
scp -r your_username@ncc:~/SCOPE/work_dirs ./
```

---

## 🌐 十四、NCC 集群特定信息

**请根据你的学校 NCC 集群填写：**

- **登录地址**: `_________________________`
- **分区名称**: `_________________________` (常见：`gpu`, `gpu-batch`, `high-mem`)
- **Python 模块**: `_________________________` (例如：`python/3.10`)
- **CUDA 模块**: `_________________________` (例如：`cuda/12.1`)
- **Conda 路径**: `_________________________` (例如：`~/miniconda3` 或 `/opt/anaconda3`)
- **共享数据路径**: `_________________________` (例如：`/data/datasets/`)
- **你的工作目录**: `_________________________` (例如：`/home2/dnr452/` 或 `/scratch/your_username/`)

---

**祝训练顺利！** 🚀

如遇问题，可以：
1. 查看 NCC 集群文档
2. 联系集群管理员
3. 或参考本项目的 `DEPLOY_ON_CLUSTER.md`

