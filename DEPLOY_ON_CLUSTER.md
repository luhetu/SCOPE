# 🖥️ 在学校计算集群（NCC）上部署 SCOPE

本指南帮助你在学校的计算集群（如 NCC）上部署和运行 SCOPE 项目。

---

## 📋 前置准备

### 1. 确认集群信息
```bash
# 登录集群
ssh your_username@ncc.your_school.edu

# 查看可用的 GPU
nvidia-smi
sinfo -o "%20N %10c %10m %25f %10G"  # SLURM
pbsnodes -a | grep gpu  # PBS

# 查看 Python 和 CUDA 模块
module avail python
module avail cuda
```

### 2. 克隆项目
```bash
# 在你的家目录或项目目录
cd $HOME  # 或 cd /scratch/your_username
git clone https://github.com/luhetu/SCOPE.git
cd SCOPE
```

---

## ⚙️ 环境设置

### 方案 A: 使用集群的 Module 系统（推荐）

```bash
# 加载模块
module load python/3.7  # 或其他可用版本
module load cuda/11.1   # 或其他可用版本
module load cudnn/8.0

# 创建虚拟环境
python3 -m venv ~/venv/scope_cls
source ~/venv/scope_cls/bin/activate

# 安装依赖（分类）
pip install -r requirements_classification.txt
```

### 方案 B: 使用 Conda（如果集群支持）

```bash
# 加载 Conda 模块
module load anaconda3

# 创建环境
conda create -n scope_cls python=3.12
conda activate scope_cls

# 安装 PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements_classification.txt
```

### 检测/分割环境

```bash
# Python 3.7 + PyTorch 1.9
module load python/3.7 cuda/11.1

# 创建检测环境
python3.7 -m venv ~/venv/scope_det
source ~/venv/scope_det/bin/activate

# 安装 PyTorch 1.9
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 安装 MMCV
pip install mmcv-full==1.3.17 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# 安装其他依赖
pip install -r requirements_detection.txt
```

---

## 📦 准备数据集

### 方案 1: 从本地上传

```bash
# 在本地压缩数据
tar -czf imagenet.tar.gz /path/to/ImageNet
tar -czf coco.tar.gz /path/to/coco

# 上传到集群（使用 scp 或 rsync）
scp imagenet.tar.gz your_username@ncc.your_school.edu:/scratch/your_username/
scp coco.tar.gz your_username@ncc.your_school.edu:/scratch/your_username/

# 在集群上解压
ssh your_username@ncc.your_school.edu
cd /scratch/your_username/
tar -xzf imagenet.tar.gz
tar -xzf coco.tar.gz
```

### 方案 2: 使用集群共享数据

```bash
# 检查是否有共享数据集
ls /data/datasets/ImageNet
ls /data/datasets/COCO

# 创建软链接（不占用你的空间）
ln -s /data/datasets/ImageNet ~/SCOPE/datasets/ImageNet
ln -s /data/datasets/COCO ~/SCOPE/datasets/coco
```

### 更新配置文件

```bash
# 修改 configs/*.yaml 中的 data_dir
vim configs/vit.yaml
# 改为: data_dir: /scratch/your_username/ImageNet
# 或: data_dir: /data/datasets/ImageNet
```

---

## 🚀 提交作业

### SLURM 作业脚本

创建 `submit_classification.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=scope_vit          # 作业名称
#SBATCH --partition=gpu               # 分区/队列名称
#SBATCH --nodes=1                     # 节点数
#SBATCH --ntasks-per-node=1           # 每个节点的任务数
#SBATCH --cpus-per-task=8             # 每个任务的 CPU 核心数
#SBATCH --gres=gpu:1                  # GPU 数量（1 块）
#SBATCH --mem=32G                     # 内存
#SBATCH --time=48:00:00               # 最大运行时间 (48小时)
#SBATCH --output=logs/vit_%j.out      # 标准输出日志
#SBATCH --error=logs/vit_%j.err       # 错误输出日志
#SBATCH --mail-type=END,FAIL          # 邮件通知
#SBATCH --mail-user=your@email.com    # 你的邮箱

# 打印作业信息
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# 加载模块
module purge
module load python/3.12
module load cuda/12.1
module load cudnn/8.9

# 激活虚拟环境
source ~/venv/scope_cls/bin/activate

# 进入项目目录
cd $HOME/SCOPE

# 创建日志目录
mkdir -p logs

# 运行训练
echo "Starting training..."
python train.py --cfg configs/vit.yaml

echo "End time: $(date)"
```

创建 `submit_detection.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=scope_det
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G                     # 检测需要更多内存
#SBATCH --time=72:00:00               # 检测训练更久
#SBATCH --output=logs/det_%j.out
#SBATCH --error=logs/det_%j.err

# 加载模块
module purge
module load python/3.7
module load cuda/11.1
module load cudnn/8.0

# 激活检测环境
source ~/venv/scope_det/bin/activate

# 进入项目目录
cd $HOME/SCOPE
mkdir -p logs

# 运行检测训练
echo "Starting detection training..."
python train.py --cfg configs/detection_swin.yaml

echo "End time: $(date)"
```

### 提交作业

```bash
# 创建日志目录
mkdir -p logs

# 提交作业
sbatch submit_classification.sh
sbatch submit_detection.sh

# 查看作业状态
squeue -u your_username
squeue -j job_id  # 查看具体作业

# 查看日志（实时）
tail -f logs/vit_12345.out
tail -f logs/det_12345.out

# 取消作业
scancel job_id
scancel -u your_username  # 取消所有你的作业
```

---

## 📊 PBS/Torque 作业脚本

如果你的集群使用 PBS，创建 `submit_pbs.sh`:

```bash
#!/bin/bash
#PBS -N scope_vit
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=48:00:00
#PBS -q gpu
#PBS -j oe
#PBS -o logs/vit_$PBS_JOBID.log
#PBS -m abe
#PBS -M your@email.com

# 进入提交目录
cd $PBS_O_WORKDIR

# 加载模块
module load python/3.12 cuda/12.1

# 激活环境
source ~/venv/scope_cls/bin/activate

# 运行训练
python train.py --cfg configs/vit.yaml
```

提交：
```bash
qsub submit_pbs.sh
qstat -u your_username
qdel job_id
```

---

## 🔍 监控和调试

### 查看 GPU 使用情况

```bash
# SLURM
srun --jobid=YOUR_JOB_ID nvidia-smi

# 或登录到计算节点
ssh node_name
nvidia-smi -l 1  # 每秒刷新
```

### 查看实时日志

```bash
# 方法 1: tail
tail -f logs/vit_12345.out

# 方法 2: less（可上下滚动）
less +F logs/vit_12345.out  # Ctrl+C 停止，F 继续

# 方法 3: watch
watch -n 5 tail -20 logs/vit_12345.out
```

### 检查 WandB 日志

```bash
# 在集群上配置 WandB
source ~/venv/scope_cls/bin/activate
wandb login YOUR_API_KEY

# 或在配置中禁用 WandB
# configs/*.yaml: nowandb: true
```

---

## 💾 管理检查点

### 定期备份

```bash
# 创建备份脚本 backup_checkpoints.sh
#!/bin/bash
BACKUP_DIR=$HOME/backup/scope_checkpoints
mkdir -p $BACKUP_DIR
rsync -avz checkpoint/ $BACKUP_DIR/checkpoint_$(date +%Y%m%d)
rsync -avz work_dirs/ $BACKUP_DIR/work_dirs_$(date +%Y%m%d)
echo "Backup completed at $(date)"
```

### 下载到本地

```bash
# 在本地执行
scp -r your_username@ncc.your_school.edu:~/SCOPE/checkpoint ./
scp -r your_username@ncc.your_school.edu:~/SCOPE/work_dirs ./
```

---

## 📈 批量实验

创建 `run_all_experiments.sh`:

```bash
#!/bin/bash

# 提交所有分类实验
for model in vit vitcope vitscope swin; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${model}_cls
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/${model}_%j.out

module load python/3.12 cuda/12.1
source ~/venv/scope_cls/bin/activate
cd $HOME/SCOPE
python train.py --cfg configs/${model}.yaml
EOF
    echo "Submitted ${model} classification job"
    sleep 2
done

echo "All jobs submitted!"
```

运行：
```bash
bash run_all_experiments.sh
```

---

## 🐛 常见问题

### 问题 1: 模块加载失败
```bash
# 查看可用模块
module avail
module spider cuda  # 查找 CUDA 模块

# 有些集群用 Environment Modules
ml av  # 查看所有模块
ml python cuda  # 加载多个模块
```

### 问题 2: 内存不足
```bash
# 增加内存请求
#SBATCH --mem=64G  # 或更多

# 或减少 batch size
# configs/*.yaml: bs: 256 -> 128
```

### 问题 3: 时间限制
```bash
# 查看队列时间限制
squeue -l

# 设置 checkpoint 自动保存
# 在配置中设置保存频率
```

### 问题 4: 网络访问（下载模型）
```bash
# 有些集群计算节点无网络
# 在登录节点预下载
pip download -r requirements_classification.txt -d ~/pip_cache
pip install --no-index --find-links=~/pip_cache -r requirements_classification.txt
```

---

## 📝 最佳实践

### 1. 使用 screen/tmux
```bash
# 长时间任务用 screen
screen -S scope_train
python train.py --cfg configs/vit.yaml
# Ctrl+A D 分离，screen -r scope_train 恢复
```

### 2. 数据预处理
```bash
# 在交互式节点预处理数据
srun --pty --gres=gpu:1 --mem=16G bash
python preprocess_data.py
```

### 3. 节省配额
```bash
# 使用 rsync 增量备份
rsync -avz --delete checkpoint/ /backup/checkpoint/

# 压缩旧日志
gzip logs/*.out
```

### 4. 环境变量
```bash
# 在作业脚本中设置
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0
```

---

## 📞 集群支持

如果遇到集群相关问题：
- 查看集群文档：通常在 `https://ncc.your_school.edu/docs`
- 联系管理员：`support@ncc.your_school.edu`
- 查看集群状态：`https://ncc.your_school.edu/status`

---

## ✅ 快速检查清单

- [ ] 成功登录集群
- [ ] 克隆项目代码
- [ ] 创建虚拟环境
- [ ] 安装依赖包
- [ ] 上传/链接数据集
- [ ] 更新配置文件中的数据路径
- [ ] 创建作业脚本
- [ ] 测试提交小作业
- [ ] 查看日志确认正常运行
- [ ] 配置 WandB（可选）
- [ ] 设置定期备份

---

**祝训练顺利！** 🚀

如有问题，请参考集群文档或联系管理员。

