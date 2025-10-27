# CIFAR-10/100 快速开始指南

## 📌 概述

现在您可以使用 ViTCoPE_Embed 模型在 CIFAR-10 和 CIFAR-100 数据集上进行训练了！

## 🎯 可用配置

### CIFAR-10
配置文件：`configs/vitcope_embed_cifar10.yaml`

```bash
python train.py --cfg configs/vitcope_embed_cifar10.yaml
```

### CIFAR-100  
配置文件：`configs/vitcope_embed_cifar100.yaml`

```bash
python train.py --cfg configs/vitcope_embed_cifar100.yaml
```

## 🔧 配置参数说明

两个配置文件的主要参数：

| 参数 | CIFAR-10/100 值 | 说明 |
|------|----------------|------|
| `dataset` | cifar10 / cifar100 | 数据集名称 |
| `data_dir` | ./data | 数据存储目录（自动下载） |
| `size` | 32 | 图像尺寸 |
| `patch` | 4 | Patch大小（32÷4=8，产生8×8=64个patches） |
| `bs` | 128 | Batch size |
| `n_epochs` | 200 | 训练轮数 |
| `dim` | 512 | 模型维度 |
| `depth` | 6 | Transformer层数 |
| `heads` | 8 | 注意力头数 |
| `mlp_dim` | 512 | MLP维度 |
| `dim_head` | 64 | 每个注意力头的维度 |
| `lr` | 1e-4 | 学习率 |
| `opt` | adam | 优化器 |
| `aug` | true | 启用RandAugment数据增强 |

## 📊 数据集信息

### CIFAR-10
- **类别数**: 10
- **训练集**: 50,000 张图片
- **测试集**: 10,000 张图片
- **图像尺寸**: 32×32×3
- **类别**: 飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车

### CIFAR-100
- **类别数**: 100
- **训练集**: 50,000 张图片
- **测试集**: 10,000 张图片
- **图像尺寸**: 32×32×3
- **细粒度分类**: 100个类别

## 🚀 代码改动说明

已完成以下修改以支持 CIFAR 数据集：

### 1. 数据加载器 (`datasets/classification.py`)
- ✅ 添加 `build_cifar10_loader()` - CIFAR-10数据加载
- ✅ 添加 `build_cifar100_loader()` - CIFAR-100数据加载
- ✅ 正确的归一化参数（CIFAR-10/100专用）
- ✅ 自动下载数据集

### 2. 分类任务 (`tasks/classification.py`)
- ✅ 动态类别数支持（10/100/1000）
- ✅ 根据 `dataset` 参数选择数据加载器
- ✅ 自动调整 WandB 项目名称

### 3. 配置加载 (`utils/cfg.py`)
- ✅ 添加 `dataset` 参数支持
- ✅ 添加 `dim_head` 参数支持

### 4. 配置文件
- ✅ `configs/vitcope_embed_cifar10.yaml`
- ✅ `configs/vitcope_embed_cifar100.yaml`

## 🎨 数据增强

使用与您之前代码相同的增强策略：
- RandomCrop(32, padding=4)
- RandomHorizontalFlip
- RandAugment(N=2, M=14) - 如果 `aug=true`

## 💡 使用示例

### 快速测试（CIFAR-10）
```bash
# 首次运行会自动下载数据集到 ./data 目录
python train.py --cfg configs/vitcope_embed_cifar10.yaml
```

### 修改参数运行
您也可以创建自定义配置文件或直接修改现有配置：

```yaml
# 例如，测试不同的学习率
task: cls
model: vitcope_embed
dataset: cifar10
data_dir: ./data
bs: 256  # 增大batch size
size: 32
n_epochs: 300  # 更多训练轮数
lr: 5e-4  # 调整学习率
...
```

## 📈 预期性能

CIFAR-10 上 ViT 类模型的典型准确率：
- 基线 ViT: ~85-90%
- 优化的 ViT + 增强: ~95-97%
- 目标：测试您的 ViTCoPE_Embed 改进效果！

## ⚙️ 训练监控

训练过程会：
1. 自动保存最佳模型到 `checkpoint/` 目录
2. 上传指标到 WandB（项目：`cifar10-scope` 或 `cifar100-scope`）
3. 显示实时训练进度条
4. 打印每个 epoch 的训练和验证准确率

## 🔍 下一步

1. **先在 CIFAR-10 上验证模型** - 快速迭代，训练时间短
2. **调优超参数** - 学习率、深度、维度等
3. **扩展到 CIFAR-100** - 测试在更多类别上的表现
4. **对比实验** - 与标准 ViT 对比您的 ViTCoPE_Embed 改进

祝训练顺利！🎉

