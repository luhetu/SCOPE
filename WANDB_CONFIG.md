# WandB 统一配置说明

## 📊 项目组织结构

所有实验按**数据集**进行组织，相同数据集的所有模型实验都上传到同一个WandB项目中。

### 项目命名规则
```
{dataset}-experiments
```

### 运行命名规则

**分类任务：**
```
{model}_{dataset}_size{size}_patch{patch}_dim{dim}_depth{depth}_lr{lr}
```

**检测任务：**
```
{model}_maskrcnn_size{size}_bs{bs}_lr{lr}
```

**分割任务：**
```
{model}_upernet_size{size}_bs{bs}_lr{lr}
```

## 🎯 WandB 项目映射

| 数据集 | WandB 项目名 | 包含的模型 |
|--------|-------------|-----------|
| CIFAR-10 | `cifar10-experiments` | vit, vitcope, vitcope_embed, vitcope_embedfull, vitscope, swin |
| CIFAR-100 | `cifar100-experiments` | vit, vitcope, vitcope_embed, vitcope_embedfull, vitscope, swin |
| ImageNet | `imagenet-experiments` | vit, vitcope, vitcope_embed, vitcope_embedfull, vitscope, swin |
| COCO | `coco-experiments` | swin+maskrcnn, vit+maskrcnn, vitcope+maskrcnn, vitscope+maskrcnn |
| ADE20K | `ade20k-experiments` | swin+upernet, vit+upernet, vitcope+upernet, vitscope+upernet |

## 📝 示例

### CIFAR-10 分类实验
```yaml
# configs/vitcope_embed_cifar10.yaml
dataset: cifar10  # 决定WandB项目名: cifar10-experiments
model: vitcope_embed
size: 32
patch: 4
dim: 512
depth: 6
lr: 1e-4
```
**WandB运行名：** `vitcope_embed_cifar10_size32_patch4_dim512_depth6_lr0.0001`

### ImageNet 分类实验
```yaml
# configs/vitcope_embed.yaml
dataset: imagenet  # 或省略（默认为imagenet）
model: vitcope_embed
size: 224
patch: 32
dim: 384
depth: 12
lr: 3e-4
```
**WandB运行名：** `vitcope_embed_imagenet_size224_patch32_dim384_depth12_lr0.0003`

### COCO 检测实验
```yaml
# configs/detection_swin.yaml
# 自动识别为COCO数据集
model: swin
size: 800
bs: 2
lr: 0.0001
```
**WandB运行名：** `swin_maskrcnn_size800_bs2_lr0.0001`

## 🔍 优势

### 1. **按数据集分组**
- 同一数据集的所有模型实验在同一个项目中，便于横向对比
- 例如：在 `cifar10-experiments` 中可以直接对比 vit、vitcope_embed、vitcope_embedfull 的性能

### 2. **运行名称信息丰富**
- 包含关键超参数，一眼就能识别实验配置
- 便于快速定位特定配置的实验

### 3. **易于管理**
- 不会产生过多的WandB项目
- 数据集级别的组织更符合科研习惯

## 🎨 WandB Dashboard 视图示例

```
cifar10-experiments/
├── vit_cifar10_size32_patch4_dim512_depth6_lr0.0001
├── vitcope_embed_cifar10_size32_patch4_dim512_depth6_lr0.0001
├── vitcope_embedfull_cifar10_size32_patch4_dim512_depth6_lr0.0001
└── swin_cifar10_size32_patch4_dim96_depth6_lr0.0001

imagenet-experiments/
├── vit_imagenet_size224_patch16_dim192_depth12_lr0.0003
├── vitcope_embed_imagenet_size224_patch32_dim384_depth12_lr0.0003
└── swin_imagenet_size224_patch4_dim96_depth12_lr0.0003

coco-experiments/
├── swin_maskrcnn_size800_bs2_lr0.0001
└── vit_maskrcnn_size512_bs4_lr0.0001
```

## 🔧 如何使用

### 1. 分类任务
确保配置文件中包含 `dataset` 字段：
```yaml
dataset: cifar10  # 或 cifar100, imagenet
```

### 2. 检测任务
自动使用 COCO 数据集，项目名为 `coco-experiments`

### 3. 分割任务
自动使用 ADE20K 数据集，项目名为 `ade20k-experiments`

### 4. 禁用 WandB
```yaml
nowandb: true
```

## 📚 相关文件

- 分类任务：`tasks/classification.py`
- 检测任务：`tasks/detection.py`
- 分割任务：`tasks/segmentation.py`

所有任务的WandB配置逻辑已统一！🎉

