# MAML on CIFAR-100 Few-Shot Learning

本项目实现了 **MAML (Model-Agnostic Meta-Learning)** 算法，用于在 CIFAR-100 数据集上进行 few-shot learning 实验。

## 项目结构

```
maml_cifar100/
├── data/
│   ├── __init__.py
│   └── cifar100_fewshot.py    # CIFAR-100 few-shot数据加载器
├── models/
│   ├── __init__.py
│   ├── conv4.py               # 4层CNN backbone
│   ├── resnet.py              # ResNet12 backbone (更大容量)
│   └── maml.py                # MAML算法实现
├── utils/
│   ├── __init__.py
│   └── visualization.py       # 可视化工具
├── configs/                    # 配置文件目录
├── checkpoints/               # 模型保存目录
├── logs/                      # 训练日志目录
├── results/                   # 评估结果目录
├── train.py                   # MAML训练脚本
├── pretrain.py                # 预训练脚本 (用于Transfer Learning对比)
├── evaluate.py                # 评估脚本
├── baseline.py                # Baseline方法
├── compare_methods.py         # 综合对比脚本
├── run_experiments.sh         # 实验运行脚本
├── requirements.txt           # 依赖包
└── README.md
```

## 模型架构

### 1. Conv4 (默认)
- 4层卷积网络，每层包含 Conv -> BN -> ReLU -> MaxPool
- 可调节 `hidden_dim` 参数 (32/64/128) 控制模型容量
- 参数量: ~100K (hidden_dim=64) / ~400K (hidden_dim=128)

### 2. ResNet12 (更大容量)
- 4个残差块，每块包含3个卷积层
- 可配置通道数: `64,128,256,512` (默认) 或 `64,160,320,640` (大型)
- 参数量: ~8M (默认配置)
- 适合数据量充足或需要更强表达能力的场景

## 对比方法

| 方法 | 描述 |
|------|------|
| **MAML** | 学习易于适应的初始化参数 |
| **FOMAML** | MAML的一阶近似，训练更快 |
| Random+Finetune | 随机初始化后直接finetune |
| Transfer Learning | 预训练后finetune (head/full) |
| ProtoNet | 基于原型的度量学习 |
| ProtoNet (Pretrained) | 使用预训练特征的ProtoNet |

## 快速开始

### 1. 环境配置

```bash
conda create -n maml python=3.10
conda activate maml
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. 运行完整实验

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

### 3. 单独训练

```bash
# Conv4 (hidden_dim=64) - 基础配置
python train.py --model conv4 --hidden_dim 64 --n_way 5 --k_shot 1

# Conv4 (hidden_dim=128) - 更大模型，减少欠拟合
python train.py --model conv4 --hidden_dim 128 --n_way 5 --k_shot 1

# 5-way 5-shot
python train.py --model conv4 --hidden_dim 128 --n_way 5 --k_shot 5

# FOMAML (一阶近似，训练更快)
python train.py --model conv4 --hidden_dim 128 --first_order
```

### 4. Transfer Learning对比

```bash
# 第一步: 预训练
python pretrain.py --model conv4 --hidden_dim 128 --epochs 100

# 第二步: 评估所有baseline (包含Transfer Learning)
python baseline.py \
    --pretrained_checkpoint checkpoints/pretrain_conv4_xxx_best.pth \
    --hidden_dim 128 \
    --method all
```

## 主要参数

### 训练参数 (train.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | conv4 | 模型架构: conv4 / resnet12 |
| `--hidden_dim` | 64 | Conv4隐藏层维度 (32/64/128) |
| `--n_way` | 5 | N-way 分类 |
| `--k_shot` | 1 | 每类样本数 (support set) |
| `--inner_lr` | 0.01 | 内层学习率 |
| `--outer_lr` | 0.001 | 外层学习率 |
| `--inner_steps` | 5 | 内层更新步数 |
| `--first_order` | False | 使用一阶近似 (FOMAML) |
| `--epochs` | 100 | 训练轮数 |

### Baseline参数 (baseline.py)

| 参数 | 说明 |
|------|------|
| `--method` | all / random_finetune / protonet / transfer_head / transfer_full |
| `--pretrained_checkpoint` | 预训练模型路径 (Transfer Learning必需) |
| `--hidden_dim` | 模型隐藏层维度 (需与预训练模型匹配) |

## 预期结果

CIFAR-100 few-shot learning 参考准确率:

| 方法 | 5-way 1-shot | 5-way 5-shot |
|------|--------------|--------------|
| Random+Finetune | ~30% | ~45% |
| ProtoNet (Random) | ~35% | ~50% |
| Transfer (Head) | ~42% | ~58% |
| Transfer (Full) | ~45% | ~62% |
| FOMAML | ~48% | ~63% |
| **MAML** | **~50%** | **~65%** |

*注: 使用 hidden_dim=128 的结果，实际结果可能因随机种子略有不同*

## 可视化输出

训练完成后自动生成:

1. **`logs/*_learning_curves.png`**: 训练/验证曲线
   - Loss曲线 (左)
   - Accuracy曲线 (右)

2. **`logs/*_test_distribution.png`**: 测试准确率分布
   - 直方图 (左)
   - 箱线图 (右)

3. **`results/*_comparison.png`**: 方法对比柱状图

## 参考文献

```bibtex
@inproceedings{finn2017model,
  title={Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks},
  author={Finn, Chelsea and Abbeel, Pieter and Levine, Sergey},
  booktitle={ICML},
  year={2017}
}
```

## License

MIT License
