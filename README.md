# MAML for Few-Shot Learning

本项目实现了 **MAML (Model-Agnostic Meta-Learning)** 算法，支持两种数据集:
1. **CIFAR-100** - 图像分类
2. **轴承故障诊断** - 时序信号分类

## 项目结构

```
maml_cifar100/
├── data/
│   ├── cifar100_fewshot.py    # CIFAR-100 few-shot数据加载器
│   ├── bearing_dataset.py     # 轴承数据集原始加载器
│   └── bearing_fewshot.py     # 轴承数据集few-shot加载器
├── models/
│   ├── conv4.py               # 2D CNN (图像)
│   ├── resnet.py              # ResNet12 (图像)
│   ├── conv1d.py              # 1D CNN (时序信号)
│   └── maml.py                # MAML算法
├── utils/
│   └── visualization.py       # 可视化工具
├── train.py                   # CIFAR-100训练
├── train_bearing.py           # 轴承数据训练
├── pretrain.py                # 预训练 (Transfer Learning)
├── evaluate.py                # 评估脚本
├── baseline.py                # Baseline方法
├── run_experiments.sh         # CIFAR-100实验脚本
├── run_bearing_experiments.sh # 轴承实验脚本
└── requirements.txt
```

## 数据集

### CIFAR-100 (图像)
- 数据形状: `[3, 32, 32]`
- 类别数: 100 (64 train / 16 val / 20 test)

### 轴承故障诊断 (时序信号)
- 数据形状: `[9, 2048]` (9通道，2048时间步)
- 类别数: 64 (40 train / 12 val / 12 test)
- 样本数: ~40k训练 + ~10k测试

## 模型架构

| 模型 | 数据类型 | 输入形状 | 参数量 |
|------|----------|----------|--------|
| Conv4 | 图像 | [3, 32, 32] | ~100K-400K |
| ResNet12 | 图像 | [3, 32, 32] | ~8M |
| Conv1D4 | 时序 | [9, 2048] | ~50K-200K |
| Conv1D6 | 时序 | [9, 2048] | ~150K-600K |

## 快速开始

### 环境配置

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### CIFAR-100 实验

```bash
# 5-way 1-shot MAML
python train.py --model conv4 --hidden_dim 128 --n_way 5 --k_shot 1

# 5-way 5-shot MAML
python train.py --model conv4 --hidden_dim 128 --n_way 5 --k_shot 5

# FOMAML (一阶近似)
python train.py --model conv4 --hidden_dim 128 --first_order

# 运行完整实验
./run_experiments.sh
```

### 轴承故障诊断实验

```bash
# 5-way 1-shot MAML
python train_bearing.py --model conv1d4 --hidden_dim 64 --n_way 5 --k_shot 1

# 5-way 5-shot MAML
python train_bearing.py --model conv1d4 --hidden_dim 128 --n_way 5 --k_shot 5

# 使用更深的网络
python train_bearing.py --model conv1d6 --hidden_dim 64 --n_way 5 --k_shot 1

# 运行完整实验
./run_bearing_experiments.sh
```

## 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | conv4/conv1d4 | 模型架构 |
| `--hidden_dim` | 64 | 隐藏层维度 |
| `--n_way` | 5 | N-way分类 |
| `--k_shot` | 1 | 每类样本数 |
| `--inner_lr` | 0.01 | 内层学习率 |
| `--outer_lr` | 0.001 | 外层学习率 |
| `--inner_steps` | 5 | 内层更新步数 |
| `--first_order` | False | 使用FOMAML |
| `--epochs` | 100 | 训练轮数 |

## 数据路径配置

默认数据路径: `/mnt/data/lev_data/`

- CIFAR-100: 自动下载到指定目录
- 轴承数据: 需要预先放置 `bearing_data.pkl`

```bash
# 修改数据路径
python train.py --data_root /your/path/to/data
python train_bearing.py --data_file /your/path/to/bearing_data.pkl
```

## 输出文件

训练完成后自动生成:

```
checkpoints/
├── *_best.pth              # 最佳模型
└── *_epoch*.pth            # 定期保存

logs/
├── *_config.json           # 配置文件
├── *_log.txt               # 训练日志
├── *_history.json          # 训练历史
├── *_learning_curves.png   # 学习曲线
└── *_test_distribution.png # 测试分布
```

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
