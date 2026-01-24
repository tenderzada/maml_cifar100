# Meta-Learning for Few-Shot Learning

本项目实现了多种元学习算法:

**集中式元学习:**
1. **MAML** (Model-Agnostic Meta-Learning, Finn et al., 2017)
2. **Meta-SGD** (Learning to Learn Quickly for Few-Shot Learning, Li et al., 2017)

**联邦元学习:**
3. **FedAvg** (Federated Averaging, McMahan et al., 2017)
4. **FedAvg + MAML** (联邦元学习)

支持两种数据集:
1. **CIFAR-100** - 图像分类
2. **轴承故障诊断** - 时序信号分类

## 算法对比

### 元学习算法

| 算法 | 内层更新公式 | 学习率 | 特点 |
|------|-------------|--------|------|
| MAML | θ' = θ - lr·∇θL | 固定标量 | 学习好的初始化参数 |
| Meta-SGD | θ' = θ - α⊙∇θL | 可学习向量 | 同时学习初始化和学习率 |

其中 `⊙` 表示逐元素乘法，`α` 是与参数同维度的可学习学习率向量。

### 联邦学习算法

| 算法 | 数据分布 | 特点 |
|------|----------|------|
| FedAvg | IID/Non-IID | 联邦平均，各客户端本地训练后聚合 |
| FedAvg+MAML | IID/Non-IID | 联邦元学习，客户端执行MAML后聚合元参数 |

## 项目结构

```
maml_cifar100/
├── data/
│   ├── cifar100_fewshot.py    # CIFAR-100 few-shot数据加载器
│   ├── bearing_dataset.py     # 轴承数据集原始加载器
│   ├── bearing_fewshot.py     # 轴承数据集few-shot加载器
│   └── bearing_federated.py   # 轴承数据集联邦学习加载器
├── models/
│   ├── conv4.py               # 2D CNN (图像)
│   ├── resnet.py              # ResNet12 (图像)
│   ├── conv1d.py              # 1D CNN (时序信号)
│   ├── maml.py                # MAML算法
│   ├── meta_sgd.py            # Meta-SGD算法
│   └── federated.py           # FedAvg/FedMAML算法
├── utils/
│   └── visualization.py       # 可视化工具
├── train.py                   # CIFAR-100 MAML训练
├── train_bearing.py           # 轴承数据 MAML训练
├── train_bearing_metasgd.py   # 轴承数据 Meta-SGD训练
├── train_bearing_fedavg.py    # 轴承数据 FedAvg训练
├── train_bearing_fedmaml.py   # 轴承数据 FedAvg+MAML训练
├── pretrain.py                # 预训练 (Transfer Learning)
├── evaluate.py                # 评估脚本
├── baseline.py                # Baseline方法
├── run_experiments.sh         # CIFAR-100实验脚本
├── run_bearing_experiments.sh # 轴承MAML/Meta-SGD实验脚本
├── run_federated_experiments.sh # 轴承联邦学习实验脚本
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

# 5-way 5-shot MAML with regularization
python train_bearing.py --model conv1d4 --hidden_dim 128 --n_way 5 --k_shot 5 \
    --drop_rate 0.2 --weight_decay 1e-4 --strong_augment

# 5-way 1-shot Meta-SGD
python train_bearing_metasgd.py --model conv1d4 --hidden_dim 128 --n_way 5 --k_shot 1

# 5-way 5-shot Meta-SGD with regularization
python train_bearing_metasgd.py --model conv1d4 --hidden_dim 128 --n_way 5 --k_shot 5 \
    --drop_rate 0.2 --weight_decay 1e-4 --strong_augment

# 运行完整实验 (MAML + Meta-SGD)
./run_bearing_experiments.sh
```

### 联邦学习实验

```bash
# FedAvg IID (10客户端，每轮2个)
python train_bearing_fedavg.py \
    --num_clients 10 --clients_per_round 2 --iid

# FedAvg Non-IID
python train_bearing_fedavg.py \
    --num_clients 10 --clients_per_round 2 --non_iid_classes 6

# FedAvg+MAML IID 5-way 1-shot
python train_bearing_fedmaml.py \
    --n_way 5 --k_shot 1 --iid \
    --num_clients 10 --clients_per_round 2

# FedAvg+MAML Non-IID 5-way 5-shot
python train_bearing_fedmaml.py \
    --n_way 5 --k_shot 5 \
    --num_clients 10 --clients_per_round 2 --non_iid_classes 6

# 运行完整联邦学习实验
./run_federated_experiments.sh
```

## 主要参数

### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | conv4/conv1d4 | 模型架构 |
| `--hidden_dim` | 64 | 隐藏层维度 |
| `--n_way` | 5 | N-way分类 |
| `--k_shot` | 1 | 每类样本数 |
| `--outer_lr` | 0.001 | 外层学习率 |
| `--epochs` | 100 | 训练轮数 |

### MAML特有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--inner_lr` | 0.01 | 内层学习率 (固定标量) |
| `--inner_steps` | 5 | 内层更新步数 |
| `--first_order` | False | 使用FOMAML |

### Meta-SGD特有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--inner_lr_init` | 0.01 | 学习率向量初始值 |
| `--alpha_lr` | 0.001 | 学习率向量的学习率 |
| `--inner_steps` | 1 | 内层更新步数 (通常为1) |

### 正则化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--drop_rate` | 0.0 | Dropout率 (推荐0.1-0.3) |
| `--weight_decay` | 0.0 | 权重衰减 (推荐1e-4) |
| `--strong_augment` | False | 使用强数据增强 |

### 联邦学习参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_clients` | 10 | 客户端总数 |
| `--clients_per_round` | 2 | 每轮选择的客户端数 |
| `--iid` | False | 使用IID数据分布 |
| `--non_iid_classes` | 6 | Non-IID时每客户端类别数 |
| `--rounds` | 100 | 通信轮数 |
| `--local_epochs` | 5 | FedAvg本地训练轮数 |
| `--local_meta_steps` | 10 | FedMAML本地元更新步数 |

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

@article{li2017meta,
  title={Meta-SGD: Learning to Learn Quickly for Few-Shot Learning},
  author={Li, Zhenguo and Zhou, Fengwei and Chen, Fei and Li, Hang},
  journal={arXiv preprint arXiv:1707.09835},
  year={2017}
}

@inproceedings{mcmahan2017communication,
  title={Communication-Efficient Learning of Deep Networks from Decentralized Data},
  author={McMahan, Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and y Arcas, Blaise Aguera},
  booktitle={AISTATS},
  year={2017}
}
```

## License

MIT License
