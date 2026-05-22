# CIFAR-100 联邦学习三方法对比实验 — 设计文档

日期: 2026-05-22

## 1. 目标

新增一组联邦学习对比实验，产出 **两幅图**：
- 图1: 测试 **loss** vs 通信轮次（3 条曲线）
- 图2: 测试 **准确率** vs 通信轮次（3 条曲线）

对比三种方法，期望排序：
- FedAvg：≈80%
- FedAvg + MAML（Per-FedAvg）：≈90%，高于 FedAvg
- FedAvg + Meta-SGD：≥ MAML，且更稳定（曲线方差更小）

**诚实性说明**：稳健可复现的结论是 **排序与稳定性**（FedAvg < +MAML ≤ +Meta-SGD，Meta-SGD 方差最小）。具体阈值（80/90）依赖调参，文档中据实呈现，不强行凑数。

## 2. 任务设定（已与用户确认）

- **数据**：CIFAR-100 固定取 20 个类（`seed=42` 选定），重映射为绝对 20-way 标签。
- **划分**：训练数据 **IID** 均分到 `N=10` 个客户端（同分布）。每客户端数据量较小（`samples_per_client`，默认 250），制造"小数据"场景。
- **骨干**：**ResNet12**（channels `[64,128,256,512]`，`n_way=20`）。三方法共用同一架构，保证公平。
- **评测协议**：Per-FedAvg **adapt-then-evaluate**（Fallah et al., 2020）。从测试集每类取 `K_shot` 张作 support，模型适应后在剩余 query 上评测；多 episode 求均值。
- **差距机制（已确认）**：IID + 每客户端小数据量 + K-shot 小样本适应评测。元学习"快速适应"优势在适应数据少时最明显，排序诚实。

## 3. 三个方法

| 方法 | 训练 | 评测（adapt-then-eval） |
|---|---|---|
| **FedAvg** | 标准联邦平均（现有 `FedAvg` 类，ResNet12 标准版） | 全局模型在 support 上 SGD 微调 `inner_steps` 步 → 评 query |
| **FedAvg+MAML**（Per-FedAvg） | 客户端内层在本地 support 适应、外层在本地 query 计算 meta-loss，服务器聚合参数增量 θ | 内层适应 `inner_steps` 步 → 评 query |
| **FedAvg+Meta-SGD** | 同 MAML，但内层用可学习逐参数学习率 α（`θ' = θ − α ⊙ ∇θL`），服务器聚合 θ 和 α | 同上 |

MAML / Meta-SGD 使用 `ResNet12Functional`（`n_way=20`）；FedAvg 用同架构的 `ResNet12` 标准版。

## 4. 代码结构（尽量复用现有，最小改动）

### 新增 `data/cifar100_federated.py`
- `select_classes(num_classes=20, seed=42)`：固定选类。
- `build_federated_loaders(root, num_classes, num_clients, samples_per_client, batch_size, iid=True, seed)`：返回
  - `client_loaders: List[DataLoader]`（每客户端本地训练数据，绝对 0..19 标签）
  - `test_loader: DataLoader`（全局测试集，FedAvg 直接评测用）
- `EvalEpisodeSampler`：从测试集每类抽 `K_shot` support + 其余作 query，`sample_episode()` 返回 `(support_x, support_y, query_x, query_y)`；支持多 episode。
- 数据增强沿用 `cifar100_fewshot.py` 的归一化常数。

### 扩展 `models/federated.py`
- 新增 `FedPerMAML`（固定类别版 Per-FedAvg）：
  - 客户端从本地 DataLoader 取 batch，拆 support/query（按 K_shot per class，或随机半分），内层适应 → 外层 meta-loss → 反传到本地参数 → 返回参数增量。
  - `aggregate_updates()` 加权平均增量（复用现有逻辑）。
  - `adapt_and_evaluate(support, query, inner_steps)` 供评测。
- 新增 `FedPerMetaSGD`：在 `FedPerMAML` 基础上维护可学习 α（与 θ 同形状），内层用 `θ − α⊙g`，服务器同时聚合 θ 与 α 的增量。
- 保留现有 `FedAvg`、`FedMAML` 不动。

### 新增 `train_cifar_fed_compare.py`
- 参数：`--num_classes 20 --num_clients 10 --clients_per_round 5 --rounds 100 --local_epochs 3 --local_meta_steps 5 --inner_lr 0.01 --outer_lr 0.001 --inner_steps 5 --k_shot 5 --samples_per_client 250 --eval_every 2 --seed 42 --save_dir ./results --smoke`
- 用**相同 seed + 相同客户端数据划分**依次训练三方法。
- 每 `eval_every` 轮对全局/元模型做 adapt-then-eval（同一组 eval episodes），记录 `(round, loss, acc, acc_std)`。
- 保存合并 history 到 `results/cifar_fed_compare_<timestamp>.json`。
- `--smoke`：`num_classes=5, num_clients=3, rounds=3, samples_per_client=60, k_shot=2`，用于本机 **CPU** 端到端正确性验证。

### 扩展 `utils/visualization.py`
- `plot_fed_comparison(history, save_dir, prefix)`：读三方法 history，输出
  - `<prefix>_loss.png`（loss vs round，3 曲线）
  - `<prefix>_acc.png`（acc vs round，3 曲线，含 ±std 阴影带体现稳定性）

### 新增 `run_cifar_fed_compare.sh`
- GPU 服务器（RTX 4090）完整配置启动脚本，对齐现有 `run_federated_experiments.sh` 风格。

## 5. 计算与环境约束

- **本机为 CPU-only**（torch 2.10.0+cpu，无 CUDA）。完整 ResNet12 × 3 方法 × 100 轮训练不可在本机完成。
- 本机职责：开发 + `--smoke` CPU 端到端验证（确保三方法、评测、绘图全链路无误）。
- **完整训练在 GPU 服务器执行**（沿用项目既有远程 GPU 工作流）。
- 输出图与 history 存 `./results`（项目既有约定）。

## 6. 验证标准

- `--smoke` 在本机 CPU 跑通：三方法各完成 3 轮、产出 2 幅 PNG、history JSON 字段完整、无异常。
- 代码遵循现有接口（`ResNet12Functional.forward(x, vars, bn_training)`、`vars`/`alpha` 约定）。
- 完整运行后核对图中排序与稳定性趋势（在 GPU 服务器上由用户执行/确认）。

## 7. 不做的事（YAGNI）

- 不引入 non-IID/Dirichlet（用户已选 IID）。
- 不改动 bearing 相关脚本与现有 few-shot 流程。
- 不新增除上述外的方法或消融。
