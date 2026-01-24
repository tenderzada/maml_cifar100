#!/bin/bash
# 轴承故障诊断 联邦学习实验脚本 - 数据并行版本
# 在Ubuntu服务器 (2x RTX 4090) 上运行
# 使用DataParallel让两张GPU同时处理同一个实验

mkdir -p checkpoints logs results

echo "=========================================="
echo "Federated Learning - Data Parallel (2x GPU)"
echo "=========================================="
echo ""

# ========================================
# FedAvg 实验 (数据并行)
# ========================================

echo "=========================================="
echo "FedAvg Experiments (DataParallel)"
echo "=========================================="

# 实验1: FedAvg IID
echo "[DataParallel] FedAvg IID"
python train_bearing_fedavg.py \
    --model conv1d4 \
    --hidden_dim 128 \
    --num_clients 10 \
    --clients_per_round 2 \
    --iid \
    --rounds 100 \
    --local_epochs 5 \
    --local_lr 0.01 \
    --drop_rate 0.2 \
    --weight_decay 1e-4 \
    --strong_augment \
    --data_parallel

# 实验2: FedAvg Non-IID
echo "[DataParallel] FedAvg Non-IID"
python train_bearing_fedavg.py \
    --model conv1d4 \
    --hidden_dim 128 \
    --num_clients 10 \
    --clients_per_round 2 \
    --rounds 100 \
    --local_epochs 5 \
    --local_lr 0.01 \
    --non_iid_classes 6 \
    --drop_rate 0.2 \
    --weight_decay 1e-4 \
    --strong_augment \
    --data_parallel

echo "FedAvg experiments completed!"
echo ""

# ========================================
# FedMAML 实验 (1-shot)
# ========================================

# echo "=========================================="
# echo "FedMAML 1-shot Experiments"
# echo "=========================================="

# # 实验3: FedMAML IID 5-way 1-shot
# echo "FedMAML IID 5-way 1-shot"
# python train_bearing_fedmaml.py \
#     --model conv1d4 \
#     --hidden_dim 128 \
#     --n_way 5 \
#     --k_shot 1 \
#     --num_clients 10 \
#     --clients_per_round 2 \
#     --iid \
#     --rounds 100 \
#     --local_meta_steps 10 \
#     --inner_lr 0.01 \
#     --outer_lr 0.001 \
#     --inner_steps 5 \
#     --drop_rate 0.2 \
#     --weight_decay 1e-4 \
#     --strong_augment

# # 实验4: FedMAML Non-IID 5-way 1-shot
# echo "FedMAML Non-IID 5-way 1-shot"
# python train_bearing_fedmaml.py \
#     --model conv1d4 \
#     --hidden_dim 128 \
#     --n_way 5 \
#     --k_shot 1 \
#     --num_clients 10 \
#     --clients_per_round 2 \
#     --rounds 100 \
#     --local_meta_steps 10 \
#     --inner_lr 0.01 \
#     --outer_lr 0.001 \
#     --inner_steps 5 \
#     --non_iid_classes 6 \
#     --drop_rate 0.2 \
#     --weight_decay 1e-4 \
#     --strong_augment

# echo "FedMAML 1-shot experiments completed!"
# echo ""

# ========================================
# FedMAML 实验 (5-shot)
# ========================================

echo "=========================================="
echo "FedMAML 5-shot Experiments"
echo "=========================================="

# 实验5: FedMAML IID 5-way 5-shot
echo "FedMAML IID 5-way 5-shot"
python train_bearing_fedmaml.py \
    --model conv1d4 \
    --hidden_dim 128 \
    --n_way 5 \
    --k_shot 5 \
    --num_clients 10 \
    --clients_per_round 2 \
    --iid \
    --rounds 100 \
    --local_meta_steps 10 \
    --inner_lr 0.01 \
    --outer_lr 0.001 \
    --inner_steps 5 \
    --drop_rate 0.2 \
    --weight_decay 1e-4 \
    --strong_augment

# 实验6: FedMAML Non-IID 5-way 5-shot
echo "FedMAML Non-IID 5-way 5-shot"
python train_bearing_fedmaml.py \
    --model conv1d4 \
    --hidden_dim 128 \
    --n_way 5 \
    --k_shot 5 \
    --num_clients 10 \
    --clients_per_round 2 \
    --rounds 100 \
    --local_meta_steps 10 \
    --inner_lr 0.01 \
    --outer_lr 0.001 \
    --inner_steps 5 \
    --non_iid_classes 6 \
    --drop_rate 0.2 \
    --weight_decay 1e-4 \
    --strong_augment

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - checkpoints/bearing_fedavg_*"
echo "  - checkpoints/bearing_fedmaml_*"
echo "  - logs/bearing_fedavg_*"
echo "  - logs/bearing_fedmaml_*"
