#!/bin/bash
# 轴承故障诊断 联邦学习实验脚本
# FedAvg 和 FedAvg+MAML
# 在Ubuntu服务器 (RTX 4090) 上运行

export CUDA_VISIBLE_DEVICES=0

mkdir -p checkpoints logs results

echo "=========================================="
echo "Bearing Fault Diagnosis Federated Learning"
echo "=========================================="

# ========================================
# FedAvg 实验
# ========================================

# 实验1: FedAvg IID
echo ""
echo "[Exp 1] FedAvg IID (10 clients, 2 per round)"
python train_bearing_fedavg.py \
    --model conv1d4 \
    --hidden_dim 128 \
    --num_clients 10 \
    --clients_per_round 2 \
    --iid \
    --rounds 100 \
    --local_epochs 5 \
    --local_lr 0.01

# 实验2: FedAvg Non-IID
echo ""
echo "[Exp 2] FedAvg Non-IID (10 clients, 2 per round)"
python train_bearing_fedavg.py \
    --model conv1d4 \
    --hidden_dim 128 \
    --num_clients 10 \
    --clients_per_round 2 \
    --rounds 100 \
    --local_epochs 5 \
    --local_lr 0.01 \
    --non_iid_classes 6

# ========================================
# FedAvg + MAML 实验
# ========================================

# 实验3: FedMAML IID 5-way 1-shot
echo ""
echo "[Exp 3] FedMAML IID 5-way 1-shot"
python train_bearing_fedmaml.py \
    --model conv1d4 \
    --hidden_dim 128 \
    --n_way 5 \
    --k_shot 1 \
    --num_clients 10 \
    --clients_per_round 2 \
    --iid \
    --rounds 100 \
    --local_meta_steps 10 \
    --inner_lr 0.01 \
    --outer_lr 0.001 \
    --inner_steps 5

# 实验4: FedMAML Non-IID 5-way 1-shot
echo ""
echo "[Exp 4] FedMAML Non-IID 5-way 1-shot"
python train_bearing_fedmaml.py \
    --model conv1d4 \
    --hidden_dim 128 \
    --n_way 5 \
    --k_shot 1 \
    --num_clients 10 \
    --clients_per_round 2 \
    --rounds 100 \
    --local_meta_steps 10 \
    --inner_lr 0.01 \
    --outer_lr 0.001 \
    --inner_steps 5 \
    --non_iid_classes 6

# 实验5: FedMAML IID 5-way 5-shot
echo ""
echo "[Exp 5] FedMAML IID 5-way 5-shot"
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
    --inner_steps 5

# 实验6: FedMAML Non-IID 5-way 5-shot
echo ""
echo "[Exp 6] FedMAML Non-IID 5-way 5-shot"
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
    --non_iid_classes 6

echo ""
echo "=========================================="
echo "All federated experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - checkpoints/bearing_fedavg_*"
echo "  - checkpoints/bearing_fedmaml_*"
echo "  - logs/bearing_fedavg_*"
echo "  - logs/bearing_fedmaml_*"
