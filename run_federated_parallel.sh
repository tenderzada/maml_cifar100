#!/bin/bash
# 轴承故障诊断 联邦学习实验脚本 - 双GPU并行版本
# 在Ubuntu服务器 (2x RTX 4090) 上运行
# GPU 0 和 GPU 1 同时运行不同实验

mkdir -p checkpoints logs results

echo "=========================================="
echo "Federated Learning - Dual GPU Parallel"
echo "=========================================="
echo "GPU 0: FedAvg experiments"
echo "GPU 1: FedMAML experiments"
echo ""

# ========================================
# GPU 0: FedAvg 实验 (后台运行)
# ========================================
(
export CUDA_VISIBLE_DEVICES=0

echo "[GPU 0] Starting FedAvg experiments..."

# 实验1: FedAvg IID
echo "[GPU 0] FedAvg IID"
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
    --strong_augment

# 实验2: FedAvg Non-IID
echo "[GPU 0] FedAvg Non-IID"
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
    --strong_augment

echo "[GPU 0] FedAvg experiments completed!"
) &

PID_GPU0=$!

# ========================================
# GPU 1: FedMAML 实验 (后台运行)
# ========================================
(
export CUDA_VISIBLE_DEVICES=1

echo "[GPU 1] Starting FedMAML experiments..."

# 实验3: FedMAML IID 5-way 1-shot
echo "[GPU 1] FedMAML IID 5-way 1-shot"
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
    --inner_steps 5 \
    --drop_rate 0.2 \
    --weight_decay 1e-4 \
    --strong_augment

# 实验4: FedMAML Non-IID 5-way 1-shot
echo "[GPU 1] FedMAML Non-IID 5-way 1-shot"
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
    --non_iid_classes 6 \
    --drop_rate 0.2 \
    --weight_decay 1e-4 \
    --strong_augment

echo "[GPU 1] FedMAML experiments completed!"
) &

PID_GPU1=$!

# 等待两个GPU完成
echo ""
echo "Waiting for experiments to complete..."
echo "GPU 0 PID: $PID_GPU0"
echo "GPU 1 PID: $PID_GPU1"

wait $PID_GPU0
wait $PID_GPU1

echo ""
echo "=========================================="
echo "All parallel experiments completed!"
echo "=========================================="

# ========================================
# 第二轮并行实验 (5-shot)
# ========================================

echo ""
echo "Starting second round (5-shot experiments)..."

# GPU 0: FedMAML IID 5-shot
(
export CUDA_VISIBLE_DEVICES=0
echo "[GPU 0] FedMAML IID 5-way 5-shot"
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
) &

PID_GPU0=$!

# GPU 1: FedMAML Non-IID 5-shot
(
export CUDA_VISIBLE_DEVICES=1
echo "[GPU 1] FedMAML Non-IID 5-way 5-shot"
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
) &

PID_GPU1=$!

wait $PID_GPU0
wait $PID_GPU1

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
