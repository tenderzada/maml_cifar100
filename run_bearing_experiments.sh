#!/bin/bash
# 轴承故障诊断 MAML实验脚本
# 在Ubuntu服务器 (RTX 4090) 上运行

export CUDA_VISIBLE_DEVICES=0

mkdir -p checkpoints logs results

echo "=========================================="
echo "Bearing Fault Diagnosis MAML Experiments"
echo "=========================================="

# 实验1: 5-way 1-shot MAML (Conv1D4, hidden_dim=64)
# echo ""
# echo "[Exp 1] 5-way 1-shot MAML (Conv1D4-64)"
# python train_bearing.py \
#     --model conv1d4 \
#     --hidden_dim 64 \
#     --n_way 5 \
#     --k_shot 1 \
#     --epochs 100 \
#     --inner_lr 0.01 \
#     --outer_lr 0.001 \
#     --inner_steps 5

# 实验2: 5-way 1-shot MAML (Conv1D4, hidden_dim=128)
# echo ""
# echo "[Exp 2] 5-way 1-shot MAML (Conv1D4-128)"
# python train_bearing.py \
#     --model conv1d4 \
#     --hidden_dim 128 \
#     --n_way 5 \
#     --k_shot 1 \
#     --epochs 100 \
#     --inner_lr 0.01 \
#     --outer_lr 0.001 \
#     --inner_steps 5

# 实验3: 5-way 5-shot MAML
# echo ""
# echo "[Exp 3] 5-way 5-shot MAML (Conv1D4-128)"
# python train_bearing.py \
#     --model conv1d4 \
#     --hidden_dim 128 \
#     --n_way 5 \
#     --k_shot 5 \
#     --epochs 100 \
#     --inner_lr 0.01 \
#     --outer_lr 0.001 \
#     --inner_steps 5

# 实验4: 5-way 1-shot FOMAML
# echo ""
# echo "[Exp 4] 5-way 1-shot FOMAML (Conv1D4-128)"
# python train_bearing.py \
#     --model conv1d4 \
#     --hidden_dim 128 \
#     --n_way 5 \
#     --k_shot 1 \
#     --epochs 100 \
#     --inner_lr 0.01 \
#     --outer_lr 0.001 \
#     --inner_steps 5 \
#     --first_order

# 实验5: 使用更深的Conv1D6
# echo ""
# echo "[Exp 5] 5-way 1-shot MAML (Conv1D6-64)"
# python train_bearing.py \
#     --model conv1d6 \
#     --hidden_dim 64 \
#     --n_way 5 \
#     --k_shot 1 \
#     --epochs 100 \
#     --inner_lr 0.01 \
#     --outer_lr 0.001 \
#     --inner_steps 5

# ========================================
# 正则化实验 (减少过拟合)
# ========================================

# 实验6: 带Dropout的MAML
# echo ""
# echo "[Exp 6] 5-way 1-shot MAML with Dropout (Conv1D4-128)"
# python train_bearing.py \
#     --model conv1d4 \
#     --hidden_dim 128 \
#     --n_way 5 \
#     --k_shot 1 \
#     --epochs 100 \
#     --inner_lr 0.01 \
#     --outer_lr 0.001 \
#     --inner_steps 5 \
#     --drop_rate 0.2

# 实验7: 带强数据增强的MAML
# echo ""
# echo "[Exp 7] 5-way 1-shot MAML with Strong Augmentation (Conv1D4-128)"
# python train_bearing.py \
#     --model conv1d4 \
#     --hidden_dim 128 \
#     --n_way 5 \
#     --k_shot 1 \
#     --epochs 100 \
#     --inner_lr 0.01 \
#     --outer_lr 0.001 \
#     --inner_steps 5 \
#     --strong_augment

# 实验8: 完整正则化 (Dropout + 权重衰减 + 强增强)
echo ""
echo "[Exp 8] 5-way 1-shot MAML with Full Regularization (Conv1D4-128)"
python train_bearing.py \
    --model conv1d4 \
    --hidden_dim 128 \
    --n_way 5 \
    --k_shot 1 \
    --epochs 50 \
    --inner_lr 0.01 \
    --outer_lr 0.001 \
    --inner_steps 5 \
    --drop_rate 0.2 \
    --weight_decay 1e-4 \
    --strong_augment

echo ""
echo "=========================================="
echo "All bearing experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - checkpoints/bearing_maml_*"
echo "  - logs/bearing_maml_*"
