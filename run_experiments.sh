#!/bin/bash
# MAML实验脚本
# 在Ubuntu服务器 (RTX 4090) 上运行

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 创建输出目录
mkdir -p checkpoints logs results

echo "=========================================="
echo "MAML Experiments on CIFAR-100"
echo "=========================================="

# ============================================
# 第一阶段: 预训练 (用于Transfer Learning对比)
# ============================================
echo ""
echo "[Phase 1] Pretraining for Transfer Learning baseline"
echo "------------------------------------------"

# 预训练Conv4 (hidden_dim=128)
python pretrain.py \
    --model conv4 \
    --hidden_dim 128 \
    --epochs 100 \
    --batch_size 128 \
    --lr 0.1

# ============================================
# 第二阶段: MAML训练 (不同模型配置)
# ============================================
echo ""
echo "[Phase 2] MAML Training"
echo "------------------------------------------"

# 实验1: 5-way 1-shot MAML (Conv4, hidden_dim=64)
echo ""
echo "[Exp 1] 5-way 1-shot MAML (Conv4-64)"
python train.py \
    --model conv4 \
    --hidden_dim 64 \
    --n_way 5 \
    --k_shot 1 \
    --epochs 100 \
    --inner_lr 0.01 \
    --outer_lr 0.001 \
    --inner_steps 5

# 实验2: 5-way 1-shot MAML (Conv4, hidden_dim=128) - 更大模型
echo ""
echo "[Exp 2] 5-way 1-shot MAML (Conv4-128)"
python train.py \
    --model conv4 \
    --hidden_dim 128 \
    --n_way 5 \
    --k_shot 1 \
    --epochs 100 \
    --inner_lr 0.01 \
    --outer_lr 0.001 \
    --inner_steps 5

# 实验3: 5-way 5-shot MAML (Conv4, hidden_dim=128)
echo ""
echo "[Exp 3] 5-way 5-shot MAML (Conv4-128)"
python train.py \
    --model conv4 \
    --hidden_dim 128 \
    --n_way 5 \
    --k_shot 5 \
    --epochs 100 \
    --inner_lr 0.01 \
    --outer_lr 0.001 \
    --inner_steps 5

# 实验4: 5-way 1-shot FOMAML (Conv4, hidden_dim=128)
echo ""
echo "[Exp 4] 5-way 1-shot FOMAML (Conv4-128)"
python train.py \
    --model conv4 \
    --hidden_dim 128 \
    --n_way 5 \
    --k_shot 1 \
    --epochs 100 \
    --inner_lr 0.01 \
    --outer_lr 0.001 \
    --inner_steps 5 \
    --first_order

# ============================================
# 第三阶段: Baseline方法评估
# ============================================
echo ""
echo "[Phase 3] Baseline Evaluation"
echo "------------------------------------------"

# 找到最新的预训练模型
PRETRAIN_CKPT=$(ls -t checkpoints/pretrain_conv4_*_best.pth 2>/dev/null | head -1)

if [ -n "$PRETRAIN_CKPT" ]; then
    echo "Using pretrained checkpoint: $PRETRAIN_CKPT"

    # 5-way 1-shot baseline (包含Transfer Learning)
    echo ""
    echo "Evaluating 5-way 1-shot baselines..."
    python baseline.py \
        --n_way 5 \
        --k_shot 1 \
        --hidden_dim 128 \
        --test_episodes 600 \
        --method all \
        --pretrained_checkpoint "$PRETRAIN_CKPT" \
        --save_dir ./results

    # 5-way 5-shot baseline
    echo ""
    echo "Evaluating 5-way 5-shot baselines..."
    python baseline.py \
        --n_way 5 \
        --k_shot 5 \
        --hidden_dim 128 \
        --test_episodes 600 \
        --method all \
        --pretrained_checkpoint "$PRETRAIN_CKPT" \
        --save_dir ./results
else
    echo "No pretrained checkpoint found, running baselines without transfer learning..."
    python baseline.py --n_way 5 --k_shot 1 --test_episodes 600 --method all
    python baseline.py --n_way 5 --k_shot 5 --test_episodes 600 --method all
fi

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Generated outputs:"
echo "  - checkpoints/    : Model checkpoints"
echo "  - logs/           : Training logs and learning curves"
echo "  - results/        : Evaluation results and comparisons"
echo ""
echo "Key files:"
echo "  - logs/*_learning_curves.png      : Train/Val curves"
echo "  - logs/*_test_distribution.png    : Test accuracy distribution"
echo "  - results/*_comparison.png        : Method comparison"
