#!/bin/bash
# CIFAR-100 联邦学习三方法对比 (FedAvg / FedAvg+MAML / FedAvg+Meta-SGD)
# 在 GPU 服务器 (RTX 4090) 上运行
# 产出: results/cifar_fed_compare_*_loss.png 与 *_acc.png

export CUDA_VISIBLE_DEVICES=0
mkdir -p results

echo "=========================================="
echo "CIFAR-100 Federated Three-Method Comparison"
echo "=========================================="

python train_cifar_fed_compare.py \
    --data_root /mnt/data/lev_data \
    --num_classes 20 \
    --num_clients 10 \
    --clients_per_round 5 \
    --non_iid \
    --dirichlet_alpha 0.3 \
    --rounds 150 \
    --eval_every 2 \
    --batch_size 32 \
    --lr_final_ratio 0.02 \
    --local_epochs 3 \
    --local_lr 0.05 \
    --local_meta_steps 10 \
    --inner_lr 0.01 \
    --inner_steps 5 \
    --outer_lr 0.001 \
    --alpha_lr 0.005 \
    --k_support 5 \
    --k_query 5 \
    --k_shot_eval 5 \
    --query_per_class 30 \
    --n_eval_episodes 5 \
    --channels 64 128 256 512 \
    --drop_rate 0.1 \
    --weight_decay 5e-4 \
    --first_order \
    --seed 42

echo ""
echo "Done. Figures and history saved to ./results/"
