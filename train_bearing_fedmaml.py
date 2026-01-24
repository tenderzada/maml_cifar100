"""
FedAvg + MAML 训练脚本 - 轴承故障诊断数据集

联邦元学习 (Federated Meta-Learning):
- 结合FedAvg的通信机制和MAML的元学习策略
- 每个客户端执行MAML内层更新，服务器聚合元参数
- 适用于客户端数据异构的few-shot学习场景

支持:
- IID数据分布
- Non-IID数据分布
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from datetime import datetime
import json

from data.bearing_federated import FederatedBearingData
from models.conv1d import Conv1D4Functional, Conv1D6Functional
from models.federated import FedMAML
from utils.visualization import plot_learning_curves, plot_episode_accuracy


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='FedAvg+MAML for Bearing Fault Diagnosis')

    # 数据参数
    parser.add_argument('--data_file', type=str, default='/mnt/data/scutfd/bearing_data.pkl',
                        help='轴承数据文件路径')
    parser.add_argument('--n_way', type=int, default=5,
                        help='N-way classification')
    parser.add_argument('--k_shot', type=int, default=1,
                        help='K-shot (support set size per class)')
    parser.add_argument('--k_query', type=int, default=15,
                        help='Query set size per class')

    # 联邦学习参数
    parser.add_argument('--num_clients', type=int, default=10,
                        help='客户端总数')
    parser.add_argument('--clients_per_round', type=int, default=2,
                        help='每轮选择的客户端数量')
    parser.add_argument('--iid', action='store_true',
                        help='使用IID数据分布 (默认Non-IID)')
    parser.add_argument('--non_iid_classes', type=int, default=6,
                        help='Non-IID时每个客户端的类别数')

    # 训练参数
    parser.add_argument('--rounds', type=int, default=100,
                        help='通信轮数')
    parser.add_argument('--local_meta_steps', type=int, default=10,
                        help='每个客户端的元更新步数 (episodes)')
    parser.add_argument('--val_episodes', type=int, default=200,
                        help='验证episode数')
    parser.add_argument('--test_episodes', type=int, default=600,
                        help='测试episode数')

    # MAML参数
    parser.add_argument('--inner_lr', type=float, default=0.01,
                        help='内层学习率 (task adaptation)')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                        help='外层学习率 (meta-learning)')
    parser.add_argument('--inner_steps', type=int, default=5,
                        help='内层梯度更新步数')
    parser.add_argument('--first_order', action='store_true',
                        help='使用一阶近似 (FOMAML)')

    # 模型参数
    parser.add_argument('--model', type=str, default='conv1d4',
                        choices=['conv1d4', 'conv1d6'],
                        help='模型架构')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='CNN隐藏层维度')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存路径')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志保存路径')

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 实验名称
    dist_type = "iid" if args.iid else "noniid"
    exp_name = f"bearing_fedmaml_{dist_type}_{args.n_way}way_{args.k_shot}shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\nExperiment: {exp_name}")
    print("=" * 60)

    # 保存配置
    config_path = os.path.join(args.log_dir, f"{exp_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 创建联邦数据
    print("\nLoading federated bearing data...")
    fed_data = FederatedBearingData(
        data_file=args.data_file,
        num_clients=args.num_clients,
        iid=args.iid,
        non_iid_classes_per_client=args.non_iid_classes,
        seed=args.seed
    )

    # 创建各客户端的Few-Shot数据加载器
    client_loaders = []
    for i in range(args.num_clients):
        loader = fed_data.get_client_fewshot_loader(
            client_id=i,
            n_way=args.n_way,
            k_shot=args.k_shot,
            k_query=args.k_query,
            num_episodes=args.local_meta_steps * 2,  # 足够的episodes
            num_workers=args.num_workers
        )
        client_loaders.append(loader)

    # 创建全局验证和测试集
    val_loader = fed_data.get_global_fewshot_loader(
        mode='val',
        n_way=args.n_way,
        k_shot=args.k_shot,
        k_query=args.k_query,
        num_episodes=args.val_episodes,
        num_workers=args.num_workers
    )

    test_loader = fed_data.get_global_fewshot_loader(
        mode='test',
        n_way=args.n_way,
        k_shot=args.k_shot,
        k_query=args.k_query,
        num_episodes=args.test_episodes,
        num_workers=args.num_workers
    )

    # 创建模型
    print("\nCreating model...")
    if args.model == 'conv1d4':
        model = Conv1D4Functional(
            in_channels=9,
            hidden_dim=args.hidden_dim,
            n_way=args.n_way
        )
        model_desc = f"Conv1D4 (hidden_dim={args.hidden_dim})"
    else:  # conv1d6
        model = Conv1D6Functional(
            in_channels=9,
            hidden_dim=args.hidden_dim,
            n_way=args.n_way
        )
        model_desc = f"Conv1D6 (hidden_dim={args.hidden_dim})"

    # 创建FedMAML
    fedmaml = FedMAML(
        model=model,
        num_clients=args.num_clients,
        clients_per_round=args.clients_per_round,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        outer_lr=args.outer_lr,
        local_meta_steps=args.local_meta_steps,
        first_order=args.first_order,
        device=device
    )

    num_params = sum(p.numel() for p in fedmaml.model.vars)
    print(f"Model: {model_desc}")
    print(f"Model parameters: {num_params:,}")
    print(f"Clients: {args.num_clients}, Per round: {args.clients_per_round}")
    print(f"Local meta steps: {args.local_meta_steps}")
    print(f"Inner LR: {args.inner_lr}, Inner Steps: {args.inner_steps}")
    print(f"Outer LR: {args.outer_lr}")
    print(f"First Order: {args.first_order}")
    print(f"Data distribution: {'IID' if args.iid else 'Non-IID'}")

    # 日志
    log_path = os.path.join(args.log_dir, f"{exp_name}_log.txt")
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"\n{'='*60}")
    log(f"Training FedAvg+MAML {args.n_way}-way {args.k_shot}-shot on Bearing Dataset")
    log(f"{'='*60}\n")

    # 记录训练历史
    history = {
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': [],
    }

    best_val_acc = 0

    # 训练循环
    for round_idx in range(args.rounds):
        log(f"\nRound {round_idx + 1}/{args.rounds}")
        log("-" * 40)

        # 执行一轮联邦元学习
        round_stats = fedmaml.train_round(client_loaders)

        log(f"Selected clients: {round_stats['selected_clients']}")
        log(f"Train - Loss: {round_stats['loss']:.4f}, Acc: {round_stats['accuracy']:.4f}")

        # 在验证集上评估
        val_stats = fedmaml.evaluate(val_loader)
        log(f"Val   - Loss: {val_stats['loss']:.4f}, Acc: {val_stats['accuracy']:.4f}")

        # 记录历史
        history['train_losses'].append(round_stats['loss'])
        history['train_accs'].append(round_stats['accuracy'])
        history['val_losses'].append(val_stats['loss'])
        history['val_accs'].append(val_stats['accuracy'])

        # 保存最佳模型
        if val_stats['accuracy'] > best_val_acc:
            best_val_acc = val_stats['accuracy']
            best_path = os.path.join(args.save_dir, f"{exp_name}_best.pth")
            torch.save({
                'round': round_idx,
                'model_vars': [p.data.clone() for p in fedmaml.model.vars],
                'best_val_acc': best_val_acc,
                'args': vars(args)
            }, best_path)
            log(f"New best model saved! Val Acc: {best_val_acc:.4f}")

        # 定期保存
        if (round_idx + 1) % 20 == 0:
            ckpt_path = os.path.join(args.save_dir, f"{exp_name}_round{round_idx+1}.pth")
            torch.save({
                'round': round_idx,
                'model_vars': [p.data.clone() for p in fedmaml.model.vars],
                'best_val_acc': best_val_acc,
                'args': vars(args)
            }, ckpt_path)

    # 最终测试
    log(f"\n{'='*60}")
    log("Final Testing")
    log(f"{'='*60}")

    # 加载最佳模型
    best_path = os.path.join(args.save_dir, f"{exp_name}_best.pth")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path)
        for i, param in enumerate(fedmaml.model.vars):
            param.data = checkpoint['model_vars'][i]
        log(f"Loaded best model from round {checkpoint['round'] + 1}")

    test_stats = fedmaml.evaluate(test_loader)
    log(f"\nTest Results:")
    log(f"  Loss: {test_stats['loss']:.4f}")
    log(f"  Accuracy: {test_stats['accuracy']:.4f} ({test_stats['accuracy'] * 100:.2f}%)")
    log(f"\nBest Val Accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")

    # 收集测试准确率分布
    log("\nCollecting test episode accuracies...")
    test_accuracies = []
    for support_x, support_y, query_x, query_y in test_loader:
        support_x = support_x.squeeze(0)
        support_y = support_y.squeeze(0)
        query_x = query_x.squeeze(0)
        query_y = query_y.squeeze(0)

        _, acc = fedmaml.adapt_and_evaluate(
            support_x, support_y, query_x, query_y
        )
        test_accuracies.append(acc)

    # 保存历史
    history['test_loss'] = test_stats['loss']
    history['test_acc'] = test_stats['accuracy']
    history['test_accuracies'] = test_accuracies
    history['best_val_acc'] = best_val_acc

    history_path = os.path.join(args.log_dir, f"{exp_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)

    # 生成可视化
    log("\nGenerating visualization...")

    curves_path = os.path.join(args.log_dir, f"{exp_name}_learning_curves.png")
    plot_learning_curves(
        history['train_losses'],
        history['train_accs'],
        history['val_losses'],
        history['val_accs'],
        save_path=curves_path
    )

    test_dist_path = os.path.join(args.log_dir, f"{exp_name}_test_distribution.png")
    method_name = f"FedMAML {'IID' if args.iid else 'Non-IID'} ({args.n_way}-way {args.k_shot}-shot)"
    plot_episode_accuracy(
        test_accuracies,
        method_name=method_name,
        save_path=test_dist_path
    )

    log_file.close()
    print(f"\nTraining complete! Logs saved to: {log_path}")


if __name__ == '__main__':
    main()
