"""
FedAvg训练脚本 - 轴承故障诊断数据集

FedAvg (Federated Averaging):
McMahan et al., "Communication-Efficient Learning of Deep Networks
from Decentralized Data", AISTATS 2017

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
from models.conv1d import Conv1D4, Conv1D6
from models.federated import FedAvg
from utils.visualization import plot_learning_curves


def get_device_info():
    """获取GPU设备信息"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        return num_gpus, gpu_names
    return 0, []


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
    parser = argparse.ArgumentParser(description='FedAvg for Bearing Fault Diagnosis')

    # 数据参数
    parser.add_argument('--data_file', type=str, default='/mnt/data/scutfd/bearing_data.pkl',
                        help='轴承数据文件路径')

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
    parser.add_argument('--local_epochs', type=int, default=5,
                        help='本地训练轮数')
    parser.add_argument('--local_lr', type=float, default=0.01,
                        help='本地学习率')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')

    # 模型参数
    parser.add_argument('--model', type=str, default='conv1d4',
                        choices=['conv1d4', 'conv1d6'],
                        help='模型架构')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='CNN隐藏层维度')

    # 正则化参数
    parser.add_argument('--drop_rate', type=float, default=0.0,
                        help='Dropout率 (推荐0.1-0.3)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='权重衰减 (推荐1e-4)')
    parser.add_argument('--strong_augment', action='store_true',
                        help='使用强数据增强')

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
    exp_name = f"bearing_fedavg_{dist_type}_{args.num_clients}clients_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    # 创建各客户端的数据加载器 (带数据增强)
    client_loaders = []
    for i in range(args.num_clients):
        loader = fed_data.get_client_dataloader(
            client_id=i,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            strong_augment=args.strong_augment
        )
        client_loaders.append(loader)

    # 创建全局测试集
    test_loader = fed_data.get_global_test_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 创建模型
    print("\nCreating model...")
    n_classes = len(fed_data.train_classes)  # 训练类别数

    if args.model == 'conv1d4':
        model = Conv1D4(
            in_channels=9,
            hidden_dim=args.hidden_dim,
            n_way=n_classes,
            drop_rate=args.drop_rate
        )
        model_desc = f"Conv1D4 (hidden_dim={args.hidden_dim}, drop_rate={args.drop_rate})"
    else:  # conv1d6
        model = Conv1D6(
            in_channels=9,
            hidden_dim=args.hidden_dim,
            n_way=n_classes,
            drop_rate=args.drop_rate
        )
        model_desc = f"Conv1D6 (hidden_dim={args.hidden_dim}, drop_rate={args.drop_rate})"

    # 创建FedAvg (带权重衰减)
    fedavg = FedAvg(
        global_model=model,
        num_clients=args.num_clients,
        clients_per_round=args.clients_per_round,
        local_epochs=args.local_epochs,
        local_lr=args.local_lr,
        weight_decay=args.weight_decay,
        device=device
    )

    num_params = sum(p.numel() for p in fedavg.global_model.parameters())
    print(f"Model: {model_desc}")
    print(f"Model parameters: {num_params:,}")
    print(f"Clients: {args.num_clients}, Per round: {args.clients_per_round}")
    print(f"Local epochs: {args.local_epochs}, Local LR: {args.local_lr}")
    print(f"Data distribution: {'IID' if args.iid else 'Non-IID'}")
    print(f"Regularization: dropout={args.drop_rate}, weight_decay={args.weight_decay}")
    print(f"Strong augmentation: {args.strong_augment}")

    # 日志
    log_path = os.path.join(args.log_dir, f"{exp_name}_log.txt")
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"\n{'='*60}")
    log(f"Training FedAvg on Bearing Dataset")
    log(f"{'='*60}\n")

    # 记录训练历史
    history = {
        'train_losses': [],
        'train_accs': [],
        'test_losses': [],
        'test_accs': [],
    }

    best_test_acc = 0

    # 训练循环
    for round_idx in range(args.rounds):
        log(f"\nRound {round_idx + 1}/{args.rounds}")
        log("-" * 40)

        # 执行一轮联邦训练
        round_stats = fedavg.train_round(client_loaders)

        log(f"Selected clients: {round_stats['selected_clients']}")
        log(f"Train - Loss: {round_stats['loss']:.4f}, Acc: {round_stats['accuracy']:.4f}")

        # 在测试集上评估
        test_stats = fedavg.evaluate(test_loader)
        log(f"Test  - Loss: {test_stats['loss']:.4f}, Acc: {test_stats['accuracy']:.4f}")

        # 记录历史
        history['train_losses'].append(round_stats['loss'])
        history['train_accs'].append(round_stats['accuracy'])
        history['test_losses'].append(test_stats['loss'])
        history['test_accs'].append(test_stats['accuracy'])

        # 保存最佳模型
        if test_stats['accuracy'] > best_test_acc:
            best_test_acc = test_stats['accuracy']
            best_path = os.path.join(args.save_dir, f"{exp_name}_best.pth")
            torch.save({
                'round': round_idx,
                'model_state_dict': fedavg.global_model.state_dict(),
                'best_test_acc': best_test_acc,
                'args': vars(args)
            }, best_path)
            log(f"New best model saved! Test Acc: {best_test_acc:.4f}")

        # 定期保存
        if (round_idx + 1) % 20 == 0:
            ckpt_path = os.path.join(args.save_dir, f"{exp_name}_round{round_idx+1}.pth")
            torch.save({
                'round': round_idx,
                'model_state_dict': fedavg.global_model.state_dict(),
                'best_test_acc': best_test_acc,
                'args': vars(args)
            }, ckpt_path)

    # 最终结果
    log(f"\n{'='*60}")
    log("Training Complete")
    log(f"{'='*60}")
    log(f"\nBest Test Accuracy: {best_test_acc:.4f} ({best_test_acc * 100:.2f}%)")

    # 保存历史
    history['best_test_acc'] = best_test_acc

    history_path = os.path.join(args.log_dir, f"{exp_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)

    # 生成可视化
    log("\nGenerating visualization...")

    curves_path = os.path.join(args.log_dir, f"{exp_name}_learning_curves.png")
    plot_learning_curves(
        history['train_losses'],
        history['train_accs'],
        history['test_losses'],
        history['test_accs'],
        save_path=curves_path
    )

    log_file.close()
    print(f"\nTraining complete! Logs saved to: {log_path}")


if __name__ == '__main__':
    main()
