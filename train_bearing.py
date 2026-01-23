"""
MAML训练脚本 - 轴承故障诊断数据集

使用轴承时序数据进行few-shot learning训练
数据形状: [9, 2048] (9通道，2048时间步)
类别数: 64 (40 train, 12 val, 12 test)
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from datetime import datetime
import json

from data.bearing_fewshot import get_bearing_fewshot_loader
from models.conv1d import Conv1D4Functional, Conv1D6Functional
from models.maml import MAML, MAMLTrainer
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
    parser = argparse.ArgumentParser(description='MAML for Bearing Fault Diagnosis')

    # 数据参数
    parser.add_argument('--data_file', type=str, default='/mnt/data/lev_data/bearing_data.pkl',
                        help='轴承数据文件路径')
    parser.add_argument('--n_way', type=int, default=5,
                        help='N-way classification')
    parser.add_argument('--k_shot', type=int, default=1,
                        help='K-shot (support set size per class)')
    parser.add_argument('--k_query', type=int, default=15,
                        help='Query set size per class')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练epoch数')
    parser.add_argument('--train_episodes', type=int, default=600,
                        help='每个epoch的训练episode数')
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
                        help='CNN隐藏层维度 (32/64/128)')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存路径')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志保存路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的checkpoint路径')

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
    exp_name = f"bearing_maml_{args.n_way}way_{args.k_shot}shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\nExperiment: {exp_name}")
    print("=" * 60)

    # 保存配置
    config_path = os.path.join(args.log_dir, f"{exp_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 创建数据加载器
    print("\nLoading bearing data...")
    train_loader = get_bearing_fewshot_loader(
        data_file=args.data_file,
        mode='train',
        n_way=args.n_way,
        k_shot=args.k_shot,
        k_query=args.k_query,
        num_episodes=args.train_episodes,
        num_workers=args.num_workers
    )

    val_loader = get_bearing_fewshot_loader(
        data_file=args.data_file,
        mode='val',
        n_way=args.n_way,
        k_shot=args.k_shot,
        k_query=args.k_query,
        num_episodes=args.val_episodes,
        num_workers=args.num_workers
    )

    test_loader = get_bearing_fewshot_loader(
        data_file=args.data_file,
        mode='test',
        n_way=args.n_way,
        k_shot=args.k_shot,
        k_query=args.k_query,
        num_episodes=args.test_episodes,
        num_workers=args.num_workers
    )

    print(f"Train episodes: {args.train_episodes}")
    print(f"Val episodes: {args.val_episodes}")
    print(f"Test episodes: {args.test_episodes}")

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

    maml = MAML(
        model=model,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        first_order=args.first_order,
        device=device
    )

    num_params = sum(p.numel() for p in maml.model.vars)
    print(f"Model: {model_desc}")
    print(f"Model parameters: {num_params:,}")
    print(f"Inner LR: {args.inner_lr}, Inner Steps: {args.inner_steps}")
    print(f"First Order (FOMAML): {args.first_order}")

    # 创建训练器
    trainer = MAMLTrainer(maml, outer_lr=args.outer_lr, device=device)

    # 恢复训练
    start_epoch = 0
    best_val_acc = 0
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume)
        maml.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']

    # 日志
    log_path = os.path.join(args.log_dir, f"{exp_name}_log.txt")
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"\n{'='*60}")
    log(f"Training {args.n_way}-way {args.k_shot}-shot on Bearing Dataset")
    log(f"{'='*60}\n")

    # 记录训练历史
    history = {
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': [],
        'learning_rates': []
    }

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        log(f"\nEpoch {epoch + 1}/{args.epochs}")
        log("-" * 40)

        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        log(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        val_loss, val_acc = trainer.validate(val_loader)
        log(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        trainer.step_scheduler()
        current_lr = trainer.optimizer.param_groups[0]['lr']
        log(f"Current LR: {current_lr:.6f}")

        # 记录历史
        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['val_losses'].append(val_loss)
        history['val_accs'].append(val_acc)
        history['learning_rates'].append(current_lr)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.save_dir, f"{exp_name}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': maml.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'args': vars(args)
            }, best_path)
            log(f"New best model saved! Val Acc: {best_val_acc:.4f}")

        # 定期保存
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(args.save_dir, f"{exp_name}_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': maml.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'args': vars(args)
            }, ckpt_path)

    # 最终测试
    log(f"\n{'='*60}")
    log("Final Testing")
    log(f"{'='*60}")

    best_path = os.path.join(args.save_dir, f"{exp_name}_best.pth")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path)
        maml.model.load_state_dict(checkpoint['model_state_dict'])
        log(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")

    test_loss, test_acc = trainer.validate(test_loader)
    log(f"\nTest Results:")
    log(f"  Loss: {test_loss:.4f}")
    log(f"  Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    log(f"\nBest Val Accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")

    # 收集测试准确率分布
    log("\nCollecting test episode accuracies...")
    test_accuracies = []
    maml.eval()
    for support_x, support_y, query_x, query_y in test_loader:
        support_x = support_x.squeeze(0).to(device)
        support_y = support_y.squeeze(0).to(device)
        query_x = query_x.squeeze(0).to(device)
        query_y = query_y.squeeze(0).to(device)

        with torch.enable_grad():
            adapted_vars = maml.adapt(support_x, support_y)
        _, acc = maml.evaluate(query_x, query_y, adapted_vars)
        test_accuracies.append(acc)

    # 保存历史
    history['test_loss'] = test_loss
    history['test_acc'] = test_acc
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
    method_name = "FOMAML" if args.first_order else "MAML"
    plot_episode_accuracy(
        test_accuracies,
        method_name=f"{method_name} Bearing ({args.n_way}-way {args.k_shot}-shot)",
        save_path=test_dist_path
    )

    log_file.close()
    print(f"\nTraining complete! Logs saved to: {log_path}")


if __name__ == '__main__':
    main()
