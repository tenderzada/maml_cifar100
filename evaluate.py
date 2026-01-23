"""
评估脚本

在测试集上评估训练好的MAML模型
支持多次运行取平均以获得更准确的结果
"""

import os
import argparse
import random
import numpy as np
import torch
from scipy import stats
from datetime import datetime

from data.cifar100_fewshot import get_dataloader
from models.conv4 import Conv4Functional
from models.maml import MAML
from utils.visualization import plot_episode_accuracy


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate_maml(maml, test_loader, device):
    """
    评估MAML模型

    Returns:
        accuracies: 每个episode的准确率列表
    """
    maml.eval()
    accuracies = []

    for support_x, support_y, query_x, query_y in test_loader:
        support_x = support_x.squeeze(0).to(device)
        support_y = support_y.squeeze(0).to(device)
        query_x = query_x.squeeze(0).to(device)
        query_y = query_y.squeeze(0).to(device)

        # 内层循环适应
        with torch.enable_grad():
            adapted_vars = maml.adapt(support_x, support_y)

        # 评估
        _, acc = maml.evaluate(query_x, query_y, adapted_vars)
        accuracies.append(acc)

    return accuracies


def compute_confidence_interval(accuracies, confidence=0.95):
    """
    计算置信区间

    Args:
        accuracies: 准确率列表
        confidence: 置信水平

    Returns:
        mean, confidence_interval
    """
    n = len(accuracies)
    mean = np.mean(accuracies)
    std = np.std(accuracies)
    se = std / np.sqrt(n)

    # t分布的置信区间
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)

    return mean, h


def main():
    parser = argparse.ArgumentParser(description='Evaluate MAML on CIFAR-100')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--data_root', type=str, default='/mnt/data/lev_data',
                        help='CIFAR-100数据路径')
    parser.add_argument('--n_way', type=int, default=5,
                        help='N-way classification')
    parser.add_argument('--k_shot', type=int, default=1,
                        help='K-shot')
    parser.add_argument('--k_query', type=int, default=15,
                        help='Query set size per class')
    parser.add_argument('--test_episodes', type=int, default=1000,
                        help='测试episode数')
    parser.add_argument('--num_runs', type=int, default=5,
                        help='运行次数 (取平均)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='结果保存目录')
    parser.add_argument('--no_plot', action='store_true',
                        help='不生成可视化图表')

    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 从checkpoint中获取配置
    saved_args = checkpoint.get('args', {})
    inner_lr = saved_args.get('inner_lr', 0.01)
    inner_steps = saved_args.get('inner_steps', 5)
    hidden_dim = saved_args.get('hidden_dim', 64)
    first_order = saved_args.get('first_order', False)

    print(f"Model config - Inner LR: {inner_lr}, Steps: {inner_steps}, "
          f"Hidden: {hidden_dim}, First Order: {first_order}")

    # 创建模型
    model = Conv4Functional(
        in_channels=3,
        hidden_dim=hidden_dim,
        n_way=args.n_way
    )

    maml = MAML(
        model=model,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        first_order=first_order,
        device=device
    )

    # 加载权重
    maml.model.load_state_dict(checkpoint['model_state_dict'])

    print(f"\n{'='*60}")
    print(f"Evaluating {args.n_way}-way {args.k_shot}-shot")
    print(f"Test episodes: {args.test_episodes}")
    print(f"Number of runs: {args.num_runs}")
    print(f"{'='*60}\n")

    all_accuracies = []

    for run in range(args.num_runs):
        # 设置不同的随机种子
        set_seed(args.seed + run)

        # 创建测试数据加载器
        test_loader = get_dataloader(
            root=args.data_root,
            mode='test',
            n_way=args.n_way,
            k_shot=args.k_shot,
            k_query=args.k_query,
            num_episodes=args.test_episodes,
            num_workers=4
        )

        # 评估
        accuracies = evaluate_maml(maml, test_loader, device)
        mean_acc = np.mean(accuracies)
        all_accuracies.extend(accuracies)

        print(f"Run {run + 1}/{args.num_runs}: Accuracy = {mean_acc * 100:.2f}%")

    # 计算总体结果
    mean_acc, ci = compute_confidence_interval(all_accuracies)

    print(f"\n{'='*60}")
    print("Final Results")
    print(f"{'='*60}")
    print(f"Accuracy: {mean_acc * 100:.2f}% ± {ci * 100:.2f}%")
    print(f"(95% confidence interval over {len(all_accuracies)} episodes)")

    # 保存结果和可视化
    os.makedirs(args.save_dir, exist_ok=True)

    # 保存结果到文件
    result_name = f"eval_{args.n_way}way_{args.k_shot}shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result_path = os.path.join(args.save_dir, f"{result_name}.txt")
    with open(result_path, 'w') as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Setting: {args.n_way}-way {args.k_shot}-shot\n")
        f.write(f"Test episodes per run: {args.test_episodes}\n")
        f.write(f"Number of runs: {args.num_runs}\n")
        f.write(f"Total episodes: {len(all_accuracies)}\n")
        f.write(f"\nAccuracy: {mean_acc * 100:.2f}% ± {ci * 100:.2f}%\n")
    print(f"\nResults saved to: {result_path}")

    # 生成可视化
    if not args.no_plot:
        method_name = "FOMAML" if first_order else "MAML"
        plot_path = os.path.join(args.save_dir, f"{result_name}_distribution.png")
        plot_episode_accuracy(
            all_accuracies,
            method_name=f"{method_name} ({args.n_way}-way {args.k_shot}-shot)",
            save_path=plot_path
        )
        print(f"Accuracy distribution plot saved to: {plot_path}")


if __name__ == '__main__':
    main()
