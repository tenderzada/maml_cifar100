"""
综合对比实验脚本

对比MAML与各种baseline方法的性能
生成完整的对比可视化
"""

import os
import argparse
import random
import numpy as np
import torch
from datetime import datetime
from scipy import stats

from data.cifar100_fewshot import get_dataloader
from models.conv4 import Conv4, Conv4Functional
from models.maml import MAML
from baseline import Finetuner, ProtoNet, evaluate_baseline
from utils.visualization import plot_comparison, plot_episode_accuracy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def compute_confidence_interval(accuracies, confidence=0.95):
    n = len(accuracies)
    mean = np.mean(accuracies)
    std = np.std(accuracies)
    se = std / np.sqrt(n)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, h


def evaluate_maml(maml, test_loader, device):
    """评估MAML模型"""
    maml.eval()
    accuracies = []

    for support_x, support_y, query_x, query_y in test_loader:
        support_x = support_x.squeeze(0).to(device)
        support_y = support_y.squeeze(0).to(device)
        query_x = query_x.squeeze(0).to(device)
        query_y = query_y.squeeze(0).to(device)

        with torch.enable_grad():
            adapted_vars = maml.adapt(support_x, support_y)
        _, acc = maml.evaluate(query_x, query_y, adapted_vars)
        accuracies.append(acc)

    return accuracies


def main():
    parser = argparse.ArgumentParser(description='Compare MAML with Baselines')

    parser.add_argument('--maml_checkpoint', type=str, required=True,
                        help='MAML模型checkpoint路径')
    parser.add_argument('--data_root', type=str, default='/mnt/data/lev_data')
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_shot', type=int, default=1)
    parser.add_argument('--k_query', type=int, default=15)
    parser.add_argument('--test_episodes', type=int, default=600)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./results')

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    result_name = f"comparison_{args.n_way}way_{args.k_shot}shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'='*60}")
    print(f"Method Comparison: {args.n_way}-way {args.k_shot}-shot")
    print(f"Test episodes: {args.test_episodes}")
    print(f"{'='*60}\n")

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

    results = {}
    all_accuracies = {}

    # 1. 评估MAML
    print("Evaluating MAML...")
    checkpoint = torch.load(args.maml_checkpoint, map_location=device)
    saved_args = checkpoint.get('args', {})

    model = Conv4Functional(
        in_channels=3,
        hidden_dim=saved_args.get('hidden_dim', 64),
        n_way=args.n_way
    )
    maml = MAML(
        model=model,
        inner_lr=saved_args.get('inner_lr', 0.01),
        inner_steps=saved_args.get('inner_steps', 5),
        first_order=saved_args.get('first_order', False),
        device=device
    )
    maml.model.load_state_dict(checkpoint['model_state_dict'])

    maml_accs = evaluate_maml(maml, test_loader, device)
    mean_acc, ci = compute_confidence_interval(maml_accs)
    method_name = "FOMAML" if saved_args.get('first_order', False) else "MAML"
    results[method_name] = (mean_acc, ci)
    all_accuracies[method_name] = maml_accs
    print(f"  {method_name}: {mean_acc * 100:.2f}% ± {ci * 100:.2f}%")

    # 2. 评估Baseline方法
    # 重新创建数据加载器以确保公平比较
    test_loader = get_dataloader(
        root=args.data_root,
        mode='test',
        n_way=args.n_way,
        k_shot=args.k_shot,
        k_query=args.k_query,
        num_episodes=args.test_episodes,
        num_workers=4
    )

    print("\nEvaluating Random Init + Finetune...")
    finetune_accs = evaluate_baseline('random_finetune', test_loader, device, args.n_way)
    mean_acc, ci = compute_confidence_interval(finetune_accs)
    results['Random+Finetune'] = (mean_acc, ci)
    all_accuracies['Random+Finetune'] = finetune_accs
    print(f"  Random+Finetune: {mean_acc * 100:.2f}% ± {ci * 100:.2f}%")

    # 重新创建数据加载器
    test_loader = get_dataloader(
        root=args.data_root,
        mode='test',
        n_way=args.n_way,
        k_shot=args.k_shot,
        k_query=args.k_query,
        num_episodes=args.test_episodes,
        num_workers=4
    )

    print("\nEvaluating ProtoNet (Random Init)...")
    proto_accs = evaluate_baseline('protonet', test_loader, device, args.n_way)
    mean_acc, ci = compute_confidence_interval(proto_accs)
    results['ProtoNet'] = (mean_acc, ci)
    all_accuracies['ProtoNet'] = proto_accs
    print(f"  ProtoNet: {mean_acc * 100:.2f}% ± {ci * 100:.2f}%")

    # 打印汇总结果
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for method, (mean_acc, ci) in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
        print(f"{method:20s}: {mean_acc * 100:.2f}% ± {ci * 100:.2f}%")

    # 保存结果
    result_path = os.path.join(args.save_dir, f"{result_name}.txt")
    with open(result_path, 'w') as f:
        f.write(f"Method Comparison: {args.n_way}-way {args.k_shot}-shot\n")
        f.write(f"Test episodes: {args.test_episodes}\n")
        f.write(f"MAML checkpoint: {args.maml_checkpoint}\n\n")
        f.write("Results (sorted by accuracy):\n")
        for method, (mean_acc, ci) in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
            f.write(f"  {method}: {mean_acc * 100:.2f}% ± {ci * 100:.2f}%\n")
    print(f"\nResults saved to: {result_path}")

    # 生成可视化
    # 1. 方法对比柱状图
    comparison_path = os.path.join(args.save_dir, f"{result_name}_comparison.png")
    plot_comparison(results, save_path=comparison_path)
    print(f"Comparison plot saved to: {comparison_path}")

    # 2. 各方法准确率分布
    for method, accuracies in all_accuracies.items():
        safe_name = method.replace('+', '_').replace(' ', '_')
        dist_path = os.path.join(args.save_dir, f"{result_name}_{safe_name}_dist.png")
        plot_episode_accuracy(
            accuracies,
            method_name=f"{method} ({args.n_way}-way {args.k_shot}-shot)",
            save_path=dist_path
        )

    print(f"\nAll visualizations saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
