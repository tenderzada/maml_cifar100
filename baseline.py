"""
Baseline方法实现

用于与MAML进行对比:
1. Random Init + Finetune: 随机初始化后在support set上finetune
2. Transfer Learning: 在meta-train类上预训练，然后finetune
3. ProtoNet: 原型网络 (基于度量学习)
4. ProtoNet (Pretrained): 使用预训练特征的原型网络
"""

import os
import argparse
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from datetime import datetime

from data.cifar100_fewshot import get_dataloader, CIFAR100FewShot
from models.conv4 import Conv4, Conv4Functional
from utils.visualization import plot_comparison, plot_episode_accuracy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class Finetuner:
    """
    Finetune方法: 在support set上finetune模型
    """

    def __init__(self, model, device, finetune_lr=0.01, finetune_steps=100):
        self.model = model
        self.device = device
        self.finetune_lr = finetune_lr
        self.finetune_steps = finetune_steps

    def finetune_and_evaluate(self, support_x, support_y, query_x, query_y):
        """
        在support set上finetune，然后在query set上评估

        Returns:
            accuracy
        """
        # 重新初始化分类器
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            support_y.max().item() + 1
        ).to(self.device)

        # Finetune
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.finetune_lr)
        self.model.train()

        for _ in range(self.finetune_steps):
            logits = self.model(support_x)
            loss = F.cross_entropy(logits, support_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 评估
        self.model.eval()
        with torch.no_grad():
            logits = self.model(query_x)
            pred = logits.argmax(dim=1)
            acc = (pred == query_y).float().mean().item()

        return acc


class TransferLearning:
    """
    Transfer Learning方法:
    1. 在meta-train类上预训练
    2. 冻结特征提取器，只finetune分类器
    """

    def __init__(self, pretrained_model, device, finetune_lr=0.01,
                 finetune_steps=100, finetune_mode='head'):
        """
        Args:
            pretrained_model: 预训练模型
            device: 计算设备
            finetune_lr: finetune学习率
            finetune_steps: finetune步数
            finetune_mode: 'head' (只finetune分类头) 或 'full' (全部finetune)
        """
        self.base_model = pretrained_model
        self.device = device
        self.finetune_lr = finetune_lr
        self.finetune_steps = finetune_steps
        self.finetune_mode = finetune_mode

    def finetune_and_evaluate(self, support_x, support_y, query_x, query_y):
        """在support set上finetune，然后在query set上评估"""
        # 复制模型以避免修改原始模型
        model = copy.deepcopy(self.base_model)

        # 替换分类头
        n_way = support_y.max().item() + 1
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, n_way).to(self.device)

        if self.finetune_mode == 'head':
            # 冻结特征提取器
            for name, param in model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
            optimizer = torch.optim.SGD(model.fc.parameters(), lr=self.finetune_lr)
        else:
            # 全部finetune，但特征提取器用较小学习率
            optimizer = torch.optim.SGD([
                {'params': model.fc.parameters(), 'lr': self.finetune_lr},
                {'params': [p for n, p in model.named_parameters() if 'fc' not in n],
                 'lr': self.finetune_lr * 0.1}
            ])

        model.train()
        for _ in range(self.finetune_steps):
            logits = model(support_x)
            loss = F.cross_entropy(logits, support_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 评估
        model.eval()
        with torch.no_grad():
            logits = model(query_x)
            pred = logits.argmax(dim=1)
            acc = (pred == query_y).float().mean().item()

        return acc


class ProtoNet:
    """
    原型网络: 基于度量学习的few-shot方法
    """

    def __init__(self, encoder, device):
        """
        Args:
            encoder: 特征提取器
            device: 计算设备
        """
        self.encoder = encoder
        self.device = device

    def compute_prototypes(self, support_x, support_y, n_way):
        """
        计算每个类的原型 (类中心)
        """
        # 提取特征
        features = self.encoder.feature_extractor(support_x)

        prototypes = []
        for i in range(n_way):
            mask = (support_y == i)
            class_features = features[mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)

        return torch.stack(prototypes)  # [n_way, feature_dim]

    def evaluate(self, support_x, support_y, query_x, query_y, n_way):
        """
        使用原型网络进行评估
        """
        self.encoder.eval()
        with torch.no_grad():
            # 计算原型
            prototypes = self.compute_prototypes(support_x, support_y, n_way)

            # 提取query特征
            query_features = self.encoder.feature_extractor(query_x)

            # 计算到每个原型的距离
            # 使用负欧氏距离作为相似度
            dists = torch.cdist(query_features, prototypes)  # [n_query, n_way]
            logits = -dists

            pred = logits.argmax(dim=1)
            acc = (pred == query_y).float().mean().item()

        return acc


def evaluate_baseline(method, test_loader, device, n_way, hidden_dim=64,
                      pretrained_model=None):
    """
    评估baseline方法

    Args:
        method: 方法名称
        test_loader: 测试数据加载器
        device: 计算设备
        n_way: N-way分类
        hidden_dim: 模型隐藏层维度
        pretrained_model: 预训练模型 (用于transfer learning)
    """
    accuracies = []

    for support_x, support_y, query_x, query_y in test_loader:
        support_x = support_x.squeeze(0).to(device)
        support_y = support_y.squeeze(0).to(device)
        query_x = query_x.squeeze(0).to(device)
        query_y = query_y.squeeze(0).to(device)

        if method == 'random_finetune':
            # 随机初始化 + finetune
            model = Conv4(hidden_dim=hidden_dim, n_way=n_way).to(device)
            finetuner = Finetuner(model, device, finetune_lr=0.01, finetune_steps=50)
            acc = finetuner.finetune_and_evaluate(support_x, support_y, query_x, query_y)

        elif method == 'protonet':
            # 原型网络 (使用随机初始化的encoder)
            encoder = Conv4(hidden_dim=hidden_dim, n_way=n_way).to(device)
            protonet = ProtoNet(encoder, device)
            acc = protonet.evaluate(support_x, support_y, query_x, query_y, n_way)

        elif method == 'transfer_head':
            # Transfer Learning: 只finetune分类头
            if pretrained_model is None:
                raise ValueError("pretrained_model required for transfer learning")
            transfer = TransferLearning(pretrained_model, device,
                                       finetune_lr=0.01, finetune_steps=50,
                                       finetune_mode='head')
            acc = transfer.finetune_and_evaluate(support_x, support_y, query_x, query_y)

        elif method == 'transfer_full':
            # Transfer Learning: 全部finetune
            if pretrained_model is None:
                raise ValueError("pretrained_model required for transfer learning")
            transfer = TransferLearning(pretrained_model, device,
                                       finetune_lr=0.01, finetune_steps=50,
                                       finetune_mode='full')
            acc = transfer.finetune_and_evaluate(support_x, support_y, query_x, query_y)

        elif method == 'protonet_pretrained':
            # ProtoNet with pretrained encoder
            if pretrained_model is None:
                raise ValueError("pretrained_model required for protonet_pretrained")
            protonet = ProtoNet(pretrained_model, device)
            acc = protonet.evaluate(support_x, support_y, query_x, query_y, n_way)

        accuracies.append(acc)

    return accuracies


def compute_confidence_interval(accuracies, confidence=0.95):
    n = len(accuracies)
    mean = np.mean(accuracies)
    std = np.std(accuracies)
    se = std / np.sqrt(n)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, h


def main():
    parser = argparse.ArgumentParser(description='Baseline methods for CIFAR-100 Few-Shot')

    parser.add_argument('--data_root', type=str, default='/mnt/data/lev_data')
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_shot', type=int, default=1)
    parser.add_argument('--k_query', type=int, default=15)
    parser.add_argument('--test_episodes', type=int, default=600)
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='模型隐藏层维度')
    parser.add_argument('--method', type=str, default='all',
                        choices=['random_finetune', 'protonet', 'transfer_head',
                                'transfer_full', 'protonet_pretrained', 'all'])
    parser.add_argument('--pretrained_checkpoint', type=str, default=None,
                        help='预训练模型checkpoint (用于transfer learning)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='结果保存目录')
    parser.add_argument('--no_plot', action='store_true',
                        help='不生成可视化图表')

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载预训练模型 (如果提供)
    pretrained_model = None
    if args.pretrained_checkpoint:
        print(f"\nLoading pretrained model: {args.pretrained_checkpoint}")
        checkpoint = torch.load(args.pretrained_checkpoint, map_location=device)
        saved_args = checkpoint.get('args', {})
        hidden_dim = saved_args.get('hidden_dim', args.hidden_dim)

        pretrained_model = Conv4(hidden_dim=hidden_dim, n_way=64).to(device)
        pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        pretrained_model.eval()
        print(f"Pretrained model loaded (hidden_dim={hidden_dim})")

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

    print(f"\n{'='*60}")
    print(f"Baseline Evaluation: {args.n_way}-way {args.k_shot}-shot")
    print(f"Test episodes: {args.test_episodes}")
    print(f"{'='*60}\n")

    # 确定要评估的方法
    if args.method == 'all':
        methods = ['random_finetune', 'protonet']
        if pretrained_model is not None:
            methods.extend(['transfer_head', 'transfer_full', 'protonet_pretrained'])
    else:
        methods = [args.method]

    results = {}
    all_accuracies = {}
    for method in methods:
        # 检查是否需要预训练模型
        needs_pretrained = method in ['transfer_head', 'transfer_full', 'protonet_pretrained']
        if needs_pretrained and pretrained_model is None:
            print(f"\nSkipping {method}: requires --pretrained_checkpoint")
            continue

        print(f"\nEvaluating: {method}")

        # 重新创建数据加载器以确保每个方法使用相同的episodes
        test_loader = get_dataloader(
            root=args.data_root,
            mode='test',
            n_way=args.n_way,
            k_shot=args.k_shot,
            k_query=args.k_query,
            num_episodes=args.test_episodes,
            num_workers=4
        )

        accuracies = evaluate_baseline(
            method, test_loader, device, args.n_way,
            hidden_dim=args.hidden_dim, pretrained_model=pretrained_model
        )
        mean_acc, ci = compute_confidence_interval(accuracies)
        results[method] = (mean_acc, ci)
        all_accuracies[method] = accuracies
        print(f"  Accuracy: {mean_acc * 100:.2f}% ± {ci * 100:.2f}%")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for method, (mean_acc, ci) in results.items():
        print(f"{method:20s}: {mean_acc * 100:.2f}% ± {ci * 100:.2f}%")

    # 保存结果和可视化
    os.makedirs(args.save_dir, exist_ok=True)
    result_name = f"baseline_{args.n_way}way_{args.k_shot}shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 保存结果到文件
    result_path = os.path.join(args.save_dir, f"{result_name}.txt")
    with open(result_path, 'w') as f:
        f.write(f"Setting: {args.n_way}-way {args.k_shot}-shot\n")
        f.write(f"Test episodes: {args.test_episodes}\n\n")
        f.write("Results:\n")
        for method, (mean_acc, ci) in results.items():
            f.write(f"  {method}: {mean_acc * 100:.2f}% ± {ci * 100:.2f}%\n")
    print(f"\nResults saved to: {result_path}")

    # 生成可视化
    if not args.no_plot:
        # 方法对比柱状图
        comparison_path = os.path.join(args.save_dir, f"{result_name}_comparison.png")
        plot_comparison(results, save_path=comparison_path)
        print(f"Comparison plot saved to: {comparison_path}")

        # 每个方法的准确率分布
        for method, accuracies in all_accuracies.items():
            dist_path = os.path.join(args.save_dir, f"{result_name}_{method}_distribution.png")
            plot_episode_accuracy(
                accuracies,
                method_name=f"{method} ({args.n_way}-way {args.k_shot}-shot)",
                save_path=dist_path
            )
            print(f"{method} distribution saved to: {dist_path}")


if __name__ == '__main__':
    main()
