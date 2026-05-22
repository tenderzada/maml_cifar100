"""
可视化工具
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(train_losses, train_accs, val_losses, val_accs,
                         save_path=None):
    """
    绘制训练曲线

    Args:
        train_losses: 训练loss列表
        train_accs: 训练准确率列表
        val_losses: 验证loss列表
        val_accs: 验证准确率列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss曲线
    axes[0].plot(train_losses, label='Train', color='blue')
    axes[0].plot(val_losses, label='Val', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy曲线
    axes[1].plot([acc * 100 for acc in train_accs], label='Train', color='blue')
    axes[1].plot([acc * 100 for acc in val_accs], label='Val', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Learning curves saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_episode_accuracy(accuracies, method_name='MAML', save_path=None):
    """
    绘制episode准确率分布

    Args:
        accuracies: 准确率列表
        method_name: 方法名称
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 直方图
    axes[0].hist(accuracies, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(accuracies), color='red', linestyle='--',
                   label=f'Mean: {np.mean(accuracies)*100:.2f}%')
    axes[0].set_xlabel('Accuracy')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'{method_name} Episode Accuracy Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 箱线图
    axes[1].boxplot([acc * 100 for acc in accuracies])
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{method_name} Accuracy Box Plot')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Episode accuracy plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_comparison(results_dict, save_path=None):
    """
    绘制不同方法的对比图

    Args:
        results_dict: {method_name: (mean_acc, ci)}
        save_path: 保存路径
    """
    methods = list(results_dict.keys())
    means = [results_dict[m][0] * 100 for m in methods]
    cis = [results_dict[m][1] * 100 for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=cis, capsize=5, color='steelblue',
                  edgecolor='black', alpha=0.8)

    ax.set_xlabel('Method')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Few-Shot Learning Method Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, mean, ci in zip(bars, means, cis):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci + 1,
                f'{mean:.1f}±{ci:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_fed_comparison(history, save_dir, prefix='cifar_fed_compare'):
    """
    绘制联邦三方法对比的两幅图: loss vs round, accuracy vs round

    Args:
        history: {method_name: {'rounds':[...], 'loss':[...],
                                'acc':[...], 'acc_std':[...]}}
        save_dir: 输出目录
        prefix: 文件名前缀

    Returns:
        (loss_path, acc_path)
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    colors = {
        'FedAvg': '#1f77b4',
        'FedAvg+MAML': '#ff7f0e',
        'FedAvg+Meta-SGD': '#2ca02c',
    }

    def color_for(name, i):
        return colors.get(name, plt.cm.tab10(i % 10))

    # 图1: Loss vs round
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (method, h) in enumerate(history.items()):
        ax.plot(h['rounds'], h['loss'], marker='o', markersize=3,
                label=method, color=color_for(method, i))
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Loss (adapt-then-eval)')
    ax.set_title('Federated Learning: Test Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(save_dir, f'{prefix}_loss.png')
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to: {loss_path}")

    # 图2: Accuracy vs round (含 ±std 阴影带, 体现稳定性)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (method, h) in enumerate(history.items()):
        rounds = np.array(h['rounds'])
        acc = np.array(h['acc']) * 100
        c = color_for(method, i)
        ax.plot(rounds, acc, marker='o', markersize=3, label=method, color=c)
        if 'acc_std' in h and h['acc_std'] is not None:
            std = np.array(h['acc_std']) * 100
            ax.fill_between(rounds, acc - std, acc + std, color=c, alpha=0.15)
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Federated Learning: Test Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(save_dir, f'{prefix}_acc.png')
    plt.savefig(acc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Accuracy plot saved to: {acc_path}")

    return loss_path, acc_path


if __name__ == '__main__':
    # 测试可视化
    # 模拟数据
    epochs = 50
    train_losses = [2.0 - i * 0.03 + np.random.randn() * 0.1 for i in range(epochs)]
    train_accs = [0.3 + i * 0.01 + np.random.randn() * 0.02 for i in range(epochs)]
    val_losses = [2.2 - i * 0.025 + np.random.randn() * 0.15 for i in range(epochs)]
    val_accs = [0.25 + i * 0.008 + np.random.randn() * 0.03 for i in range(epochs)]

    plot_learning_curves(train_losses, train_accs, val_losses, val_accs)

    # 模拟episode准确率
    accuracies = np.random.normal(0.5, 0.1, 600).clip(0, 1).tolist()
    plot_episode_accuracy(accuracies)

    # 方法对比
    results = {
        'Random Finetune': (0.32, 0.02),
        'ProtoNet': (0.38, 0.02),
        'FOMAML': (0.45, 0.02),
        'MAML': (0.48, 0.02)
    }
    plot_comparison(results)
