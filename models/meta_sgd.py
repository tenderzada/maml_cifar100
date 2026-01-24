"""
Meta-SGD (Meta-SGD: Learning to Learn Quickly for Few-Shot Learning)
Li et al., 2017

核心思想:
- 不仅学习初始参数θ，还学习可学习的学习率向量α
- α与θ同维度，允许每个参数有独立的学习率和更新方向
- 内层更新: θ' = θ - α ⊙ ∇θ L(θ)，其中⊙表示逐元素乘法

与MAML的区别:
- MAML: 使用固定标量学习率 lr
- Meta-SGD: 使用可学习的向量学习率 α (与参数同维度)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import numpy as np


class MetaSGD(nn.Module):
    """
    Meta-SGD元学习器
    """

    def __init__(self, model, inner_lr_init=0.01, inner_steps=1,
                 first_order=False, device='cuda'):
        """
        Args:
            model: 基础模型 (Functional模型)
            inner_lr_init: 学习率向量的初始值
            inner_steps: 内层梯度更新步数 (Meta-SGD原论文用1步)
            first_order: 是否使用一阶近似
            device: 计算设备
        """
        super(MetaSGD, self).__init__()

        self.model = model
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.device = device

        # 将模型移到指定设备
        self.model = self.model.to(device)

        # 创建可学习的学习率向量α (与模型参数同维度)
        # 每个参数对应一个学习率
        self.alpha = nn.ParameterList()
        for param in self.model.vars:
            # 初始化为inner_lr_init，使用log空间确保正值
            alpha_param = nn.Parameter(
                torch.ones_like(param) * inner_lr_init
            )
            self.alpha.append(alpha_param)

        # 将alpha移到设备
        self.alpha = self.alpha.to(device)

    def inner_loop(self, support_x, support_y, vars=None, alpha=None):
        """
        内层循环: 使用可学习学习率在support set上更新

        Args:
            support_x: support set [n_way * k_shot, C, L]
            support_y: support set labels [n_way * k_shot]
            vars: 初始参数
            alpha: 学习率向量

        Returns:
            更新后的参数
        """
        if vars is None:
            vars = list(self.model.vars)
        if alpha is None:
            alpha = list(self.alpha)

        for step in range(self.inner_steps):
            # 前向传播
            logits = self.model(support_x, vars=vars, bn_training=True)
            loss = F.cross_entropy(logits, support_y)

            # 计算梯度
            grads = autograd.grad(loss, vars, create_graph=not self.first_order)

            # Meta-SGD更新: θ' = θ - α ⊙ ∇θL
            # α是可学习的，与参数同维度
            vars = [v - a * g for v, a, g in zip(vars, alpha, grads)]

        return vars

    def forward(self, support_x, support_y, query_x, query_y):
        """
        Meta-SGD前向传播

        Returns:
            query_losses: 每个任务的query loss
            query_accs: 每个任务的query accuracy
        """
        batch_size = support_x.size(0)
        query_losses = []
        query_accs = []

        for i in range(batch_size):
            sx = support_x[i]
            sy = support_y[i]
            qx = query_x[i]
            qy = query_y[i]

            # 内层循环: 使用可学习学习率适应
            adapted_vars = self.inner_loop(sx, sy)

            # 在query set上评估
            query_logits = self.model(qx, vars=adapted_vars, bn_training=True)
            query_loss = F.cross_entropy(query_logits, qy)
            query_losses.append(query_loss)

            # 计算准确率
            with torch.no_grad():
                query_pred = query_logits.argmax(dim=1)
                query_acc = (query_pred == qy).float().mean()
                query_accs.append(query_acc)

        meta_loss = torch.stack(query_losses).mean()
        meta_acc = torch.stack(query_accs).mean()

        return meta_loss, meta_acc

    def adapt(self, support_x, support_y):
        """
        适应新任务 (用于测试)

        Returns:
            adapted_vars: 适应后的模型参数
        """
        return self.inner_loop(support_x, support_y)

    def evaluate(self, query_x, query_y, adapted_vars):
        """
        使用适应后的参数在query set上评估

        Returns:
            loss, accuracy
        """
        with torch.no_grad():
            logits = self.model(query_x, vars=adapted_vars, bn_training=False)
            loss = F.cross_entropy(logits, query_y)
            pred = logits.argmax(dim=1)
            acc = (pred == query_y).float().mean()

        return loss.item(), acc.item()

    def get_all_parameters(self):
        """
        获取所有可学习参数 (模型参数 + 学习率向量)
        用于外层优化
        """
        return list(self.model.vars) + list(self.alpha)


class MetaSGDTrainer:
    """
    Meta-SGD训练器
    """

    def __init__(self, meta_sgd, outer_lr=0.001, weight_decay=0.0,
                 alpha_lr=0.001, device='cuda'):
        """
        Args:
            meta_sgd: MetaSGD模型
            outer_lr: 外层学习率 (模型参数)
            weight_decay: 权重衰减
            alpha_lr: 学习率向量的学习率
            device: 计算设备
        """
        self.meta_sgd = meta_sgd
        self.device = device

        # 分开优化模型参数和学习率向量
        # 可以使用不同的学习率
        self.optimizer = torch.optim.Adam([
            {'params': meta_sgd.model.vars, 'lr': outer_lr, 'weight_decay': weight_decay},
            {'params': meta_sgd.alpha, 'lr': alpha_lr, 'weight_decay': 0}  # alpha不用weight_decay
        ])

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5
        )

    def train_epoch(self, train_loader, epoch):
        """
        训练一个epoch

        Returns:
            平均loss和准确率
        """
        self.meta_sgd.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        for batch_idx, (support_x, support_y, query_x, query_y) in enumerate(train_loader):
            support_x = support_x.squeeze(0).to(self.device)
            support_y = support_y.squeeze(0).to(self.device)
            query_x = query_x.squeeze(0).to(self.device)
            query_y = query_y.squeeze(0).to(self.device)

            support_x = support_x.unsqueeze(0)
            support_y = support_y.unsqueeze(0)
            query_x = query_x.unsqueeze(0)
            query_y = query_y.unsqueeze(0)

            self.optimizer.zero_grad()
            meta_loss, meta_acc = self.meta_sgd(support_x, support_y, query_x, query_y)

            meta_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.meta_sgd.model.vars, max_norm=10)
            torch.nn.utils.clip_grad_norm_(self.meta_sgd.alpha, max_norm=10)

            self.optimizer.step()

            # 确保学习率向量为正值 (使用clamp)
            with torch.no_grad():
                for alpha in self.meta_sgd.alpha:
                    alpha.data.clamp_(min=1e-6, max=1.0)

            total_loss += meta_loss.item()
            total_acc += meta_acc.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                # 打印学习率统计
                alpha_mean = np.mean([a.mean().item() for a in self.meta_sgd.alpha])
                alpha_std = np.mean([a.std().item() for a in self.meta_sgd.alpha])
                print(f'  Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {meta_loss.item():.4f}, Acc: {meta_acc.item():.4f}, '
                      f'Alpha: {alpha_mean:.4f}±{alpha_std:.4f}')

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches

        return avg_loss, avg_acc

    def validate(self, val_loader):
        """
        在验证集上评估

        Returns:
            平均loss和准确率
        """
        self.meta_sgd.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        with torch.no_grad():
            for support_x, support_y, query_x, query_y in val_loader:
                support_x = support_x.squeeze(0).to(self.device)
                support_y = support_y.squeeze(0).to(self.device)
                query_x = query_x.squeeze(0).to(self.device)
                query_y = query_y.squeeze(0).to(self.device)

                with torch.enable_grad():
                    adapted_vars = self.meta_sgd.adapt(support_x, support_y)

                loss, acc = self.meta_sgd.evaluate(query_x, query_y, adapted_vars)

                total_loss += loss
                total_acc += acc
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches

        return avg_loss, avg_acc

    def step_scheduler(self):
        """更新学习率"""
        self.scheduler.step()


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from conv1d import Conv1D4Functional

    # 测试Meta-SGD
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Conv1D4Functional(in_channels=9, hidden_dim=64, n_way=5)
    meta_sgd = MetaSGD(model, inner_lr_init=0.01, inner_steps=1, device=device)

    # 打印参数数量
    model_params = sum(p.numel() for p in meta_sgd.model.vars)
    alpha_params = sum(p.numel() for p in meta_sgd.alpha)
    print(f"Model parameters: {model_params:,}")
    print(f"Alpha parameters: {alpha_params:,}")
    print(f"Total parameters: {model_params + alpha_params:,}")

    # 模拟数据
    support_x = torch.randn(1, 5, 9, 2048).to(device)  # 5-way 1-shot
    support_y = torch.arange(5).unsqueeze(0).to(device)
    query_x = torch.randn(1, 75, 9, 2048).to(device)
    query_y = torch.arange(5).repeat(15).unsqueeze(0).to(device)

    # 前向传播
    meta_loss, meta_acc = meta_sgd(support_x, support_y, query_x, query_y)
    print(f"\nMeta Loss: {meta_loss.item():.4f}")
    print(f"Meta Acc: {meta_acc.item():.4f}")

    # 打印学习率分布
    print("\nLearned learning rates:")
    for i, alpha in enumerate(meta_sgd.alpha):
        print(f"  Layer {i}: mean={alpha.mean().item():.6f}, "
              f"std={alpha.std().item():.6f}, "
              f"min={alpha.min().item():.6f}, max={alpha.max().item():.6f}")
