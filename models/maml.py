"""
MAML (Model-Agnostic Meta-Learning) 算法实现

核心思想:
- 学习一个好的模型初始化参数θ
- 使得从θ出发，经过少量梯度更新后，模型能够快速适应新任务

算法流程:
1. 对每个任务Ti采样support set和query set
2. 内层循环: 在support set上进行K步梯度下降，得到θ'i
3. 外层循环: 基于query set的loss更新θ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from copy import deepcopy
import numpy as np


class MAML(nn.Module):
    """
    MAML元学习器
    """

    def __init__(self, model, inner_lr=0.01, inner_steps=5,
                 first_order=False, device='cuda'):
        """
        Args:
            model: 基础模型 (Conv4Functional)
            inner_lr: 内层学习率 (task-specific adaptation)
            inner_steps: 内层梯度更新步数
            first_order: 是否使用一阶近似 (FOMAML)
            device: 计算设备
        """
        super(MAML, self).__init__()

        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.device = device

        # 将模型移到指定设备
        self.model = self.model.to(device)

    def inner_loop(self, support_x, support_y, vars=None):
        """
        内层循环: 在support set上进行梯度更新

        Args:
            support_x: support set images [n_way * k_shot, C, H, W]
            support_y: support set labels [n_way * k_shot]
            vars: 初始参数 (如果为None则使用模型当前参数)

        Returns:
            更新后的参数
        """
        if vars is None:
            vars = list(self.model.vars)

        for step in range(self.inner_steps):
            # 前向传播
            logits = self.model(support_x, vars=vars, bn_training=True)
            loss = F.cross_entropy(logits, support_y)

            # 计算梯度
            grads = autograd.grad(loss, vars, create_graph=not self.first_order)

            # 梯度下降更新
            vars = [v - self.inner_lr * g for v, g in zip(vars, grads)]

        return vars

    def forward(self, support_x, support_y, query_x, query_y):
        """
        MAML前向传播

        Args:
            support_x: [batch, n_way * k_shot, C, H, W]
            support_y: [batch, n_way * k_shot]
            query_x: [batch, n_way * k_query, C, H, W]
            query_y: [batch, n_way * k_query]

        Returns:
            query_losses: 每个任务在query set上的loss
            query_accs: 每个任务在query set上的准确率
        """
        batch_size = support_x.size(0)
        query_losses = []
        query_accs = []

        for i in range(batch_size):
            # 获取当前任务的数据
            sx = support_x[i]  # [n_way * k_shot, C, H, W]
            sy = support_y[i]  # [n_way * k_shot]
            qx = query_x[i]    # [n_way * k_query, C, H, W]
            qy = query_y[i]    # [n_way * k_query]

            # 内层循环: 在support set上适应
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

        # 平均loss用于外层优化
        meta_loss = torch.stack(query_losses).mean()
        meta_acc = torch.stack(query_accs).mean()

        return meta_loss, meta_acc

    def adapt(self, support_x, support_y):
        """
        适应新任务 (用于测试时)

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


class MAMLTrainer:
    """
    MAML训练器
    """

    def __init__(self, maml, outer_lr=0.001, weight_decay=0.0, device='cuda'):
        """
        Args:
            maml: MAML模型
            outer_lr: 外层学习率 (meta-learning rate)
            weight_decay: 权重衰减系数
            device: 计算设备
        """
        self.maml = maml
        self.device = device

        # 外层优化器 (带权重衰减)
        self.optimizer = torch.optim.Adam(
            maml.model.vars, lr=outer_lr, weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5
        )

    def train_epoch(self, train_loader, epoch):
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch数

        Returns:
            平均loss和准确率
        """
        self.maml.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        for batch_idx, (support_x, support_y, query_x, query_y) in enumerate(train_loader):
            # 移到GPU
            support_x = support_x.squeeze(0).to(self.device)
            support_y = support_y.squeeze(0).to(self.device)
            query_x = query_x.squeeze(0).to(self.device)
            query_y = query_y.squeeze(0).to(self.device)

            # 添加batch维度 (因为dataloader返回的batch_size通常为1)
            support_x = support_x.unsqueeze(0)
            support_y = support_y.unsqueeze(0)
            query_x = query_x.unsqueeze(0)
            query_y = query_y.unsqueeze(0)

            # 前向传播
            self.optimizer.zero_grad()
            meta_loss, meta_acc = self.maml(support_x, support_y, query_x, query_y)

            # 反向传播
            meta_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.maml.model.vars, max_norm=10)

            # 更新参数
            self.optimizer.step()

            total_loss += meta_loss.item()
            total_acc += meta_acc.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {meta_loss.item():.4f}, Acc: {meta_acc.item():.4f}')

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches

        return avg_loss, avg_acc

    def validate(self, val_loader):
        """
        在验证集上评估

        Returns:
            平均loss和准确率
        """
        self.maml.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        with torch.no_grad():
            for support_x, support_y, query_x, query_y in val_loader:
                support_x = support_x.squeeze(0).to(self.device)
                support_y = support_y.squeeze(0).to(self.device)
                query_x = query_x.squeeze(0).to(self.device)
                query_y = query_y.squeeze(0).to(self.device)

                # 内层循环仍需要梯度 (但不用于外层优化)
                with torch.enable_grad():
                    adapted_vars = self.maml.adapt(support_x, support_y)

                loss, acc = self.maml.evaluate(query_x, query_y, adapted_vars)

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
    from conv4 import Conv4Functional

    # 测试MAML
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Conv4Functional(n_way=5)
    maml = MAML(model, inner_lr=0.01, inner_steps=5, device=device)

    # 模拟数据
    support_x = torch.randn(1, 5, 3, 32, 32).to(device)  # 5-way 1-shot
    support_y = torch.arange(5).unsqueeze(0).to(device)
    query_x = torch.randn(1, 75, 3, 32, 32).to(device)   # 15 queries per class
    query_y = torch.arange(5).repeat(15).unsqueeze(0).to(device)

    # 前向传播
    meta_loss, meta_acc = maml(support_x, support_y, query_x, query_y)
    print(f"Meta Loss: {meta_loss.item():.4f}")
    print(f"Meta Acc: {meta_acc.item():.4f}")
