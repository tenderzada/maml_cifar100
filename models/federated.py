"""
联邦学习算法实现

1. FedAvg (Federated Averaging)
   - McMahan et al., "Communication-Efficient Learning of Deep Networks
     from Decentralized Data", AISTATS 2017

2. FedAvg + MAML (联邦元学习)
   - 结合FedAvg的通信机制和MAML的元学习策略
   - 每个客户端执行MAML内层更新，服务器聚合元参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import numpy as np
from copy import deepcopy
from typing import List, Dict, Optional
import random


class FedAvg:
    """
    FedAvg (Federated Averaging) 算法

    流程:
    1. 服务器广播全局模型给选中的客户端
    2. 各客户端在本地数据上训练模型
    3. 服务器收集并平均更新后的模型参数
    """

    def __init__(
        self,
        global_model: nn.Module,
        num_clients: int = 10,
        clients_per_round: int = 2,
        local_epochs: int = 5,
        local_lr: float = 0.01,
        weight_decay: float = 0.0,
        device: str = 'cuda',
        use_data_parallel: bool = False
    ):
        """
        Args:
            global_model: 全局模型
            num_clients: 客户端总数
            clients_per_round: 每轮选择的客户端数量
            local_epochs: 本地训练轮数
            local_lr: 本地学习率
            weight_decay: 权重衰减
            device: 计算设备
            use_data_parallel: 是否使用多GPU数据并行
        """
        self.device = device
        self.use_data_parallel = use_data_parallel

        # 多GPU数据并行
        if use_data_parallel and torch.cuda.device_count() > 1:
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            self.global_model = nn.DataParallel(global_model).to(device)
        else:
            self.global_model = global_model.to(device)

        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.weight_decay = weight_decay

        # 客户端模型 (训练时创建)
        self.client_models = None

    def select_clients(self) -> List[int]:
        """随机选择参与本轮训练的客户端"""
        return random.sample(range(self.num_clients), self.clients_per_round)

    def broadcast_global_model(self, client_ids: List[int]) -> List[nn.Module]:
        """将全局模型广播给选中的客户端"""
        client_models = []
        # 获取基础模型 (去除DataParallel包装)
        base_model = self.global_model.module if self.use_data_parallel else self.global_model
        for _ in client_ids:
            # 深拷贝基础模型
            client_model = deepcopy(base_model)
            # 客户端训练时也使用DataParallel
            if self.use_data_parallel and torch.cuda.device_count() > 1:
                client_model = nn.DataParallel(client_model).to(self.device)
            client_models.append(client_model)
        return client_models

    def local_train(
        self,
        client_model: nn.Module,
        train_loader,
        client_id: int
    ) -> Dict:
        """
        客户端本地训练

        Returns:
            训练统计信息
        """
        client_model.train()
        optimizer = torch.optim.SGD(
            client_model.parameters(),
            lr=self.local_lr,
            momentum=0.9,
            weight_decay=self.weight_decay
        )

        total_loss = 0
        total_correct = 0
        total_samples = 0

        for epoch in range(self.local_epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                logits = client_model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                total_correct += (pred == y).sum().item()
                total_samples += x.size(0)

        return {
            'loss': total_loss / total_samples if total_samples > 0 else 0,
            'accuracy': total_correct / total_samples if total_samples > 0 else 0,
            'samples': total_samples
        }

    def aggregate(self, client_models: List[nn.Module], client_weights: List[float]):
        """
        聚合客户端模型参数 (加权平均)

        Args:
            client_models: 客户端模型列表
            client_weights: 权重 (通常按样本数加权)
        """
        # 归一化权重
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]

        # 获取全局模型的state_dict
        global_state = self.global_model.state_dict()

        # 加权平均
        for key in global_state.keys():
            global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
            for client_model, weight in zip(client_models, client_weights):
                client_state = client_model.state_dict()
                # DataParallel的state_dict键名相同
                global_state[key] += weight * client_state[key].float()

        self.global_model.load_state_dict(global_state)

    def get_model_state_dict(self):
        """获取模型state_dict (处理DataParallel)"""
        if self.use_data_parallel:
            return self.global_model.module.state_dict()
        return self.global_model.state_dict()

    def train_round(self, client_loaders: List) -> Dict:
        """
        执行一轮联邦训练

        Args:
            client_loaders: 各客户端的数据加载器

        Returns:
            本轮训练统计
        """
        # 1. 选择客户端
        selected_clients = self.select_clients()

        # 2. 广播全局模型
        client_models = self.broadcast_global_model(selected_clients)

        # 3. 本地训练
        client_stats = []
        client_weights = []

        for i, client_id in enumerate(selected_clients):
            stats = self.local_train(
                client_models[i],
                client_loaders[client_id],
                client_id
            )
            client_stats.append(stats)
            client_weights.append(stats['samples'])

        # 4. 聚合模型
        self.aggregate(client_models, client_weights)

        # 统计
        avg_loss = np.mean([s['loss'] for s in client_stats])
        avg_acc = np.mean([s['accuracy'] for s in client_stats])

        return {
            'selected_clients': selected_clients,
            'loss': avg_loss,
            'accuracy': avg_acc,
            'client_stats': client_stats
        }

    def evaluate(self, test_loader) -> Dict:
        """
        在测试集上评估全局模型
        """
        self.global_model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.global_model(x)
                loss = F.cross_entropy(logits, y)

                total_loss += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                total_correct += (pred == y).sum().item()
                total_samples += x.size(0)

        return {
            'loss': total_loss / total_samples if total_samples > 0 else 0,
            'accuracy': total_correct / total_samples if total_samples > 0 else 0
        }


class FedMAML:
    """
    FedAvg + MAML (联邦元学习)

    结合联邦学习和元学习:
    - 服务器维护全局元参数 θ
    - 每轮选择部分客户端
    - 各客户端在本地执行MAML训练 (内层适应 + 外层更新)
    - 服务器聚合各客户端的元参数更新

    适用于客户端数据异构的场景
    """

    def __init__(
        self,
        model,  # Functional model
        num_clients: int = 10,
        clients_per_round: int = 2,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        outer_lr: float = 0.001,
        weight_decay: float = 0.0,
        local_meta_steps: int = 5,  # 每个客户端的元更新步数
        first_order: bool = False,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Functional模型 (支持传入外部参数)
            num_clients: 客户端总数
            clients_per_round: 每轮选择的客户端数量
            inner_lr: MAML内层学习率
            inner_steps: MAML内层更新步数
            outer_lr: 外层 (元) 学习率
            weight_decay: 权重衰减
            local_meta_steps: 每个客户端执行的元更新步数
            first_order: 是否使用一阶近似
            device: 计算设备
        """
        self.model = model.to(device)
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.outer_lr = outer_lr
        self.weight_decay = weight_decay
        self.local_meta_steps = local_meta_steps
        self.first_order = first_order
        self.device = device

    def select_clients(self) -> List[int]:
        """随机选择参与本轮训练的客户端"""
        return random.sample(range(self.num_clients), self.clients_per_round)

    def inner_loop(self, support_x, support_y, vars):
        """
        MAML内层循环

        Args:
            support_x: support set
            support_y: support labels
            vars: 初始参数

        Returns:
            adapted_vars: 适应后的参数
        """
        for step in range(self.inner_steps):
            logits = self.model(support_x, vars=vars, bn_training=True)
            loss = F.cross_entropy(logits, support_y)
            grads = autograd.grad(loss, vars, create_graph=not self.first_order)
            vars = [v - self.inner_lr * g for v, g in zip(vars, grads)]
        return vars

    def client_meta_update(
        self,
        client_loader,
        client_id: int
    ) -> Dict:
        """
        客户端执行本地MAML更新

        Returns:
            更新后的参数和统计信息
        """
        # 复制全局参数
        local_vars = [p.clone().detach().requires_grad_(True) for p in self.model.vars]

        # 本地元优化器 (带权重衰减)
        local_optimizer = torch.optim.Adam(
            local_vars, lr=self.outer_lr, weight_decay=self.weight_decay
        )

        total_loss = 0
        total_acc = 0
        num_episodes = 0

        for step, (support_x, support_y, query_x, query_y) in enumerate(client_loader):
            if step >= self.local_meta_steps:
                break

            support_x = support_x.squeeze(0).to(self.device)
            support_y = support_y.squeeze(0).to(self.device)
            query_x = query_x.squeeze(0).to(self.device)
            query_y = query_y.squeeze(0).to(self.device)

            local_optimizer.zero_grad()

            # 内层适应
            adapted_vars = self.inner_loop(support_x, support_y, local_vars)

            # 外层loss (query set)
            query_logits = self.model(query_x, vars=adapted_vars, bn_training=True)
            query_loss = F.cross_entropy(query_logits, query_y)

            # 计算准确率
            with torch.no_grad():
                query_pred = query_logits.argmax(dim=1)
                query_acc = (query_pred == query_y).float().mean().item()

            # 反向传播到local_vars
            query_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(local_vars, max_norm=10)

            local_optimizer.step()

            total_loss += query_loss.item()
            total_acc += query_acc
            num_episodes += 1

        # 计算参数更新量 (delta)
        param_updates = []
        for new_var, old_var in zip(local_vars, self.model.vars):
            param_updates.append(new_var.detach() - old_var.detach())

        return {
            'param_updates': param_updates,
            'loss': total_loss / num_episodes if num_episodes > 0 else 0,
            'accuracy': total_acc / num_episodes if num_episodes > 0 else 0,
            'num_episodes': num_episodes
        }

    def aggregate_updates(self, client_updates: List[Dict], client_weights: List[float]):
        """
        聚合客户端的参数更新

        Args:
            client_updates: 各客户端的更新信息
            client_weights: 权重
        """
        # 归一化权重
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]

        # 加权平均更新量
        with torch.no_grad():
            for i, param in enumerate(self.model.vars):
                avg_update = torch.zeros_like(param)
                for update_info, weight in zip(client_updates, client_weights):
                    avg_update += weight * update_info['param_updates'][i]
                param.data += avg_update

    def train_round(self, client_loaders: List) -> Dict:
        """
        执行一轮联邦元学习

        Args:
            client_loaders: 各客户端的Few-Shot数据加载器

        Returns:
            本轮训练统计
        """
        # 1. 选择客户端
        selected_clients = self.select_clients()

        # 2. 各客户端执行本地MAML更新
        client_updates = []
        client_weights = []

        for client_id in selected_clients:
            update_info = self.client_meta_update(
                client_loaders[client_id],
                client_id
            )
            client_updates.append(update_info)
            client_weights.append(update_info['num_episodes'])

        # 3. 聚合更新
        self.aggregate_updates(client_updates, client_weights)

        # 统计
        avg_loss = np.mean([u['loss'] for u in client_updates])
        avg_acc = np.mean([u['accuracy'] for u in client_updates])

        return {
            'selected_clients': selected_clients,
            'loss': avg_loss,
            'accuracy': avg_acc,
            'client_updates': client_updates
        }

    def adapt_and_evaluate(self, support_x, support_y, query_x, query_y):
        """
        适应并评估 (用于测试)
        """
        support_x = support_x.to(self.device)
        support_y = support_y.to(self.device)
        query_x = query_x.to(self.device)
        query_y = query_y.to(self.device)

        # 内层适应
        vars = list(self.model.vars)
        with torch.enable_grad():
            adapted_vars = self.inner_loop(support_x, support_y, vars)

        # 评估
        with torch.no_grad():
            logits = self.model(query_x, vars=adapted_vars, bn_training=False)
            loss = F.cross_entropy(logits, query_y)
            pred = logits.argmax(dim=1)
            acc = (pred == query_y).float().mean()

        return loss.item(), acc.item()

    def evaluate(self, test_loader) -> Dict:
        """
        在Few-Shot测试集上评估
        """
        self.model.eval()
        total_loss = 0
        total_acc = 0
        num_episodes = 0

        for support_x, support_y, query_x, query_y in test_loader:
            support_x = support_x.squeeze(0)
            support_y = support_y.squeeze(0)
            query_x = query_x.squeeze(0)
            query_y = query_y.squeeze(0)

            loss, acc = self.adapt_and_evaluate(
                support_x, support_y, query_x, query_y
            )

            total_loss += loss
            total_acc += acc
            num_episodes += 1

        return {
            'loss': total_loss / num_episodes if num_episodes > 0 else 0,
            'accuracy': total_acc / num_episodes if num_episodes > 0 else 0
        }


class FedPerMAML:
    """
    固定类别版 Per-FedAvg (FedAvg + MAML)
    Fallah et al., "Personalized Federated Learning: A Meta-Learning Approach", NeurIPS 2020

    与 FedMAML (episodic, 相对标签) 不同, 这里做绝对 num_classes 类分类:
    - 服务器维护全局 functional 模型参数 θ
    - 每轮选客户端, 客户端用本地数据做 MAML 本地更新 (内层 support 适应, 外层 query meta-loss)
    - 服务器聚合参数增量 Δθ
    - 评测: 全局模型在测试 support 上内层适应几步, 再在 query 上评测 (adapt-then-eval)
    """

    def __init__(self, model, num_clients=10, clients_per_round=5,
                 inner_lr=0.01, inner_steps=5, outer_lr=0.001,
                 local_meta_steps=5, weight_decay=0.0,
                 first_order=True, grad_clip=10.0, device='cuda'):
        self.model = model.to(device)
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.outer_lr = outer_lr
        self.local_meta_steps = local_meta_steps
        self.weight_decay = weight_decay
        self.first_order = first_order
        self.grad_clip = grad_clip
        self.device = device

    def select_clients(self):
        return random.sample(range(self.num_clients), self.clients_per_round)

    def inner_loop(self, support_x, support_y, vars, create_graph):
        """MAML 内层: 标量学习率 self.inner_lr"""
        for _ in range(self.inner_steps):
            logits = self.model(support_x, vars=vars, bn_training=True)
            loss = F.cross_entropy(logits, support_y)
            grads = autograd.grad(loss, vars, create_graph=create_graph)
            vars = [v - self.inner_lr * g for v, g in zip(vars, grads)]
        return vars

    def _local_params(self):
        """客户端本地可优化参数 (此处仅 θ)"""
        return [p.clone().detach().requires_grad_(True) for p in self.model.vars]

    def client_update(self, sampler_fn, client_id):
        """客户端本地 MAML 更新, 返回参数增量与统计"""
        local_vars = self._local_params()
        optimizer = torch.optim.Adam(local_vars, lr=self.outer_lr,
                                     weight_decay=self.weight_decay)

        total_loss, total_acc, n = 0.0, 0.0, 0
        for _ in range(self.local_meta_steps):
            sx, sy, qx, qy = sampler_fn(client_id)
            sx, sy = sx.to(self.device), sy.to(self.device)
            qx, qy = qx.to(self.device), qy.to(self.device)

            optimizer.zero_grad()
            adapted = self.inner_loop(sx, sy, local_vars,
                                      create_graph=not self.first_order)
            q_logits = self.model(qx, vars=adapted, bn_training=True)
            q_loss = F.cross_entropy(q_logits, qy)
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(local_vars, self.grad_clip)
            optimizer.step()

            with torch.no_grad():
                acc = (q_logits.argmax(1) == qy).float().mean().item()
            total_loss += q_loss.item(); total_acc += acc; n += 1

        theta_delta = [nv.detach() - ov.detach()
                       for nv, ov in zip(local_vars, self.model.vars)]
        return {
            'theta_delta': theta_delta,
            'loss': total_loss / n if n else 0.0,
            'accuracy': total_acc / n if n else 0.0,
            'steps': n,
        }

    def aggregate(self, updates, weights):
        total = sum(weights)
        weights = [w / total for w in weights]
        with torch.no_grad():
            for i, param in enumerate(self.model.vars):
                agg = torch.zeros_like(param)
                for u, w in zip(updates, weights):
                    agg += w * u['theta_delta'][i]
                param.data += agg

    def train_round(self, sampler_fn):
        selected = self.select_clients()
        updates, weights = [], []
        for cid in selected:
            u = self.client_update(sampler_fn, cid)
            updates.append(u); weights.append(u['steps'])
        self.aggregate(updates, weights)
        return {
            'selected_clients': selected,
            'loss': float(np.mean([u['loss'] for u in updates])),
            'accuracy': float(np.mean([u['accuracy'] for u in updates])),
        }

    def adapt_and_evaluate(self, support_x, support_y, query_x, query_y):
        support_x, support_y = support_x.to(self.device), support_y.to(self.device)
        query_x, query_y = query_x.to(self.device), query_y.to(self.device)
        vars = [p.clone().detach().requires_grad_(True) for p in self.model.vars]
        with torch.enable_grad():
            adapted = self.inner_loop(support_x, support_y, vars, create_graph=False)
        with torch.no_grad():
            logits = self.model(query_x, vars=adapted, bn_training=True)
            loss = F.cross_entropy(logits, query_y).item()
            acc = (logits.argmax(1) == query_y).float().mean().item()
        return loss, acc

    def evaluate_episodes(self, episodes):
        losses, accs = [], []
        for sx, sy, qx, qy in episodes:
            l, a = self.adapt_and_evaluate(sx, sy, qx, qy)
            losses.append(l); accs.append(a)
        return float(np.mean(losses)), float(np.mean(accs)), float(np.std(accs))


class FedPerMetaSGD(FedPerMAML):
    """
    固定类别版 Per-FedAvg + Meta-SGD

    在 FedPerMAML 基础上引入可学习的逐参数学习率向量 α (与 θ 同形状):
    - 内层更新: θ' = θ - α ⊙ ∇θ L
    - α 是全局元参数, 与 θ 一同在外层优化、聚合
    - 相比标量 inner_lr, 每个参数有独立学习率/方向, 适应更快更稳
    """

    def __init__(self, model, alpha_init=0.01, alpha_lr=0.001, **kwargs):
        super().__init__(model, **kwargs)
        self.alpha_lr = alpha_lr
        # 全局学习率向量 (与 θ 同形状), 存为普通 tensor (非 nn.Parameter)
        self.alpha = [torch.ones_like(p, device=self.device) * alpha_init
                      for p in self.model.vars]

    def inner_loop(self, support_x, support_y, vars, create_graph, alpha=None):
        if alpha is None:
            alpha = self.alpha
        for _ in range(self.inner_steps):
            logits = self.model(support_x, vars=vars, bn_training=True)
            loss = F.cross_entropy(logits, support_y)
            grads = autograd.grad(loss, vars, create_graph=create_graph)
            vars = [v - a * g for v, a, g in zip(vars, alpha, grads)]
        return vars

    def client_update(self, sampler_fn, client_id):
        local_vars = self._local_params()
        local_alpha = [a.clone().detach().requires_grad_(True) for a in self.alpha]
        optimizer = torch.optim.Adam(
            [{'params': local_vars, 'lr': self.outer_lr,
              'weight_decay': self.weight_decay},
             {'params': local_alpha, 'lr': self.alpha_lr, 'weight_decay': 0.0}]
        )

        total_loss, total_acc, n = 0.0, 0.0, 0
        for _ in range(self.local_meta_steps):
            sx, sy, qx, qy = sampler_fn(client_id)
            sx, sy = sx.to(self.device), sy.to(self.device)
            qx, qy = qx.to(self.device), qy.to(self.device)

            optimizer.zero_grad()
            adapted = self.inner_loop(sx, sy, local_vars,
                                      create_graph=not self.first_order,
                                      alpha=local_alpha)
            q_logits = self.model(qx, vars=adapted, bn_training=True)
            q_loss = F.cross_entropy(q_logits, qy)
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(local_vars, self.grad_clip)
            torch.nn.utils.clip_grad_norm_(local_alpha, self.grad_clip)
            optimizer.step()
            with torch.no_grad():
                for a in local_alpha:
                    a.clamp_(min=1e-6, max=1.0)
                acc = (q_logits.argmax(1) == qy).float().mean().item()
            total_loss += q_loss.item(); total_acc += acc; n += 1

        theta_delta = [nv.detach() - ov.detach()
                       for nv, ov in zip(local_vars, self.model.vars)]
        alpha_delta = [na.detach() - oa.detach()
                       for na, oa in zip(local_alpha, self.alpha)]
        return {
            'theta_delta': theta_delta,
            'alpha_delta': alpha_delta,
            'loss': total_loss / n if n else 0.0,
            'accuracy': total_acc / n if n else 0.0,
            'steps': n,
        }

    def aggregate(self, updates, weights):
        total = sum(weights)
        weights = [w / total for w in weights]
        with torch.no_grad():
            for i, param in enumerate(self.model.vars):
                agg = torch.zeros_like(param)
                for u, w in zip(updates, weights):
                    agg += w * u['theta_delta'][i]
                param.data += agg
            for i in range(len(self.alpha)):
                agg = torch.zeros_like(self.alpha[i])
                for u, w in zip(updates, weights):
                    agg += w * u['alpha_delta'][i]
                self.alpha[i] += agg
                self.alpha[i].clamp_(min=1e-6, max=1.0)


if __name__ == '__main__':
    from conv1d import Conv1D4, Conv1D4Functional

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 测试FedAvg
    print("\n" + "=" * 50)
    print("Testing FedAvg")
    print("=" * 50)

    model = Conv1D4(in_channels=9, hidden_dim=64, n_way=40)  # 40 classes for training
    fedavg = FedAvg(
        global_model=model,
        num_clients=10,
        clients_per_round=2,
        local_epochs=5,
        local_lr=0.01,
        device=device
    )

    print(f"Clients per round: {fedavg.clients_per_round}")
    selected = fedavg.select_clients()
    print(f"Selected clients: {selected}")

    # 测试FedMAML
    print("\n" + "=" * 50)
    print("Testing FedMAML")
    print("=" * 50)

    model_func = Conv1D4Functional(in_channels=9, hidden_dim=64, n_way=5)
    fedmaml = FedMAML(
        model=model_func,
        num_clients=10,
        clients_per_round=2,
        inner_lr=0.01,
        inner_steps=5,
        outer_lr=0.001,
        local_meta_steps=5,
        device=device
    )

    print(f"Clients per round: {fedmaml.clients_per_round}")
    print(f"Inner steps: {fedmaml.inner_steps}")
    print(f"Local meta steps: {fedmaml.local_meta_steps}")

    # 模拟数据测试
    support_x = torch.randn(5, 9, 2048).to(device)  # 5-way 1-shot
    support_y = torch.arange(5).to(device)
    query_x = torch.randn(75, 9, 2048).to(device)
    query_y = torch.arange(5).repeat(15).to(device)

    loss, acc = fedmaml.adapt_and_evaluate(support_x, support_y, query_x, query_y)
    print(f"Test episode - Loss: {loss:.4f}, Acc: {acc:.4f}")
