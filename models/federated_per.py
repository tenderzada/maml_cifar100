"""
Per-FedAvg 与 Meta-SGD 联邦版的通用实现 (任意 nn.Module 骨干)

使用 torch.func.functional_call 解除对手写 functional 模型的依赖,
可直接接入 ResNet18 / ResNet34 等任意标准 nn.Module 骨干。

类:
- FedPerMAML:    Fallah et al., NeurIPS 2020 (Per-FedAvg, 基于 MAML)
- FedPerMetaSGD: 在 Per-FedAvg 基础上引入可学习逐参数学习率 alpha

约定:
- 骨干使用 BN 时建议 track_running_stats=False (避免内层适应时 BN 缓冲区不一致),
  本项目的 ResNet12 / ResNet18 / ResNet34 (CIFAR 版) 已默认这样配置。
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.func import functional_call


class FedPerMAML:
    def __init__(self, model, num_clients=10, clients_per_round=5,
                 inner_lr=0.01, inner_steps=5, outer_lr=0.001,
                 local_meta_steps=30, weight_decay=0.0,
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

    def _clone_params(self):
        return {n: p.detach().clone().requires_grad_(True)
                for n, p in self.model.named_parameters()}

    def _call(self, params, x):
        return functional_call(self.model, params, (x,))

    def inner_loop(self, sx, sy, params, create_graph):
        for _ in range(self.inner_steps):
            logits = self._call(params, sx)
            loss = F.cross_entropy(logits, sy)
            grads = autograd.grad(loss, list(params.values()),
                                  create_graph=create_graph)
            params = {n: p - self.inner_lr * g
                      for (n, p), g in zip(params.items(), grads)}
        return params

    def client_update(self, sampler_fn, client_id):
        local = self._clone_params()
        opt = torch.optim.Adam(list(local.values()), lr=self.outer_lr,
                               weight_decay=self.weight_decay)
        old = {n: p.detach().clone()
               for n, p in self.model.named_parameters()}

        total_loss, total_acc, n = 0.0, 0.0, 0
        for _ in range(self.local_meta_steps):
            sx, sy, qx, qy = sampler_fn(client_id)
            sx, sy = sx.to(self.device), sy.to(self.device)
            qx, qy = qx.to(self.device), qy.to(self.device)

            opt.zero_grad()
            adapted = self.inner_loop(sx, sy, local,
                                      create_graph=not self.first_order)
            q_logits = self._call(adapted, qx)
            q_loss = F.cross_entropy(q_logits, qy)
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(local.values()), self.grad_clip)
            opt.step()

            with torch.no_grad():
                acc = (q_logits.argmax(1) == qy).float().mean().item()
            total_loss += q_loss.item(); total_acc += acc; n += 1

        delta = {nm: (local[nm].detach() - old[nm]) for nm in local}
        return {'delta': delta,
                'loss': total_loss / n if n else 0.0,
                'accuracy': total_acc / n if n else 0.0,
                'steps': n}

    def aggregate(self, updates, weights):
        total = sum(weights)
        weights = [w / total for w in weights]
        with torch.no_grad():
            for nm, p in self.model.named_parameters():
                agg = torch.zeros_like(p)
                for u, w in zip(updates, weights):
                    agg += w * u['delta'][nm]
                p.data += agg

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

    def adapt_and_evaluate(self, sx, sy, qx, qy):
        sx, sy = sx.to(self.device), sy.to(self.device)
        qx, qy = qx.to(self.device), qy.to(self.device)
        local = self._clone_params()
        with torch.enable_grad():
            adapted = self.inner_loop(sx, sy, local, create_graph=False)
        with torch.no_grad():
            logits = self._call(adapted, qx)
            loss = F.cross_entropy(logits, qy).item()
            acc = (logits.argmax(1) == qy).float().mean().item()
        return loss, acc

    def evaluate_episodes(self, episodes):
        losses, accs = [], []
        for sx, sy, qx, qy in episodes:
            l, a = self.adapt_and_evaluate(sx, sy, qx, qy)
            losses.append(l); accs.append(a)
        return (float(np.mean(losses)),
                float(np.mean(accs)),
                float(np.std(accs)))


class FedPerMetaSGD(FedPerMAML):
    def __init__(self, model, alpha_init=0.01, alpha_lr=0.005, **kwargs):
        super().__init__(model, **kwargs)
        self.alpha_lr = alpha_lr
        self.alpha = {n: torch.ones_like(p, device=self.device) * alpha_init
                      for n, p in self.model.named_parameters()}

    def inner_loop(self, sx, sy, params, create_graph, alpha=None):
        if alpha is None:
            alpha = self.alpha
        for _ in range(self.inner_steps):
            logits = self._call(params, sx)
            loss = F.cross_entropy(logits, sy)
            grads = autograd.grad(loss, list(params.values()),
                                  create_graph=create_graph)
            params = {n: p - alpha[n] * g
                      for (n, p), g in zip(params.items(), grads)}
        return params

    def client_update(self, sampler_fn, client_id):
        local = self._clone_params()
        local_alpha = {n: a.detach().clone().requires_grad_(True)
                       for n, a in self.alpha.items()}
        opt = torch.optim.Adam([
            {'params': list(local.values()), 'lr': self.outer_lr,
             'weight_decay': self.weight_decay},
            {'params': list(local_alpha.values()), 'lr': self.alpha_lr,
             'weight_decay': 0.0},
        ])
        old = {n: p.detach().clone()
               for n, p in self.model.named_parameters()}
        old_alpha = {n: a.clone() for n, a in self.alpha.items()}

        total_loss, total_acc, n = 0.0, 0.0, 0
        for _ in range(self.local_meta_steps):
            sx, sy, qx, qy = sampler_fn(client_id)
            sx, sy = sx.to(self.device), sy.to(self.device)
            qx, qy = qx.to(self.device), qy.to(self.device)

            opt.zero_grad()
            adapted = self.inner_loop(sx, sy, local,
                                      create_graph=not self.first_order,
                                      alpha=local_alpha)
            q_logits = self._call(adapted, qx)
            q_loss = F.cross_entropy(q_logits, qy)
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(local.values()), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(list(local_alpha.values()),
                                           self.grad_clip)
            opt.step()
            with torch.no_grad():
                for a in local_alpha.values():
                    a.clamp_(min=1e-6, max=1.0)
                acc = (q_logits.argmax(1) == qy).float().mean().item()
            total_loss += q_loss.item(); total_acc += acc; n += 1

        delta = {nm: local[nm].detach() - old[nm] for nm in local}
        alpha_delta = {nm: local_alpha[nm].detach() - old_alpha[nm]
                       for nm in local_alpha}
        return {'delta': delta, 'alpha_delta': alpha_delta,
                'loss': total_loss / n if n else 0.0,
                'accuracy': total_acc / n if n else 0.0,
                'steps': n}

    def aggregate(self, updates, weights):
        total = sum(weights)
        weights = [w / total for w in weights]
        with torch.no_grad():
            for nm, p in self.model.named_parameters():
                agg = torch.zeros_like(p)
                for u, w in zip(updates, weights):
                    agg += w * u['delta'][nm]
                p.data += agg
            for nm in self.alpha:
                agg = torch.zeros_like(self.alpha[nm])
                for u, w in zip(updates, weights):
                    agg += w * u['alpha_delta'][nm]
                self.alpha[nm] = self.alpha[nm] + agg
                self.alpha[nm].clamp_(min=1e-6, max=1.0)
