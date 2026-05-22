"""
CIFAR-100 联邦学习三方法对比实验

对比 FedAvg / FedAvg+MAML (Per-FedAvg) / FedAvg+Meta-SGD:
- 固定 num_classes 类子集, IID 划分到 num_clients 个客户端 (小数据场景)
- 统一 Per-FedAvg 评测协议 (adapt-then-eval)
- 每轮记录测试 loss / accuracy, 产出两幅对比图

用法 (GPU 服务器完整实验):
    python train_cifar_fed_compare.py --rounds 100

本机 CPU 快速验证全链路:
    python train_cifar_fed_compare.py --smoke
"""

import os
import json
import random
import argparse
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from data.cifar100_federated import FederatedCIFAR100
from models.resnet import ResNet12, ResNet12Functional
from models.federated import FedAvg, FedPerMAML, FedPerMetaSGD
from utils.visualization import plot_fed_comparison


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr(base_lr, step, total, final_ratio=0.02):
    """余弦退火至 base_lr*final_ratio; step 从 1 到 total"""
    if total <= 1:
        return base_lr
    lr_min = base_lr * final_ratio
    progress = (step - 1) / (total - 1)
    return lr_min + 0.5 * (base_lr - lr_min) * (1 + np.cos(np.pi * progress))


def eval_fedavg_adapt(global_model, episodes, inner_steps, inner_lr, device):
    """FedAvg 的 adapt-then-eval: 克隆全局模型, 在 support 上 SGD 微调, 评 query"""
    losses, accs = [], []
    for sx, sy, qx, qy in episodes:
        model = deepcopy(global_model).to(device)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=inner_lr, momentum=0.9)
        sx_d, sy_d = sx.to(device), sy.to(device)
        for _ in range(inner_steps):
            opt.zero_grad()
            loss = F.cross_entropy(model(sx_d), sy_d)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(qx.to(device))
            qy_d = qy.to(device)
            losses.append(F.cross_entropy(logits, qy_d).item())
            accs.append((logits.argmax(1) == qy_d).float().mean().item())
    return float(np.mean(losses)), float(np.mean(accs)), float(np.std(accs))


def run_fedavg(args, fed, episodes, device):
    set_seed(args.seed)
    print('\n' + '=' * 60 + '\n[1/3] FedAvg\n' + '=' * 60)
    model = ResNet12(in_channels=3, channels=args.channels,
                     n_way=args.num_classes, drop_rate=args.drop_rate)
    algo = FedAvg(global_model=model, num_clients=args.num_clients,
                  clients_per_round=args.clients_per_round,
                  local_epochs=args.local_epochs, local_lr=args.local_lr,
                  weight_decay=args.weight_decay, device=device)
    loaders = fed.get_client_loaders()
    hist = {'rounds': [], 'loss': [], 'acc': [], 'acc_std': []}
    for r in range(1, args.rounds + 1):
        algo.local_lr = cosine_lr(args.local_lr, r, args.rounds, args.lr_final_ratio)
        algo.train_round(loaders)
        if r % args.eval_every == 0 or r == args.rounds:
            loss, acc, std = eval_fedavg_adapt(
                algo.global_model, episodes, args.inner_steps, args.inner_lr, device)
            hist['rounds'].append(r); hist['loss'].append(loss)
            hist['acc'].append(acc); hist['acc_std'].append(std)
            print(f"  round {r:3d} | test loss {loss:.4f} | acc {acc*100:.2f}% ± {std*100:.2f}%")
    return hist


def run_meta(args, fed, episodes, device, method):
    set_seed(args.seed)
    tag = {'maml': '[2/3] FedAvg+MAML', 'metasgd': '[3/3] FedAvg+Meta-SGD'}[method]
    print('\n' + '=' * 60 + f'\n{tag}\n' + '=' * 60)
    model = ResNet12Functional(in_channels=3, channels=args.channels,
                               n_way=args.num_classes, drop_rate=args.drop_rate)
    common = dict(num_clients=args.num_clients,
                  clients_per_round=args.clients_per_round,
                  inner_lr=args.inner_lr, inner_steps=args.inner_steps,
                  outer_lr=args.outer_lr, local_meta_steps=args.local_meta_steps,
                  weight_decay=args.weight_decay, first_order=args.first_order,
                  device=device)
    if method == 'maml':
        algo = FedPerMAML(model, **common)
    else:
        algo = FedPerMetaSGD(model, alpha_init=args.inner_lr,
                             alpha_lr=args.alpha_lr, **common)

    def sampler_fn(cid):
        return fed.sample_meta_batch(cid, k_support=args.k_support,
                                     k_query=args.k_query, augment=True)

    hist = {'rounds': [], 'loss': [], 'acc': [], 'acc_std': []}
    for r in range(1, args.rounds + 1):
        algo.outer_lr = cosine_lr(args.outer_lr, r, args.rounds, args.lr_final_ratio)
        if method == 'metasgd':
            algo.alpha_lr = cosine_lr(args.alpha_lr, r, args.rounds, args.lr_final_ratio)
        algo.train_round(sampler_fn)
        if r % args.eval_every == 0 or r == args.rounds:
            loss, acc, std = algo.evaluate_episodes(episodes)
            hist['rounds'].append(r); hist['loss'].append(loss)
            hist['acc'].append(acc); hist['acc_std'].append(std)
            print(f"  round {r:3d} | test loss {loss:.4f} | acc {acc*100:.2f}% ± {std*100:.2f}%")
    return hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default='/mnt/data/lev_data')
    p.add_argument('--download', action='store_true', default=False,
                   help='数据不存在时才下载; 服务器已有数据请勿开启')
    p.add_argument('--save_dir', type=str, default='./results')
    p.add_argument('--num_classes', type=int, default=20)
    p.add_argument('--num_clients', type=int, default=10)
    p.add_argument('--clients_per_round', type=int, default=5)
    p.add_argument('--samples_per_client', type=int, default=500)
    p.add_argument('--rounds', type=int, default=150)
    p.add_argument('--eval_every', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr_final_ratio', type=float, default=0.02,
                   help='余弦退火终点 = 初始lr * 该比例')
    # FedAvg 本地
    p.add_argument('--local_epochs', type=int, default=3)
    p.add_argument('--local_lr', type=float, default=0.05)
    # 元学习
    p.add_argument('--local_meta_steps', type=int, default=5)
    p.add_argument('--inner_lr', type=float, default=0.01)
    p.add_argument('--inner_steps', type=int, default=5)
    p.add_argument('--outer_lr', type=float, default=0.001)
    p.add_argument('--alpha_lr', type=float, default=0.001)
    p.add_argument('--k_support', type=int, default=3)
    p.add_argument('--k_query', type=int, default=3)
    p.add_argument('--first_order', action='store_true', default=True)
    p.add_argument('--second_order', dest='first_order', action='store_false')
    # 评测
    p.add_argument('--k_shot_eval', type=int, default=5)
    p.add_argument('--query_per_class', type=int, default=30)
    p.add_argument('--n_eval_episodes', type=int, default=5)
    # 模型/正则
    p.add_argument('--channels', type=int, nargs='+', default=[64, 128, 256, 512])
    p.add_argument('--drop_rate', type=float, default=0.1)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke', action='store_true', help='本机CPU小规模全链路验证')
    args = p.parse_args()

    if args.smoke:
        args.num_classes = 5; args.num_clients = 3; args.clients_per_round = 2
        args.samples_per_client = 60; args.rounds = 3; args.eval_every = 1
        args.local_epochs = 1; args.local_meta_steps = 2
        args.k_shot_eval = 2; args.query_per_class = 8; args.n_eval_episodes = 2
        args.k_support = 2; args.k_query = 2
        print('>>> SMOKE MODE: 小规模 CPU 验证')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    fed = FederatedCIFAR100(
        root=args.data_root, num_classes=args.num_classes,
        num_clients=args.num_clients, samples_per_client=args.samples_per_client,
        batch_size=args.batch_size, k_shot_eval=args.k_shot_eval,
        query_per_class=args.query_per_class, n_eval_episodes=args.n_eval_episodes,
        augment=True, seed=args.seed, download=args.download)
    print(f"Selected classes ({args.num_classes}): {fed.classes}")
    episodes = fed.get_eval_episodes()

    history = {}
    history['FedAvg'] = run_fedavg(args, fed, episodes, device)
    history['FedAvg+MAML'] = run_meta(args, fed, episodes, device, 'maml')
    history['FedAvg+Meta-SGD'] = run_meta(args, fed, episodes, device, 'metasgd')

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f"cifar_fed_compare_{args.num_classes}cls_{ts}"
    json_path = os.path.join(args.save_dir, f"{prefix}.json")
    with open(json_path, 'w') as f:
        json.dump({'args': vars(args), 'history': history}, f, indent=2)
    print(f"\nHistory saved to: {json_path}")

    plot_fed_comparison(history, args.save_dir, prefix)

    print('\n' + '=' * 60 + '\nFinal (last round) test accuracy:')
    for m, h in history.items():
        print(f"  {m:18s}: {h['acc'][-1]*100:.2f}% ± {h['acc_std'][-1]*100:.2f}%")
    print('=' * 60)


if __name__ == '__main__':
    main()
