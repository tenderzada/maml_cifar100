"""
预训练 ResNet50 在 CIFAR-100 的剩余 80 类上 (联邦实验用 20 类, 剩余 80 类作预训练)

标准 few-shot / meta-learning 协议: base classes 预训练 -> meta-train / 评测在 novel classes
联邦实验里 select_classes(20, seed=42) 选出的 20 类不参与预训练, 避免数据泄漏。

用法 (GPU 服务器):
    python pretrain_cifar_meta.py --data_root /mnt/data/lev_data --epochs 100

输出 (默认):
    results/pretrain_resnet50.pt   (含 state_dict, pretrain_classes, fed_classes)

随后联邦训练:
    python train_cifar_fed_compare.py \\
        --meta_backbone resnet50 \\
        --pretrained_meta results/pretrain_resnet50.pt
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from data.cifar100_federated import select_classes, CIFAR_MEAN, CIFAR_STD
from models.resnet_cifar import resnet50_cifar


class CIFAR100Subset(Dataset):
    def __init__(self, root, classes, train, transform, download=False):
        ds = datasets.CIFAR100(root, train=train, download=download)
        targets = np.array(ds.targets)
        mask = np.isin(targets, classes)
        self.images = ds.data[mask]
        cls_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        self.labels = np.array([cls_to_idx[int(t)] for t in targets[mask]],
                               dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(self.images[i]), int(self.labels[i])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default='/mnt/data/lev_data')
    p.add_argument('--download', action='store_true', default=False)
    p.add_argument('--save_dir', type=str, default='./results')
    p.add_argument('--save_path', type=str, default=None,
                   help='默认 <save_dir>/pretrain_resnet50.pt')
    p.add_argument('--fed_num_classes', type=int, default=20,
                   help='联邦实验用的类别数; 这里要从总类中扣除')
    p.add_argument('--fed_seed', type=int, default=42,
                   help='select_classes 的 seed, 必须与联邦训练脚本一致')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = args.save_path or os.path.join(args.save_dir,
                                                'pretrain_resnet50.pt')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    fed_classes = select_classes(args.fed_num_classes, seed=args.fed_seed)
    pretrain_classes = sorted(set(range(100)) - set(fed_classes))
    print(f"fed (excluded) classes [{len(fed_classes)}]: {fed_classes}")
    print(f"pretrain classes [{len(pretrain_classes)}]: head={pretrain_classes[:10]}...")

    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_ds = CIFAR100Subset(args.data_root, pretrain_classes, train=True,
                              transform=train_tf, download=args.download)
    test_ds = CIFAR100Subset(args.data_root, pretrain_classes, train=False,
                             transform=eval_tf, download=args.download)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    print(f"train samples: {len(train_ds)}, test: {len(test_ds)}")

    model = resnet50_cifar(num_classes=len(pretrain_classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet50: {n_params/1e6:.2f}M params")

    opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += x.size(0)
        sched.step()
        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            t_correct, t_samples = 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    t_correct += (logits.argmax(1) == y).sum().item()
                    t_samples += x.size(0)
            test_acc = t_correct / t_samples
            cur_lr = opt.param_groups[0]['lr']
            print(f"epoch {epoch:3d}: lr={cur_lr:.4f} "
                  f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% "
                  f"test_acc={test_acc*100:.2f}%")
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'state_dict': model.state_dict(),
                    'pretrain_classes': pretrain_classes,
                    'fed_classes': fed_classes,
                    'epoch': epoch,
                    'test_acc': test_acc,
                    'args': vars(args),
                }, save_path)
                print(f"  -> saved best {test_acc*100:.2f}% to {save_path}")
        else:
            print(f"epoch {epoch:3d}: train_acc={train_acc*100:.2f}%")

    print(f"\nBest test acc: {best_acc*100:.2f}%  -> {save_path}")


if __name__ == '__main__':
    main()
