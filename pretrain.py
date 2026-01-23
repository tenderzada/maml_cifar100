"""
预训练脚本

在meta-train的64个类上进行标准分类预训练
用于Transfer Learning baseline对比
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from datetime import datetime

from models.conv4 import Conv4
from models.resnet import ResNet12


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


class CIFAR100MetaTrain(Dataset):
    """
    CIFAR-100 Meta-Train数据集
    只包含用于meta-training的64个类
    """

    def __init__(self, root, train=True, transform=None, download=True):
        self.transform = transform

        # 加载完整CIFAR-100
        cifar100 = datasets.CIFAR100(root=root, train=train,
                                      download=download, transform=None)

        # 类别划分 (与few-shot相同的划分)
        all_classes = list(range(100))
        random.seed(42)
        random.shuffle(all_classes)
        meta_train_classes = set(all_classes[:64])

        # 类别重映射
        self.class_map = {c: i for i, c in enumerate(sorted(meta_train_classes))}

        # 筛选数据
        self.data = []
        self.targets = []
        for img, label in zip(cifar100.data, cifar100.targets):
            if label in meta_train_classes:
                self.data.append(img)
                self.targets.append(self.class_map[label])

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target


def get_pretrain_loaders(data_root, batch_size=128, num_workers=4):
    """获取预训练数据加载器"""

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                           std=[0.2675, 0.2565, 0.2761])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                           std=[0.2675, 0.2565, 0.2761])
    ])

    train_dataset = CIFAR100MetaTrain(data_root, train=True, transform=train_transform)
    test_dataset = CIFAR100MetaTrain(data_root, train=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    return total_loss / len(train_loader), correct / total


def evaluate(model, test_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return total_loss / len(test_loader), correct / total


def main():
    parser = argparse.ArgumentParser(description='Pretrain on CIFAR-100 Meta-Train Classes')

    parser.add_argument('--data_root', type=str, default='./cifar100_data')
    parser.add_argument('--model', type=str, default='conv4',
                        choices=['conv4', 'resnet12'])
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Conv4隐藏层维度')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # 数据加载
    print("Loading data...")
    train_loader, test_loader = get_pretrain_loaders(
        args.data_root, args.batch_size, args.num_workers
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # 创建模型 (64类分类)
    if args.model == 'conv4':
        model = Conv4(hidden_dim=args.hidden_dim, n_way=64)
    else:
        model = ResNet12(n_way=64)

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, Parameters: {num_params:,}")

    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 80], gamma=0.1
    )

    # 训练
    exp_name = f"pretrain_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    best_acc = 0

    print(f"\n{'='*60}")
    print(f"Pretraining {args.model} on 64 meta-train classes")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train: {train_loss:.4f}/{train_acc:.4f}, "
              f"Test: {test_loss:.4f}/{test_acc:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(args.save_dir, f"{exp_name}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': vars(args)
            }, save_path)

    print(f"\nPretraining complete! Best test accuracy: {best_acc:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == '__main__':
    main()
