"""
CIFAR-100 Few-Shot Learning Dataset
将CIFAR-100转换为N-way K-shot的episode格式
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from collections import defaultdict
import random


class CIFAR100FewShot(Dataset):
    """
    CIFAR-100 Few-Shot Dataset

    将CIFAR-100划分为:
    - meta-train: 64 classes
    - meta-val: 16 classes
    - meta-test: 20 classes
    """

    def __init__(self, root, mode='train', n_way=5, k_shot=1, k_query=15,
                 num_episodes=600, transform=None, download=True):
        """
        Args:
            root: CIFAR-100数据存放路径
            mode: 'train', 'val', 或 'test'
            n_way: N-way分类
            k_shot: support set中每类样本数
            k_query: query set中每类样本数
            num_episodes: 每个epoch的episode数量
            transform: 数据增强
            download: 是否下载数据
        """
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.num_episodes = num_episodes

        # 默认transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                )
            ])
        else:
            self.transform = transform

        # 加载CIFAR-100
        is_train = mode in ['train', 'val']
        cifar100 = datasets.CIFAR100(root=root, train=is_train,
                                      download=download, transform=None)

        # 按类别组织数据
        self.data_by_class = defaultdict(list)
        for img, label in zip(cifar100.data, cifar100.targets):
            self.data_by_class[label].append(img)

        # 类别划分 (固定划分以保证可复现性)
        all_classes = list(range(100))
        random.seed(42)
        random.shuffle(all_classes)

        if mode == 'train':
            self.classes = all_classes[:64]
        elif mode == 'val':
            self.classes = all_classes[64:80]
        else:  # test
            self.classes = all_classes[80:]

        # 重新映射类别标签
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        """
        返回一个episode

        Returns:
            support_x: [n_way * k_shot, C, H, W]
            support_y: [n_way * k_shot]
            query_x: [n_way * k_query, C, H, W]
            query_y: [n_way * k_query]
        """
        # 随机选择n_way个类
        selected_classes = random.sample(self.classes, self.n_way)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for i, cls in enumerate(selected_classes):
            # 从该类中随机选择k_shot + k_query个样本
            samples = random.sample(self.data_by_class[cls],
                                   self.k_shot + self.k_query)

            # 划分support和query
            support_samples = samples[:self.k_shot]
            query_samples = samples[self.k_shot:]

            for sample in support_samples:
                img = self.transform(sample)
                support_x.append(img)
                support_y.append(i)  # 使用episode内的相对标签

            for sample in query_samples:
                img = self.transform(sample)
                query_x.append(img)
                query_y.append(i)

        # 转换为tensor
        support_x = torch.stack(support_x)
        support_y = torch.tensor(support_y)
        query_x = torch.stack(query_x)
        query_y = torch.tensor(query_y)

        return support_x, support_y, query_x, query_y


def get_dataloader(root, mode, n_way, k_shot, k_query,
                   num_episodes, batch_size=1, num_workers=4):
    """
    获取few-shot dataloader

    Args:
        batch_size: episode的batch size（通常为1）
    """
    # 训练时使用数据增强
    if mode == 'train':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])

    dataset = CIFAR100FewShot(
        root=root,
        mode=mode,
        n_way=n_way,
        k_shot=k_shot,
        k_query=k_query,
        num_episodes=num_episodes,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


if __name__ == '__main__':
    # 测试数据加载
    loader = get_dataloader(
        root='./cifar100_data',
        mode='train',
        n_way=5,
        k_shot=1,
        k_query=15,
        num_episodes=100,
        batch_size=1
    )

    for support_x, support_y, query_x, query_y in loader:
        print(f"Support X shape: {support_x.shape}")  # [1, 5, C, H, W]
        print(f"Support Y shape: {support_y.shape}")  # [1, 5]
        print(f"Query X shape: {query_x.shape}")      # [1, 75, C, H, W]
        print(f"Query Y shape: {query_y.shape}")      # [1, 75]
        break
