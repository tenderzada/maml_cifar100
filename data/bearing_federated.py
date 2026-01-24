"""
轴承故障诊断数据集 - 联邦学习版本

支持两种数据分布:
1. IID (独立同分布): 数据随机均匀分配给各客户端
2. Non-IID (非独立同分布): 每个客户端只有部分类别的数据

用于 FedAvg 和 FedAvg+MAML 实验
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from data.bearing_fewshot import RandomJitter, get_strong_augmentation


DEFAULT_DATA_FILE = '/mnt/data/scutfd/bearing_data.pkl'


class BearingFederatedDataset(Dataset):
    """
    联邦学习的单客户端数据集
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray, transform=None):
        """
        Args:
            data: 数据 [N, window_size, channels]
            labels: 标签 [N]
            transform: 数据变换
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # [window_size, channels] -> [channels, window_size]
        x = self.data[idx].T.astype(np.float32)
        y = int(self.labels[idx])

        x = torch.from_numpy(x)
        if self.transform:
            x = self.transform(x)

        return x, y


class BearingFederatedFewShot(Dataset):
    """
    联邦Few-Shot数据集 (用于FedAvg+MAML)
    每个客户端返回episode格式的数据
    """

    def __init__(
        self,
        data_by_class: Dict[int, List],
        classes: List[int],
        n_way: int = 5,
        k_shot: int = 1,
        k_query: int = 15,
        num_episodes: int = 100,
        transform=None
    ):
        self.data_by_class = data_by_class
        self.classes = classes
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.num_episodes = num_episodes
        self.transform = transform

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        # 从客户端可用的类别中随机选择n_way个
        if len(self.classes) < self.n_way:
            # 如果类别不足，允许重复选择
            selected_classes = random.choices(self.classes, k=self.n_way)
        else:
            selected_classes = random.sample(self.classes, self.n_way)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for i, cls in enumerate(selected_classes):
            class_samples = self.data_by_class[cls]
            n_samples = len(class_samples)

            # 确保有足够的样本
            if n_samples < self.k_shot + self.k_query:
                indices = random.choices(range(n_samples), k=self.k_shot + self.k_query)
            else:
                indices = random.sample(range(n_samples), self.k_shot + self.k_query)

            samples = [class_samples[j] for j in indices]

            for sample in samples[:self.k_shot]:
                data = sample.T.astype(np.float32)
                data = torch.from_numpy(data)
                if self.transform:
                    data = self.transform(data)
                support_x.append(data)
                support_y.append(i)

            for sample in samples[self.k_shot:self.k_shot + self.k_query]:
                data = sample.T.astype(np.float32)
                data = torch.from_numpy(data)
                if self.transform:
                    data = self.transform(data)
                query_x.append(data)
                query_y.append(i)

        support_x = torch.stack(support_x)
        support_y = torch.tensor(support_y)
        query_x = torch.stack(query_x)
        query_y = torch.tensor(query_y)

        return support_x, support_y, query_x, query_y


class FederatedBearingData:
    """
    联邦学习数据管理器

    将轴承数据划分给多个客户端，支持IID和Non-IID分布
    """

    def __init__(
        self,
        data_file: str = DEFAULT_DATA_FILE,
        num_clients: int = 10,
        iid: bool = True,
        non_iid_classes_per_client: int = 6,  # Non-IID时每个客户端的类别数
        seed: int = 42
    ):
        """
        Args:
            data_file: 数据文件路径
            num_clients: 客户端数量
            iid: 是否使用IID分布
            non_iid_classes_per_client: Non-IID时每个客户端分配的类别数
            seed: 随机种子
        """
        self.num_clients = num_clients
        self.iid = iid
        self.non_iid_classes_per_client = non_iid_classes_per_client

        random.seed(seed)
        np.random.seed(seed)

        # 加载数据
        data_path = Path(data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件未找到: {data_file}")

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.num_classes = data['num_classes']
        self.idx_to_class = data['idx_to_class']

        # 合并训练和测试数据
        train_data = data['train_data']
        train_labels = data['train_labels']
        test_data = data['test_data']
        test_labels = data['test_labels']

        all_data = np.concatenate([train_data, test_data], axis=0)
        all_labels = np.concatenate([train_labels, test_labels], axis=0)

        # 按类别组织数据
        self.data_by_class = defaultdict(list)
        for i in range(len(all_data)):
            label = int(all_labels[i])
            self.data_by_class[label].append(all_data[i])

        # 类别划分 (与few-shot保持一致)
        all_classes = list(range(self.num_classes))
        random.seed(42)
        random.shuffle(all_classes)

        self.train_classes = all_classes[:40]
        self.val_classes = all_classes[40:52]
        self.test_classes = all_classes[52:64]

        # 创建标签映射 (原始类别索引 -> 0, 1, 2, ..., n-1)
        self.train_label_map = {cls: i for i, cls in enumerate(self.train_classes)}
        self.test_label_map = {cls: i for i, cls in enumerate(self.test_classes)}

        # 分离训练和测试数据 (训练类别中80%用于训练，20%用于测试)
        self.client_data_by_class = defaultdict(list)  # 用于客户端训练
        self.test_data_by_class = defaultdict(list)    # 用于全局测试

        for cls in self.train_classes:
            samples = self.data_by_class[cls].copy()
            random.shuffle(samples)
            n_test = max(1, len(samples) // 5)  # 20%用于测试
            self.test_data_by_class[cls] = samples[:n_test]
            self.client_data_by_class[cls] = samples[n_test:]

        # 恢复随机种子
        random.seed(seed)

        # 划分客户端数据
        self.client_data = self._split_data_to_clients()

        print(f"Federated Bearing Data: {num_clients} clients, "
              f"{'IID' if iid else 'Non-IID'}")
        self._print_client_stats()

    def _split_data_to_clients(self) -> List[Dict]:
        """
        将数据划分给各客户端

        Returns:
            client_data: 每个客户端的数据字典
        """
        client_data = []

        if self.iid:
            # IID分布: 随机均匀分配
            client_data = self._split_iid()
        else:
            # Non-IID分布: 每个客户端只有部分类别
            client_data = self._split_non_iid()

        return client_data

    def _split_iid(self) -> List[Dict]:
        """
        IID数据划分: 每个类别的数据随机均匀分配给各客户端
        """
        client_data = [{'data_by_class': defaultdict(list), 'classes': set()}
                       for _ in range(self.num_clients)]

        for cls in self.train_classes:
            samples = self.client_data_by_class[cls].copy()
            random.shuffle(samples)

            # 均匀分配给各客户端
            samples_per_client = len(samples) // self.num_clients
            for i in range(self.num_clients):
                start = i * samples_per_client
                end = start + samples_per_client if i < self.num_clients - 1 else len(samples)
                client_samples = samples[start:end]

                if client_samples:
                    client_data[i]['data_by_class'][cls].extend(client_samples)
                    client_data[i]['classes'].add(cls)

        # 转换classes为list
        for i in range(self.num_clients):
            client_data[i]['classes'] = list(client_data[i]['classes'])

        return client_data

    def _split_non_iid(self) -> List[Dict]:
        """
        Non-IID数据划分: 每个客户端只分配部分类别的数据

        使用Dirichlet分布或简单的类别分配
        """
        client_data = [{'data_by_class': defaultdict(list), 'classes': []}
                       for _ in range(self.num_clients)]

        # 将类别分配给客户端 (每个客户端获得non_iid_classes_per_client个类别)
        # 类别可以在客户端之间重叠
        classes_per_client = []
        available_classes = self.train_classes.copy()

        for i in range(self.num_clients):
            if len(available_classes) >= self.non_iid_classes_per_client:
                selected = random.sample(available_classes, self.non_iid_classes_per_client)
            else:
                # 如果可用类别不足，从所有训练类别中选择
                selected = random.sample(self.train_classes, self.non_iid_classes_per_client)
            classes_per_client.append(selected)

        # 分配数据
        for i, client_classes in enumerate(classes_per_client):
            client_data[i]['classes'] = client_classes

            for cls in client_classes:
                samples = self.client_data_by_class[cls].copy()

                # 每个客户端获得该类别的一部分数据
                # 计算有多少客户端共享这个类别
                clients_with_class = sum(1 for cc in classes_per_client if cls in cc)
                samples_for_client = len(samples) // clients_with_class

                # 根据客户端ID确定获取哪部分数据
                client_idx_for_class = sum(1 for j in range(i) if cls in classes_per_client[j])
                start = client_idx_for_class * samples_for_client
                end = start + samples_for_client

                client_samples = samples[start:end]
                if client_samples:
                    client_data[i]['data_by_class'][cls].extend(client_samples)

        return client_data

    def _print_client_stats(self):
        """打印客户端数据统计"""
        print("\nClient data distribution:")
        for i, cdata in enumerate(self.client_data):
            n_samples = sum(len(v) for v in cdata['data_by_class'].values())
            n_classes = len(cdata['classes'])
            print(f"  Client {i}: {n_samples} samples, {n_classes} classes")

    def get_client_dataloader(
        self,
        client_id: int,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        transform=None,
        strong_augment: bool = False
    ) -> DataLoader:
        """
        获取指定客户端的DataLoader (用于FedAvg)

        Args:
            strong_augment: 是否使用强数据增强
        """
        cdata = self.client_data[client_id]

        # 设置数据增强
        if transform is None and shuffle:  # 训练时
            if strong_augment:
                transform = get_strong_augmentation()
            else:
                transform = RandomJitter(sigma=0.03)

        # 合并所有样本 (使用映射后的标签)
        all_samples = []
        all_labels = []
        for cls, samples in cdata['data_by_class'].items():
            all_samples.extend(samples)
            # 使用映射后的标签 (0, 1, 2, ..., n_classes-1)
            mapped_label = self.train_label_map[cls]
            all_labels.extend([mapped_label] * len(samples))

        dataset = BearingFederatedDataset(
            np.array(all_samples),
            np.array(all_labels),
            transform=transform
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

    def get_client_fewshot_loader(
        self,
        client_id: int,
        n_way: int = 5,
        k_shot: int = 1,
        k_query: int = 15,
        num_episodes: int = 100,
        num_workers: int = 0,
        transform=None,
        strong_augment: bool = False
    ) -> DataLoader:
        """
        获取指定客户端的Few-Shot DataLoader (用于FedAvg+MAML)

        Args:
            strong_augment: 是否使用强数据增强
        """
        cdata = self.client_data[client_id]

        # 设置数据增强
        if transform is None:
            if strong_augment:
                transform = get_strong_augmentation()
            else:
                transform = RandomJitter(sigma=0.03)

        dataset = BearingFederatedFewShot(
            data_by_class=cdata['data_by_class'],
            classes=cdata['classes'],
            n_way=n_way,
            k_shot=k_shot,
            k_query=k_query,
            num_episodes=num_episodes,
            transform=transform
        )

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

    def get_global_test_loader(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        transform=None
    ) -> DataLoader:
        """
        获取全局测试集DataLoader (用于FedAvg评估)

        使用预留的测试数据 (训练类别的20%)
        """
        all_samples = []
        all_labels = []

        for cls in self.train_classes:
            samples = self.test_data_by_class[cls]
            all_samples.extend(samples)
            # 使用映射后的标签
            mapped_label = self.train_label_map[cls]
            all_labels.extend([mapped_label] * len(samples))

        dataset = BearingFederatedDataset(
            np.array(all_samples),
            np.array(all_labels),
            transform=transform
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    def get_global_fewshot_loader(
        self,
        mode: str = 'test',
        n_way: int = 5,
        k_shot: int = 1,
        k_query: int = 15,
        num_episodes: int = 600,
        num_workers: int = 0,
        transform=None
    ) -> DataLoader:
        """
        获取全局Few-Shot测试集 (用于FedAvg+MAML评估)
        """
        if mode == 'val':
            classes = self.val_classes
        else:
            classes = self.test_classes

        dataset = BearingFederatedFewShot(
            data_by_class=self.data_by_class,
            classes=classes,
            n_way=n_way,
            k_shot=k_shot,
            k_query=k_query,
            num_episodes=num_episodes,
            transform=transform
        )

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )


if __name__ == '__main__':
    # 测试
    try:
        # IID分布
        print("=" * 50)
        print("Testing IID distribution")
        print("=" * 50)
        fed_data_iid = FederatedBearingData(
            num_clients=10,
            iid=True
        )

        # 获取客户端数据
        loader = fed_data_iid.get_client_dataloader(0, batch_size=32)
        for x, y in loader:
            print(f"Client 0 batch: x={x.shape}, y={y.shape}")
            break

        # Non-IID分布
        print("\n" + "=" * 50)
        print("Testing Non-IID distribution")
        print("=" * 50)
        fed_data_noniid = FederatedBearingData(
            num_clients=10,
            iid=False,
            non_iid_classes_per_client=6
        )

        # 获取Few-Shot数据
        loader = fed_data_noniid.get_client_fewshot_loader(
            0, n_way=5, k_shot=1, k_query=15, num_episodes=10
        )
        for support_x, support_y, query_x, query_y in loader:
            print(f"Client 0 episode: support={support_x.shape}, query={query_x.shape}")
            break

    except FileNotFoundError as e:
        print(e)
