"""
CIFAR-100 固定类别联邦学习数据模块

用于 FedAvg / FedAvg+MAML / FedAvg+Meta-SGD 三方法对比:
- 从 CIFAR-100 固定取 num_classes 个类 (绝对标签 0..num_classes-1)
- 训练数据 IID 均分到 num_clients 个客户端, 每客户端数据量受限 (small-data 场景)
- 提供:
    * 每客户端 DataLoader (FedAvg 标准本地训练)
    * 元学习的 (support, query) 批采样 (FedPerMAML / FedPerMetaSGD 本地更新)
    * adapt-then-eval 评测 episode (Per-FedAvg 协议: support 适应 -> query 评测)
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from collections import defaultdict


CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR_STD = [0.2675, 0.2565, 0.2761]


def select_classes(num_classes=20, seed=42):
    """固定选取 num_classes 个 CIFAR-100 类别 (可复现)"""
    classes = list(range(100))
    rng = random.Random(seed)
    rng.shuffle(classes)
    return sorted(classes[:num_classes])


def _train_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])


def _eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])


class _ArrayDataset(Dataset):
    """保存 uint8 HWC 数组 + 标签, 访问时应用 transform"""

    def __init__(self, images, labels, transform):
        self.images = images          # np.uint8 [N, 32, 32, 3]
        self.labels = labels          # np.int64 [N]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.transform(self.images[idx])
        return img, int(self.labels[idx])


class FederatedCIFAR100:
    """固定类别联邦 CIFAR-100 数据容器"""

    def __init__(self, root, num_classes=20, num_clients=10,
                 samples_per_client=250, batch_size=32,
                 k_shot_eval=5, query_per_class=30, n_eval_episodes=5,
                 augment=True, seed=42, download=True):
        self.num_classes = num_classes
        self.num_clients = num_clients
        self.samples_per_client = samples_per_client
        self.batch_size = batch_size
        self.k_shot_eval = k_shot_eval
        self.query_per_class = query_per_class
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed

        self.train_tf = _train_transform() if augment else _eval_transform()
        self.eval_tf = _eval_transform()

        self.classes = select_classes(num_classes, seed)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # 加载 CIFAR-100 train / test
        train_set = datasets.CIFAR100(root=root, train=True, download=download)
        test_set = datasets.CIFAR100(root=root, train=False, download=download)

        train_x, train_y = self._filter(train_set)
        test_x, test_y = self._filter(test_set)

        self.test_x, self.test_y = test_x, test_y

        # 按类组织 test (用于 eval episode 采样)
        self.test_by_class = defaultdict(list)
        for i, y in enumerate(test_y):
            self.test_by_class[int(y)].append(i)

        # IID 划分训练数据到客户端
        self._partition_iid(train_x, train_y)

        # 固定一组 eval episodes (跨方法/跨轮次复用, 保证可比)
        self._eval_episodes = None

    def _filter(self, dataset):
        """筛出选中类别的样本, 标签重映射为 0..num_classes-1"""
        data = dataset.data                       # np.uint8 [N, 32, 32, 3]
        targets = np.array(dataset.targets)
        mask = np.isin(targets, self.classes)
        x = data[mask]
        y = np.array([self.class_to_idx[int(t)] for t in targets[mask]], dtype=np.int64)
        return x, y

    def _partition_iid(self, train_x, train_y):
        """IID 均分到客户端, 每客户端最多 samples_per_client 张"""
        rng = np.random.RandomState(self.seed)
        idx = np.arange(len(train_x))
        rng.shuffle(idx)

        per_client = self.samples_per_client
        self.client_x = []
        self.client_y = []
        self.client_by_class = []  # 每客户端 {label: [local_idx]}

        cursor = 0
        for _ in range(self.num_clients):
            sel = idx[cursor:cursor + per_client]
            cursor += per_client
            cx = train_x[sel]
            cy = train_y[sel]
            self.client_x.append(cx)
            self.client_y.append(cy)
            by_cls = defaultdict(list)
            for li, yy in enumerate(cy):
                by_cls[int(yy)].append(li)
            self.client_by_class.append(by_cls)

    # ------------------------------------------------------------------
    # FedAvg: 标准本地训练 DataLoader
    # ------------------------------------------------------------------
    def get_client_loaders(self, num_workers=0):
        loaders = []
        for cid in range(self.num_clients):
            ds = _ArrayDataset(self.client_x[cid], self.client_y[cid], self.train_tf)
            loaders.append(DataLoader(
                ds, batch_size=self.batch_size, shuffle=True,
                num_workers=num_workers, drop_last=False
            ))
        return loaders

    def get_test_loader(self, num_workers=0):
        ds = _ArrayDataset(self.test_x, self.test_y, self.eval_tf)
        return DataLoader(ds, batch_size=128, shuffle=False, num_workers=num_workers)

    # ------------------------------------------------------------------
    # 元学习本地更新: 从某客户端采样 (support, query) 批
    # ------------------------------------------------------------------
    def _apply(self, transform, images):
        return torch.stack([transform(img) for img in images])

    def sample_meta_batch(self, client_id, k_support=3, k_query=3, augment=True):
        """
        从客户端本地数据采样 support / query (类均衡)

        Returns: support_x, support_y, query_x, query_y  (CPU tensors)
        """
        tf = self.train_tf if augment else self.eval_tf
        by_cls = self.client_by_class[client_id]
        cx = self.client_x[client_id]

        s_imgs, s_lbls, q_imgs, q_lbls = [], [], [], []
        for cls in range(self.num_classes):
            pool = by_cls.get(cls, [])
            if len(pool) == 0:
                continue
            need = k_support + k_query
            if len(pool) >= need:
                chosen = random.sample(pool, need)
            else:
                chosen = [random.choice(pool) for _ in range(need)]
            s_idx = chosen[:k_support]
            q_idx = chosen[k_support:]
            for li in s_idx:
                s_imgs.append(cx[li]); s_lbls.append(cls)
            for li in q_idx:
                q_imgs.append(cx[li]); q_lbls.append(cls)

        support_x = self._apply(tf, s_imgs)
        support_y = torch.tensor(s_lbls, dtype=torch.long)
        query_x = self._apply(tf, q_imgs)
        query_y = torch.tensor(q_lbls, dtype=torch.long)
        return support_x, support_y, query_x, query_y

    # ------------------------------------------------------------------
    # adapt-then-eval 评测 episode (固定复用)
    # ------------------------------------------------------------------
    def get_eval_episodes(self):
        if self._eval_episodes is not None:
            return self._eval_episodes

        rng = random.Random(self.seed + 12345)
        episodes = []
        for _ in range(self.n_eval_episodes):
            s_imgs, s_lbls, q_imgs, q_lbls = [], [], [], []
            for cls in range(self.num_classes):
                pool = list(self.test_by_class[cls])
                rng.shuffle(pool)
                s_idx = pool[:self.k_shot_eval]
                q_idx = pool[self.k_shot_eval:self.k_shot_eval + self.query_per_class]
                for i in s_idx:
                    s_imgs.append(self.test_x[i]); s_lbls.append(cls)
                for i in q_idx:
                    q_imgs.append(self.test_x[i]); q_lbls.append(cls)
            support_x = self._apply(self.eval_tf, s_imgs)
            support_y = torch.tensor(s_lbls, dtype=torch.long)
            query_x = self._apply(self.eval_tf, q_imgs)
            query_y = torch.tensor(q_lbls, dtype=torch.long)
            episodes.append((support_x, support_y, query_x, query_y))

        self._eval_episodes = episodes
        return episodes


if __name__ == '__main__':
    fed = FederatedCIFAR100(
        root='./cifar100_data', num_classes=5, num_clients=3,
        samples_per_client=60, k_shot_eval=2, query_per_class=10,
        n_eval_episodes=2, download=True
    )
    print('classes:', fed.classes)
    loaders = fed.get_client_loaders()
    print('num client loaders:', len(loaders))
    sx, sy, qx, qy = fed.sample_meta_batch(0, k_support=2, k_query=2)
    print('meta batch:', sx.shape, sy.shape, qx.shape, qy.shape)
    eps = fed.get_eval_episodes()
    print('eval episodes:', len(eps), 'support', eps[0][0].shape, 'query', eps[0][2].shape)
