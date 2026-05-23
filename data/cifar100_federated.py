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
                 iid=True, dirichlet_alpha=0.3,
                 augment=True, seed=42, download=True):
        self.num_classes = num_classes
        self.num_clients = num_clients
        self.samples_per_client = samples_per_client
        self.batch_size = batch_size
        self.k_shot_eval = k_shot_eval
        self.query_per_class = query_per_class
        self.n_eval_episodes = n_eval_episodes
        self.iid = iid
        self.dirichlet_alpha = dirichlet_alpha
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

        # 按类组织 test (IID 全局 eval episode 采样用)
        self.test_by_class = defaultdict(list)
        for i, y in enumerate(test_y):
            self.test_by_class[int(y)].append(i)

        # 客户端测试分区 (non-IID 个性化评测用)
        self.client_test_x = None
        self.client_test_y = None
        self.client_test_by_class = None

        # 划分训练数据到客户端
        if self.iid:
            self._partition_iid(train_x, train_y)
        else:
            self._partition_dirichlet(train_x, train_y, test_x, test_y)

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

    def _partition_dirichlet(self, train_x, train_y, test_x, test_y):
        """
        non-IID 标签偏斜划分: 每个类的样本按 Dirichlet(alpha) 比例分到各客户端

        train 与 test 用同一组 per-class 比例, 使每个客户端的训练/测试分布一致
        (个性化评测要求 test 与该客户端训练分布同分布)。
        alpha 越小越异构。
        """
        rng = np.random.RandomState(self.seed)
        nc = self.num_clients

        train_idx = [[] for _ in range(nc)]
        test_idx = [[] for _ in range(nc)]

        for cls in range(self.num_classes):
            props = rng.dirichlet([self.dirichlet_alpha] * nc)  # 长度 nc

            tr_c = np.where(train_y == cls)[0]
            rng.shuffle(tr_c)
            tr_split = (np.cumsum(props)[:-1] * len(tr_c)).astype(int)
            for cid, part in enumerate(np.split(tr_c, tr_split)):
                train_idx[cid].extend(part.tolist())

            te_c = np.where(test_y == cls)[0]
            rng.shuffle(te_c)
            te_split = (np.cumsum(props)[:-1] * len(te_c)).astype(int)
            for cid, part in enumerate(np.split(te_c, te_split)):
                test_idx[cid].extend(part.tolist())

        self.client_x, self.client_y, self.client_by_class = [], [], []
        self.client_test_x, self.client_test_y, self.client_test_by_class = [], [], []
        for cid in range(nc):
            tr = np.array(train_idx[cid], dtype=int)
            rng.shuffle(tr)
            cx, cy = train_x[tr], train_y[tr]
            self.client_x.append(cx); self.client_y.append(cy)
            by_cls = defaultdict(list)
            for li, yy in enumerate(cy):
                by_cls[int(yy)].append(li)
            self.client_by_class.append(by_cls)

            te = np.array(test_idx[cid], dtype=int)
            tex, tey = test_x[te], test_y[te]
            self.client_test_x.append(tex); self.client_test_y.append(tey)
            tby = defaultdict(list)
            for li, yy in enumerate(tey):
                tby[int(yy)].append(li)
            self.client_test_by_class.append(tby)

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

    def sample_meta_batch_random(self, client_id, n_support, n_query, augment=True):
        """
        non-IID 元采样: 直接从客户端本地数据随机采 (反映其真实分布)

        不强制类均衡 -- 这是 Per-FedAvg 在异构客户端上的标准做法。
        样本不足时启用替换采样, 但因为按总数采且分布与客户端一致, 不会引发
        与"强制 20 类"相同的退化。
        """
        tf = self.train_tf if augment else self.eval_tf
        cx = self.client_x[client_id]
        n = len(cx)
        if n == 0:
            raise ValueError(f"client {client_id} has no local samples")
        replace = n < (n_support + n_query)
        idx = np.random.choice(n, size=n_support + n_query, replace=replace)
        s_idx, q_idx = idx[:n_support], idx[n_support:]
        s_imgs = [cx[i] for i in s_idx]
        q_imgs = [cx[i] for i in q_idx]
        s_lbls = self.client_y[client_id][s_idx]
        q_lbls = self.client_y[client_id][q_idx]
        return (self._apply(tf, s_imgs),
                torch.tensor(s_lbls, dtype=torch.long),
                self._apply(tf, q_imgs),
                torch.tensor(q_lbls, dtype=torch.long))

    def report_partition(self):
        """打印客户端类分布统计 (用于核对 non-IID 偏斜程度)"""
        print("[partition] per-client (size, #classes, top-3 classes):")
        for cid in range(self.num_clients):
            sizes = [(c, len(idxs)) for c, idxs in self.client_by_class[cid].items()]
            sizes.sort(key=lambda x: -x[1])
            total = sum(s for _, s in sizes)
            top3 = ", ".join(f"{c}:{s}" for c, s in sizes[:3])
            print(f"  client {cid}: n={total:4d}, classes={len(sizes):2d}, top: {top3}")

    # ------------------------------------------------------------------
    # adapt-then-eval 评测 episode (固定复用)
    # ------------------------------------------------------------------
    def get_eval_episodes(self):
        """
        IID:     全局 episode (每类从全局测试集抽 support/query)
        non-IID: 个性化 episode (每客户端在自己分布的测试分区上 adapt-then-eval)
        episode 准确率的跨 episode 标准差反映 (IID) 采样波动 / (non-IID) 客户端异构,
        是稳定性对比的依据。
        """
        if self._eval_episodes is not None:
            return self._eval_episodes

        rng = random.Random(self.seed + 12345)
        if self.iid:
            self._eval_episodes = self._global_episodes(rng)
        else:
            self._eval_episodes = self._personalized_episodes(rng)
        return self._eval_episodes

    def _global_episodes(self, rng):
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
            episodes.append((
                self._apply(self.eval_tf, s_imgs),
                torch.tensor(s_lbls, dtype=torch.long),
                self._apply(self.eval_tf, q_imgs),
                torch.tensor(q_lbls, dtype=torch.long),
            ))
        return episodes

    def _personalized_episodes(self, rng):
        """每个客户端一个 episode: 在其测试分区上抽 support/query (绝对 20-way 标签)"""
        episodes = []
        for cid in range(self.num_clients):
            tby = self.client_test_by_class[cid]
            tex = self.client_test_x[cid]
            s_imgs, s_lbls, q_imgs, q_lbls = [], [], [], []
            for cls, pool in tby.items():
                pool = list(pool)
                rng.shuffle(pool)
                if len(pool) < 2:
                    continue
                k = min(self.k_shot_eval, max(1, len(pool) // 2))
                s_idx = pool[:k]
                q_idx = pool[k:k + self.query_per_class]
                for i in s_idx:
                    s_imgs.append(tex[i]); s_lbls.append(cls)
                for i in q_idx:
                    q_imgs.append(tex[i]); q_lbls.append(cls)
            if len(s_imgs) == 0 or len(q_imgs) == 0:
                continue
            episodes.append((
                self._apply(self.eval_tf, s_imgs),
                torch.tensor(s_lbls, dtype=torch.long),
                self._apply(self.eval_tf, q_imgs),
                torch.tensor(q_lbls, dtype=torch.long),
            ))
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
