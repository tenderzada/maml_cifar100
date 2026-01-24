"""
轴承故障诊断数据集 - Few-Shot Learning版本
将64类轴承数据转换为N-way K-shot的episode格式
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
import pickle
from pathlib import Path
from typing import Optional, List, Tuple


# 默认数据路径
DEFAULT_DATA_FILE = '/mnt/data/lev_data/bearing_data.pkl'


class BearingFewShot(Dataset):
    """
    轴承故障诊断 Few-Shot Dataset

    将64类轴承数据划分为:
    - meta-train: 40 classes
    - meta-val: 12 classes
    - meta-test: 12 classes

    数据形状: [9, 2048] (9通道时序信号)
    """

    def __init__(
        self,
        data_file: str = DEFAULT_DATA_FILE,
        mode: str = 'train',
        n_way: int = 5,
        k_shot: int = 1,
        k_query: int = 15,
        num_episodes: int = 600,
        transform=None,
    ):
        """
        Args:
            data_file: 数据文件路径
            mode: 'train', 'val', 或 'test'
            n_way: N-way分类
            k_shot: support set中每类样本数
            k_query: query set中每类样本数
            num_episodes: 每个epoch的episode数量
            transform: 数据变换
        """
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.num_episodes = num_episodes
        self.transform = transform
        self.mode = mode

        # 加载数据
        data_path = Path(data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件未找到: {data_file}")

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.num_classes = data['num_classes']
        self.idx_to_class = data['idx_to_class']

        # 合并训练集和测试集用于meta-learning
        # (few-shot learning中按类别划分，而非按样本划分)
        train_data = data['train_data']  # [N, window_size, channels]
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

        # 类别划分 (固定划分以保证可复现性)
        # 64类: 40 meta-train, 12 meta-val, 12 meta-test
        all_classes = list(range(self.num_classes))
        random.seed(42)
        random.shuffle(all_classes)

        if mode == 'train':
            self.classes = all_classes[:40]
        elif mode == 'val':
            self.classes = all_classes[40:52]
        else:  # test
            self.classes = all_classes[52:64]

        # 验证每个类有足够的样本
        min_samples_needed = k_shot + k_query
        for cls in self.classes:
            if len(self.data_by_class[cls]) < min_samples_needed:
                print(f"警告: 类别 {cls} 只有 {len(self.data_by_class[cls])} 个样本, "
                      f"需要 {min_samples_needed} 个")

        print(f"Bearing FewShot [{mode}]: {len(self.classes)} classes, "
              f"{num_episodes} episodes, {n_way}-way {k_shot}-shot")

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        """
        返回一个episode

        Returns:
            support_x: [n_way * k_shot, 9, 2048]
            support_y: [n_way * k_shot]
            query_x: [n_way * k_query, 9, 2048]
            query_y: [n_way * k_query]
        """
        # 随机选择n_way个类
        selected_classes = random.sample(self.classes, self.n_way)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for i, cls in enumerate(selected_classes):
            # 从该类中随机选择k_shot + k_query个样本
            class_samples = self.data_by_class[cls]
            indices = random.sample(range(len(class_samples)),
                                   min(self.k_shot + self.k_query, len(class_samples)))

            samples = [class_samples[j] for j in indices]

            # 划分support和query
            support_samples = samples[:self.k_shot]
            query_samples = samples[self.k_shot:self.k_shot + self.k_query]

            for sample in support_samples:
                # sample: [window_size, channels] -> [channels, window_size]
                data = sample.T.astype(np.float32)
                if self.transform:
                    data = self.transform(torch.from_numpy(data))
                else:
                    data = torch.from_numpy(data)
                support_x.append(data)
                support_y.append(i)  # 使用episode内的相对标签

            for sample in query_samples:
                data = sample.T.astype(np.float32)
                if self.transform:
                    data = self.transform(torch.from_numpy(data))
                else:
                    data = torch.from_numpy(data)
                query_x.append(data)
                query_y.append(i)

        support_x = torch.stack(support_x)
        support_y = torch.tensor(support_y)
        query_x = torch.stack(query_x)
        query_y = torch.tensor(query_y)

        return support_x, support_y, query_x, query_y


def get_bearing_fewshot_loader(
    data_file: str = DEFAULT_DATA_FILE,
    mode: str = 'train',
    n_way: int = 5,
    k_shot: int = 1,
    k_query: int = 15,
    num_episodes: int = 600,
    batch_size: int = 1,
    num_workers: int = 4,
    transform=None,
    strong_augment: bool = False,
):
    """
    获取轴承数据的few-shot dataloader

    Args:
        strong_augment: 是否使用强数据增强（推荐用于减少过拟合）
    """
    # 训练时可以添加数据增强
    if mode == 'train' and transform is None:
        if strong_augment:
            transform = get_strong_augmentation()
        else:
            transform = RandomJitter(sigma=0.03)

    dataset = BearingFewShot(
        data_file=data_file,
        mode=mode,
        n_way=n_way,
        k_shot=k_shot,
        k_query=k_query,
        num_episodes=num_episodes,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


class RandomJitter:
    """随机抖动增强 (高斯噪声)"""

    def __init__(self, sigma=0.05):
        self.sigma = sigma

    def __call__(self, x):
        noise = torch.randn_like(x) * self.sigma
        return x + noise


class RandomScale:
    """随机缩放增强"""

    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, x):
        scale = random.uniform(*self.scale_range)
        return x * scale


class RandomTimeWarp:
    """随机时间扭曲 (局部拉伸/压缩)"""

    def __init__(self, sigma=0.2, knot=4):
        self.sigma = sigma
        self.knot = knot

    def __call__(self, x):
        # x: [channels, length]
        from scipy.interpolate import CubicSpline
        import numpy as np

        orig_steps = np.arange(x.shape[1])

        # 生成扭曲锚点
        random_warps = np.random.normal(loc=1.0, scale=self.sigma, size=(self.knot + 2,))
        warp_steps = np.linspace(0, x.shape[1] - 1, num=self.knot + 2)

        # 累积扭曲
        time_warp = np.cumsum(random_warps)
        time_warp = (time_warp - time_warp[0]) / (time_warp[-1] - time_warp[0]) * (x.shape[1] - 1)

        # 插值
        cs = CubicSpline(warp_steps, time_warp)
        new_steps = cs(orig_steps)
        new_steps = np.clip(new_steps, 0, x.shape[1] - 1).astype(np.int32)

        return x[:, new_steps]


class RandomCrop:
    """随机裁剪并resize回原长度"""

    def __init__(self, crop_ratio=(0.8, 1.0)):
        self.crop_ratio = crop_ratio

    def __call__(self, x):
        # x: [channels, length]
        length = x.shape[1]
        crop_len = int(length * random.uniform(*self.crop_ratio))
        start = random.randint(0, length - crop_len)

        cropped = x[:, start:start + crop_len]

        # Resize back using interpolation
        if cropped.shape[1] != length:
            cropped = torch.nn.functional.interpolate(
                cropped.unsqueeze(0), size=length, mode='linear', align_corners=False
            ).squeeze(0)

        return cropped


class ChannelDropout:
    """随机丢弃部分通道"""

    def __init__(self, drop_prob=0.1):
        self.drop_prob = drop_prob

    def __call__(self, x):
        # x: [channels, length]
        if random.random() < self.drop_prob:
            num_channels = x.shape[0]
            drop_idx = random.randint(0, num_channels - 1)
            x = x.clone()
            x[drop_idx] = 0
        return x


class Mixup:
    """通道内Mixup (同一样本不同位置混合)"""

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        if random.random() < 0.5:
            lam = np.random.beta(self.alpha, self.alpha)
            # 随机移位混合
            shift = random.randint(1, x.shape[1] // 4)
            x_shifted = torch.roll(x, shifts=shift, dims=1)
            x = lam * x + (1 - lam) * x_shifted
        return x


class Compose:
    """组合多个增强"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class RandomApply:
    """以一定概率应用增强"""

    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            return self.transform(x)
        return x


def get_strong_augmentation():
    """获取强数据增强组合"""
    return Compose([
        RandomApply(RandomJitter(sigma=0.05), p=0.8),
        RandomApply(RandomScale(scale_range=(0.8, 1.2)), p=0.5),
        RandomApply(RandomCrop(crop_ratio=(0.85, 1.0)), p=0.3),
        RandomApply(ChannelDropout(drop_prob=0.1), p=0.2),
    ])


if __name__ == '__main__':
    # 测试
    try:
        loader = get_bearing_fewshot_loader(
            mode='train',
            n_way=5,
            k_shot=1,
            k_query=15,
            num_episodes=100,
            num_workers=0
        )

        for support_x, support_y, query_x, query_y in loader:
            print(f"Support X shape: {support_x.shape}")  # [1, 5, 9, 2048]
            print(f"Support Y shape: {support_y.shape}")  # [1, 5]
            print(f"Query X shape: {query_x.shape}")      # [1, 75, 9, 2048]
            print(f"Query Y shape: {query_y.shape}")      # [1, 75]
            break

    except FileNotFoundError as e:
        print(e)
