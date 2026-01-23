"""
轴承故障诊断数据集 - 原始数据加载器
支持从.pkl文件加载预处理好的数据
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import pickle


# 默认数据路径
DEFAULT_DATA_FILE = '/mnt/data/lev_data/bearing_data.pkl'


class BearingDataset(Dataset):
    """
    轴承故障诊断数据集

    数据形状: [N, 9, 2048] (9通道时序信号，2048时间步)
    类别数: 64
    """

    def __init__(
        self,
        data_file: str = DEFAULT_DATA_FILE,
        train: bool = True,
        transform=None,
    ):
        self.train = train
        self.transform = transform
        self.data_file = Path(data_file)

        if not self.data_file.exists():
            raise FileNotFoundError(
                f"数据文件未找到: {self.data_file}\n"
                f"请确保数据文件存在，或修改data_file参数"
            )

        self._load_dataset()

    def _load_dataset(self):
        """加载数据集"""
        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)

        self.num_classes = data['num_classes']
        self.class_to_idx = data['class_to_idx']
        self.idx_to_class = data['idx_to_class']
        self.window_size = data['window_size']
        self.num_channels = data['num_channels']

        if self.train:
            self.data = data['train_data']
            self.labels = data['train_labels']
        else:
            self.data = data['test_data']
            self.labels = data['test_labels']

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data = self.data[idx].astype(np.float32)  # [window_size, channels]
        label = int(self.labels[idx])

        # 转置为 [channels, window_size] 格式
        data = data.T  # [9, 2048]

        data = torch.from_numpy(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, label


def get_dataloaders(
    data_file: str = DEFAULT_DATA_FILE,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    获取训练集和测试集的DataLoader
    """
    train_dataset = BearingDataset(data_file=data_file, train=True)
    test_dataset = BearingDataset(data_file=data_file, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


if __name__ == '__main__':
    # 测试
    try:
        train_loader, test_loader = get_dataloaders(batch_size=32, num_workers=0)
        print(f"训练集: {len(train_loader.dataset)} 样本")
        print(f"测试集: {len(test_loader.dataset)} 样本")
        print(f"类别数: {train_loader.dataset.num_classes}")

        data, labels = next(iter(train_loader))
        print(f"Batch shape: {data.shape}")  # [32, 9, 2048]
    except FileNotFoundError as e:
        print(e)
