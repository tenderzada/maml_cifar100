"""
4-layer Convolutional Network for MAML
经典的few-shot learning backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    卷积块: Conv -> BN -> ReLU -> MaxPool
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, max_pool=True):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, momentum=1.0,
                                  track_running_stats=False)
        self.max_pool = max_pool

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        if self.max_pool:
            x = F.max_pool2d(x, 2)
        return x


class Conv4(nn.Module):
    """
    4层卷积网络
    输入: 32x32 RGB图像
    输出: n_way维度的logits
    """

    def __init__(self, in_channels=3, hidden_dim=64, n_way=5):
        super(Conv4, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_way = n_way

        # 4层卷积块
        self.layer1 = ConvBlock(in_channels, hidden_dim)
        self.layer2 = ConvBlock(hidden_dim, hidden_dim)
        self.layer3 = ConvBlock(hidden_dim, hidden_dim)
        self.layer4 = ConvBlock(hidden_dim, hidden_dim)

        # 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2
        # 最终特征维度: hidden_dim * 2 * 2 = 256 (当hidden_dim=64时)
        self.fc = nn.Linear(hidden_dim * 2 * 2, n_way)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_extractor(self, x):
        """仅返回特征，不经过分类器"""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        return x


class Conv4Functional(nn.Module):
    """
    函数式的Conv4，用于MAML的内层优化
    允许传入外部参数进行前向传播
    """

    def __init__(self, in_channels=3, hidden_dim=64, n_way=5):
        super(Conv4Functional, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_way = n_way
        self.in_channels = in_channels

        # 初始化参数
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()

        # 4层卷积的参数
        for i in range(4):
            in_ch = in_channels if i == 0 else hidden_dim
            # Conv weights and bias
            w = nn.Parameter(torch.ones(hidden_dim, in_ch, 3, 3))
            nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
            self.vars.append(w)
            self.vars.append(nn.Parameter(torch.zeros(hidden_dim)))
            # BN weights and bias
            self.vars.append(nn.Parameter(torch.ones(hidden_dim)))
            self.vars.append(nn.Parameter(torch.zeros(hidden_dim)))
            # BN running mean and var (not learnable)
            self.vars_bn.append(nn.Parameter(torch.zeros(hidden_dim), requires_grad=False))
            self.vars_bn.append(nn.Parameter(torch.ones(hidden_dim), requires_grad=False))

        # FC layer
        fc_in = hidden_dim * 2 * 2
        w = nn.Parameter(torch.ones(n_way, fc_in))
        nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='linear')
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(n_way)))

    def forward(self, x, vars=None, bn_training=True):
        """
        Args:
            x: 输入图像
            vars: 可选的外部参数，用于MAML内层更新
            bn_training: BN是否使用training模式
        """
        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for i in range(4):
            w, b = vars[idx], vars[idx + 1]
            x = F.conv2d(x, w, b, stride=1, padding=1)

            # Batch normalization
            w_bn, b_bn = vars[idx + 2], vars[idx + 3]
            running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
            x = F.batch_norm(x, running_mean, running_var,
                            weight=w_bn, bias=b_bn, training=bn_training)

            x = F.relu(x, inplace=True)
            x = F.max_pool2d(x, 2)

            idx += 4
            bn_idx += 2

        x = x.view(x.size(0), -1)

        # FC layer
        w, b = vars[idx], vars[idx + 1]
        x = F.linear(x, w, b)

        return x

    def parameters(self):
        """返回所有可学习参数"""
        return self.vars


if __name__ == '__main__':
    # 测试模型
    model = Conv4(n_way=5)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print(f"Conv4 output shape: {out.shape}")  # [4, 5]

    model_func = Conv4Functional(n_way=5)
    out = model_func(x)
    print(f"Conv4Functional output shape: {out.shape}")  # [4, 5]
    print(f"Number of parameter groups: {len(list(model_func.vars))}")
