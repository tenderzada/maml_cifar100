"""
1D Convolutional Network for Time-Series Data (Bearing Fault Diagnosis)
用于MAML的时序信号backbone

输入: [batch, 9, 2048] (9通道，2048时间步)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DBlock(nn.Module):
    """
    1D卷积块: Conv1D -> BN -> ReLU -> MaxPool
    """

    def __init__(self, in_channels, out_channels, kernel_size=7,
                 stride=1, padding=3, pool_size=2):
        super(Conv1DBlock, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels, momentum=1.0,
                                  track_running_stats=False)
        self.pool_size = pool_size

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        if self.pool_size > 1:
            x = F.max_pool1d(x, self.pool_size)
        return x


class Conv1D4(nn.Module):
    """
    4层1D卷积网络

    输入: [batch, 9, 2048]
    输出: [batch, n_way]

    结构:
    - Conv1D: 9 -> hidden_dim, pool/4 -> 512
    - Conv1D: hidden_dim -> hidden_dim, pool/4 -> 128
    - Conv1D: hidden_dim -> hidden_dim, pool/4 -> 32
    - Conv1D: hidden_dim -> hidden_dim, pool/4 -> 8
    - FC: hidden_dim * 8 -> n_way
    """

    def __init__(self, in_channels=9, hidden_dim=64, n_way=5, drop_rate=0.0):
        super(Conv1D4, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_way = n_way
        self.drop_rate = drop_rate

        self.layer1 = Conv1DBlock(in_channels, hidden_dim, kernel_size=7, padding=3, pool_size=4)
        self.layer2 = Conv1DBlock(hidden_dim, hidden_dim, kernel_size=5, padding=2, pool_size=4)
        self.layer3 = Conv1DBlock(hidden_dim, hidden_dim, kernel_size=5, padding=2, pool_size=4)
        self.layer4 = Conv1DBlock(hidden_dim, hidden_dim, kernel_size=3, padding=1, pool_size=4)

        # Dropout层
        self.dropout = nn.Dropout(p=drop_rate) if drop_rate > 0 else None

        # 2048 -> 512 -> 128 -> 32 -> 8
        self.fc = nn.Linear(hidden_dim * 8, n_way)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.dropout:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        if self.dropout:
            x = self.dropout(x)
        x = self.fc(x)
        return x

    def feature_extractor(self, x):
        """仅返回特征"""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        return x


class Conv1D4Functional(nn.Module):
    """
    函数式1D Conv4，用于MAML的内层优化
    允许传入外部参数进行前向传播

    输入: [batch, 9, 2048]
    """

    def __init__(self, in_channels=9, hidden_dim=64, n_way=5, drop_rate=0.0):
        super(Conv1D4Functional, self).__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.n_way = n_way
        self.drop_rate = drop_rate  # Dropout率

        # 卷积核大小和池化大小
        self.kernel_sizes = [7, 5, 5, 3]
        self.paddings = [3, 2, 2, 1]
        self.pool_sizes = [4, 4, 4, 4]

        # 初始化参数
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()

        # 4层卷积的参数
        in_ch = in_channels
        for i in range(4):
            out_ch = hidden_dim
            kernel_size = self.kernel_sizes[i]

            # Conv weights and bias
            w = nn.Parameter(torch.empty(out_ch, in_ch, kernel_size))
            nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
            self.vars.append(w)
            self.vars.append(nn.Parameter(torch.zeros(out_ch)))

            # BN weights and bias
            self.vars.append(nn.Parameter(torch.ones(out_ch)))
            self.vars.append(nn.Parameter(torch.zeros(out_ch)))

            # BN running mean and var (not learnable)
            self.vars_bn.append(nn.Parameter(torch.zeros(out_ch), requires_grad=False))
            self.vars_bn.append(nn.Parameter(torch.ones(out_ch), requires_grad=False))

            in_ch = hidden_dim

        # FC layer: hidden_dim * 8 -> n_way
        fc_in = hidden_dim * 8
        w = nn.Parameter(torch.empty(n_way, fc_in))
        nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='linear')
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(n_way)))

    def forward(self, x, vars=None, bn_training=True):
        """
        Args:
            x: 输入 [batch, 9, 2048]
            vars: 可选的外部参数
            bn_training: BN是否使用training模式
        """
        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for i in range(4):
            w, b = vars[idx], vars[idx + 1]
            padding = self.paddings[i]
            x = F.conv1d(x, w, b, stride=1, padding=padding)

            # Batch normalization
            w_bn, b_bn = vars[idx + 2], vars[idx + 3]
            running_mean = self.vars_bn[bn_idx]
            running_var = self.vars_bn[bn_idx + 1]
            x = F.batch_norm(x, running_mean, running_var,
                            weight=w_bn, bias=b_bn, training=bn_training)

            x = F.relu(x, inplace=True)
            x = F.max_pool1d(x, self.pool_sizes[i])

            # Dropout (在最后两层之后)
            if self.drop_rate > 0 and i >= 2 and bn_training:
                x = F.dropout(x, p=self.drop_rate, training=True)

            idx += 4
            bn_idx += 2

        x = x.view(x.size(0), -1)

        # Dropout before FC
        if self.drop_rate > 0 and bn_training:
            x = F.dropout(x, p=self.drop_rate, training=True)

        # FC layer
        w, b = vars[idx], vars[idx + 1]
        x = F.linear(x, w, b)

        return x

    def parameters(self):
        return self.vars


class Conv1D6Functional(nn.Module):
    """
    6层1D卷积网络 (更大容量)

    输入: [batch, 9, 2048]
    """

    def __init__(self, in_channels=9, hidden_dim=64, n_way=5, drop_rate=0.0):
        super(Conv1D6Functional, self).__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.n_way = n_way
        self.drop_rate = drop_rate

        # 6层配置
        self.kernel_sizes = [7, 5, 5, 3, 3, 3]
        self.paddings = [3, 2, 2, 1, 1, 1]
        self.pool_sizes = [2, 2, 2, 2, 2, 2]  # 2048 -> 32

        # 初始化参数
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()

        in_ch = in_channels
        for i in range(6):
            out_ch = hidden_dim if i < 3 else hidden_dim * 2
            kernel_size = self.kernel_sizes[i]

            # Conv weights and bias
            w = nn.Parameter(torch.empty(out_ch, in_ch, kernel_size))
            nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
            self.vars.append(w)
            self.vars.append(nn.Parameter(torch.zeros(out_ch)))

            # BN weights and bias
            self.vars.append(nn.Parameter(torch.ones(out_ch)))
            self.vars.append(nn.Parameter(torch.zeros(out_ch)))

            # BN running mean and var
            self.vars_bn.append(nn.Parameter(torch.zeros(out_ch), requires_grad=False))
            self.vars_bn.append(nn.Parameter(torch.ones(out_ch), requires_grad=False))

            in_ch = out_ch

        # FC layer
        fc_in = hidden_dim * 2 * 32  # 最后特征维度
        w = nn.Parameter(torch.empty(n_way, fc_in))
        nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='linear')
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(n_way)))

    def forward(self, x, vars=None, bn_training=True):
        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for i in range(6):
            w, b = vars[idx], vars[idx + 1]
            padding = self.paddings[i]
            x = F.conv1d(x, w, b, stride=1, padding=padding)

            w_bn, b_bn = vars[idx + 2], vars[idx + 3]
            running_mean = self.vars_bn[bn_idx]
            running_var = self.vars_bn[bn_idx + 1]
            x = F.batch_norm(x, running_mean, running_var,
                            weight=w_bn, bias=b_bn, training=bn_training)

            x = F.relu(x, inplace=True)
            x = F.max_pool1d(x, self.pool_sizes[i])

            # Dropout (在后3层之后)
            if self.drop_rate > 0 and i >= 3 and bn_training:
                x = F.dropout(x, p=self.drop_rate, training=True)

            idx += 4
            bn_idx += 2

        x = x.view(x.size(0), -1)

        # Dropout before FC
        if self.drop_rate > 0 and bn_training:
            x = F.dropout(x, p=self.drop_rate, training=True)

        w, b = vars[idx], vars[idx + 1]
        x = F.linear(x, w, b)

        return x

    def parameters(self):
        return self.vars


if __name__ == '__main__':
    # 测试模型
    x = torch.randn(4, 9, 2048)

    # Conv1D4
    model = Conv1D4(n_way=5)
    out = model(x)
    print(f"Conv1D4 output shape: {out.shape}")
    print(f"Conv1D4 parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Conv1D4Functional
    model_func = Conv1D4Functional(n_way=5)
    out = model_func(x)
    print(f"Conv1D4Functional output shape: {out.shape}")
    print(f"Conv1D4Functional parameters: {sum(p.numel() for p in model_func.vars):,}")

    # Conv1D6Functional
    model_func6 = Conv1D6Functional(n_way=5, hidden_dim=64)
    out = model_func6(x)
    print(f"Conv1D6Functional output shape: {out.shape}")
    print(f"Conv1D6Functional parameters: {sum(p.numel() for p in model_func6.vars):,}")
