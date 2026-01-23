"""
ResNet12 for Few-Shot Learning

ResNet12是few-shot learning中常用的更强大的backbone
相比Conv4有更大的模型容量，能够学习更复杂的特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


class BasicBlock(nn.Module):
    """
    ResNet基本块 (无bottleneck)
    """
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, momentum=1.0, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=1.0, track_running_stats=False)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes, momentum=1.0, track_running_stats=False)

        self.downsample = downsample
        self.drop_rate = drop_rate
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out, 0.1, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.leaky_relu(out, 0.1, inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.leaky_relu(out, 0.1, inplace=True)

        out = self.maxpool(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        return out


class ResNet12(nn.Module):
    """
    ResNet12 for Few-Shot Learning

    4个stage，每个stage包含一个BasicBlock
    channels: [64, 128, 256, 512] (可配置)
    """

    def __init__(self, in_channels=3, channels=[64, 128, 256, 512],
                 n_way=5, drop_rate=0.1):
        super(ResNet12, self).__init__()

        self.inplanes = in_channels
        self.n_way = n_way
        self.channels = channels

        self.layer1 = self._make_layer(channels[0], drop_rate=drop_rate)
        self.layer2 = self._make_layer(channels[1], drop_rate=drop_rate)
        self.layer3 = self._make_layer(channels[2], drop_rate=drop_rate)
        self.layer4 = self._make_layer(channels[3], drop_rate=drop_rate)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 分类器
        self.fc = nn.Linear(channels[3], n_way)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, drop_rate=0.0):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            nn.BatchNorm2d(planes, momentum=1.0, track_running_stats=False),
        )
        layer = BasicBlock(self.inplanes, planes, downsample, drop_rate)
        self.inplanes = planes
        return layer

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_extractor(self, x):
        """仅返回特征，不经过分类器"""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    @property
    def feature_dim(self):
        return self.channels[-1]


class ResNet12Functional(nn.Module):
    """
    函数式的ResNet12，用于MAML的内层优化
    允许传入外部参数进行前向传播
    """

    def __init__(self, in_channels=3, channels=[64, 128, 256, 512],
                 n_way=5, drop_rate=0.1):
        super(ResNet12Functional, self).__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.n_way = n_way
        self.drop_rate = drop_rate

        # 初始化参数
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()

        in_ch = in_channels
        for out_ch in channels:
            # 3个conv层
            for _ in range(3):
                w = nn.Parameter(torch.empty(out_ch, in_ch, 3, 3))
                nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='leaky_relu')
                self.vars.append(w)
                # BN参数
                self.vars.append(nn.Parameter(torch.ones(out_ch)))
                self.vars.append(nn.Parameter(torch.zeros(out_ch)))
                # BN running stats
                self.vars_bn.append(nn.Parameter(torch.zeros(out_ch), requires_grad=False))
                self.vars_bn.append(nn.Parameter(torch.ones(out_ch), requires_grad=False))
                in_ch = out_ch

            # downsample conv
            if in_channels != out_ch or True:  # 总是需要downsample
                in_ch_ds = in_channels if out_ch == channels[0] else channels[channels.index(out_ch) - 1]
                w = nn.Parameter(torch.empty(out_ch, in_ch_ds, 1, 1))
                nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='leaky_relu')
                self.vars.append(w)
                # BN参数
                self.vars.append(nn.Parameter(torch.ones(out_ch)))
                self.vars.append(nn.Parameter(torch.zeros(out_ch)))
                # BN running stats
                self.vars_bn.append(nn.Parameter(torch.zeros(out_ch), requires_grad=False))
                self.vars_bn.append(nn.Parameter(torch.ones(out_ch), requires_grad=False))

            in_channels = out_ch

        # FC layer
        w = nn.Parameter(torch.empty(n_way, channels[-1]))
        nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='linear')
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(n_way)))

    def forward(self, x, vars=None, bn_training=True):
        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        in_ch = self.in_channels
        for stage_idx, out_ch in enumerate(self.channels):
            identity = x

            # 3个conv层
            for conv_idx in range(3):
                w = vars[idx]
                x = F.conv2d(x, w, None, stride=1, padding=1)

                w_bn, b_bn = vars[idx + 1], vars[idx + 2]
                running_mean = self.vars_bn[bn_idx]
                running_var = self.vars_bn[bn_idx + 1]
                x = F.batch_norm(x, running_mean, running_var,
                                weight=w_bn, bias=b_bn, training=bn_training)

                if conv_idx < 2:  # 前两层加激活
                    x = F.leaky_relu(x, 0.1, inplace=True)

                idx += 3
                bn_idx += 2
                in_ch = out_ch

            # Downsample
            w = vars[idx]
            identity = F.conv2d(identity, w, None, stride=1, padding=0)
            w_bn, b_bn = vars[idx + 1], vars[idx + 2]
            running_mean = self.vars_bn[bn_idx]
            running_var = self.vars_bn[bn_idx + 1]
            identity = F.batch_norm(identity, running_mean, running_var,
                                   weight=w_bn, bias=b_bn, training=bn_training)
            idx += 3
            bn_idx += 2

            # Residual connection
            x = x + identity
            x = F.leaky_relu(x, 0.1, inplace=True)
            x = F.max_pool2d(x, 2)

            if self.drop_rate > 0 and bn_training:
                x = F.dropout(x, p=self.drop_rate, training=True)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        # FC layer
        w, b = vars[idx], vars[idx + 1]
        x = F.linear(x, w, b)

        return x

    def parameters(self):
        return self.vars


# 预定义的模型配置
def resnet12_small(n_way=5, drop_rate=0.1):
    """小型ResNet12: [64, 128, 256, 512]"""
    return ResNet12(channels=[64, 128, 256, 512], n_way=n_way, drop_rate=drop_rate)


def resnet12_large(n_way=5, drop_rate=0.1):
    """大型ResNet12: [64, 160, 320, 640]"""
    return ResNet12(channels=[64, 160, 320, 640], n_way=n_way, drop_rate=drop_rate)


def resnet12_functional_small(n_way=5, drop_rate=0.1):
    """函数式小型ResNet12"""
    return ResNet12Functional(channels=[64, 128, 256, 512], n_way=n_way, drop_rate=drop_rate)


def resnet12_functional_large(n_way=5, drop_rate=0.1):
    """函数式大型ResNet12"""
    return ResNet12Functional(channels=[64, 160, 320, 640], n_way=n_way, drop_rate=drop_rate)


if __name__ == '__main__':
    # 测试模型
    x = torch.randn(4, 3, 32, 32)

    model = ResNet12(n_way=5)
    out = model(x)
    print(f"ResNet12 output shape: {out.shape}")
    print(f"ResNet12 parameters: {sum(p.numel() for p in model.parameters()):,}")

    model_func = ResNet12Functional(n_way=5)
    out = model_func(x)
    print(f"ResNet12Functional output shape: {out.shape}")
    print(f"ResNet12Functional parameters: {sum(p.numel() for p in model_func.vars):,}")
