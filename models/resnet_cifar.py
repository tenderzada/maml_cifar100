"""
CIFAR-adapted ResNet18 / ResNet34

与 torchvision 标准 ResNet18/34 (输入 224x224) 的差异:
- 首层 conv 改为 3x3 stride=1, 去掉首层 maxpool, 适配 32x32 输入
- BN 设 track_running_stats=False, momentum=1.0 (与项目原有 ResNet12 一致),
  避免 MAML/Meta-SGD 内层适应时需要管理 BN running stats
- 不使用 dropout

适用于 FedAvg / FedAvg+MAML / FedAvg+Meta-SGD 联邦学习对比实验。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=1.0,
                                  track_running_stats=False)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=1.0,
                                  track_running_stats=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion, momentum=1.0,
                               track_running_stats=False),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=20, in_channels=3):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=1.0, track_running_stats=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=1.0,
                                  track_running_stats=False)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=1.0,
                                  track_running_stats=False)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=1.0,
                                  track_running_stats=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion, momentum=1.0,
                               track_running_stats=False),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


def resnet18_cifar(num_classes=20):
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34_cifar(num_classes=20):
    return ResNetCIFAR(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50_cifar(num_classes=20):
    return ResNetCIFAR(Bottleneck, [3, 4, 6, 3], num_classes)


if __name__ == '__main__':
    for name, fn in [('resnet18', resnet18_cifar),
                     ('resnet34', resnet34_cifar),
                     ('resnet50', resnet50_cifar)]:
        m = fn(num_classes=20)
        n = sum(p.numel() for p in m.parameters())
        x = torch.randn(2, 3, 32, 32)
        y = m(x)
        print(f"{name}: params={n/1e6:.2f}M, output={tuple(y.shape)}")
