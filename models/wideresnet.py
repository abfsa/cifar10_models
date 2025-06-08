import torch
import torch.nn as nn
import torch.nn.functional as F

class WideBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super().__init__()
        # 第一层 3×3
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None
        # 第二层 3×3
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        # 如果 in_planes != planes 或 stride!=1，需要用 1×1 卷积调整残差捷径
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class WideResNet_CIFAR(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=10, dropout_rate=0.3):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth−4 必须能被 6 整除"
        n = (depth - 4) // 6
        k = widen_factor
        self.in_planes = 16 * k  # 从 16×k 开始

        # conv1: 3×3, 3→16×k
        self.conv1 = nn.Conv2d(3, 16 * k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16 * k)

        # Stage1：16k → 16k，共 n 个 block，stride=1
        self.layer1 = self._make_layer(WideBasicBlock, 16 * k, n, stride=1, dropout_rate=dropout_rate)
        # Stage2：16k → 32k，n 个 block，第一个 block stride=2
        self.layer2 = self._make_layer(WideBasicBlock, 32 * k, n, stride=2, dropout_rate=dropout_rate)
        # Stage3：32k → 64k，n 个 block，第一个 block stride=2
        self.layer3 = self._make_layer(WideBasicBlock, 64 * k, n, stride=2, dropout_rate=dropout_rate)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * k, num_classes)

        # 初始化权重（同 ResNet）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, dropout_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# 举例：构造一个 WRN-28-10，用于 CIFAR-10
def wrn_28_10():
    return WideResNet_CIFAR(depth=28, widen_factor=10, num_classes=10, dropout_rate=0.3)