"""
与resnet.py中的ImageNet版本不同，这里的ResNet版本是为了cifar10数据集的分类任务而设计的。
通道数更小，模型的整体参数量更小
"""

import torch.nn as nn
import torch
from .resnet import conv3x3, BasicBlock, Bottleneck

class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, base_channels=16):
        super().__init__()
        self.in_planes = base_channels
        # 第一层用 3×3 卷积，输出 base_channels
        self.conv1 = conv3x3(3, base_channels)
        self.bn1   = nn.BatchNorm2d(base_channels)
        # 三个 stage，channel = base, 2×base, 4×base
        self.layer1 = self._make_layer(block, base_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels*4, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(base_channels*4*block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# 常用变体：
def ResNet20_cifar(base=16):  return ResNetCIFAR(BasicBlock, [3, 3, 3], base_channels=base)
def ResNet32_cifar(base=16):  return ResNetCIFAR(BasicBlock, [5, 5, 5], base_channels=base)
def ResNet44_cifar(base=16):  return ResNetCIFAR(BasicBlock, [7, 7, 7], base_channels=base)
def ResNet56(base=16):  return ResNetCIFAR(Bottleneck, [9, 9, 9], base_channels=base)
#def ResNet110(): return ResNetCIFAR(BasicBlock, [18,18,18])