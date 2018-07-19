# -*- coding: utf-8 -*-
"""
Defines a Wide Attention Residual Network (WARN) Model on Cifar10 and Cifar 100.
"""

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1 at gmail.com"

import torch
import torch.nn.functional as F


class Block(torch.nn.Module):
    def __init__(self, ni, no, stride):
        super(Block, self).__init__()
        self.bn0 = torch.nn.BatchNorm2d(ni)
        self.conv0 = torch.nn.Conv2d(ni, no, 3, stride=stride, padding=1, bias=False)
        torch.nn.init.kaiming_normal_(self.conv0.weight.data)
        self.bn1 = torch.nn.BatchNorm2d(no)
        self.conv1 = torch.nn.Conv2d(no, no, 3, stride=1, padding=1, bias=False)
        torch.nn.init.kaiming_normal_(self.conv1.weight.data)
        self.reduce = ni != no
        if self.reduce:
            self.conv_reduce = torch.nn.Conv2d(ni, no, 1, stride=stride, bias=False)
            torch.nn.init.kaiming_normal_(self.conv_reduce.weight.data)

    def forward(self, x):
        o1 = F.relu(self.bn0(x), inplace=True)
        y = self.conv0(o1)
        o2 = F.relu(self.bn1(y), inplace=True)
        z = self.conv1(o2)
        if self.reduce:
            return z + self.conv_reduce(x)
        else:
            return z + x


class Group(torch.nn.Module):
    def __init__(self, ni, no, n, stride):
        super(Group, self).__init__()
        self.n = n
        for i in range(n):
            self.__setattr__("block_%d" % i, Block(ni if i == 0 else no, no, stride if i == 0 else 1))

    def forward(self, x):
        for i in range(self.n):
            x = self.__getattr__("block_%d" % i)(x)
        return x


class WideResNet(torch.nn.Module):
    def __init__(self, depth, width, num_classes):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        self.n = (depth - 4) // 6
        self.num_classes = num_classes
        widths = [int(x * width) for x in [16, 32, 64]]
        self.conv0 = torch.nn.Conv2d(3, 16, 3, padding=1, bias=False)
        torch.nn.init.kaiming_normal_(self.conv0.weight.data)
        self.group_0 = Group(16, widths[0], self.n, 1)
        self.group_1 = Group(widths[0], widths[1], self.n, 2)
        self.group_2 = Group(widths[1], widths[2], self.n, 2)
        self.bn = torch.nn.BatchNorm2d(widths[2])
        self.classifier = torch.nn.Linear(widths[2], self.num_classes)
        torch.nn.init.kaiming_normal_(self.classifier.weight)

    def forward(self, x):
        x = self.conv0(x)
        g0 = self.group_0(x)
        g1 = self.group_1(g0)
        g2 = self.group_2(g1)
        o = F.relu(self.bn(g2))
        o = F.avg_pool2d(o, 8, 1, 0)
        o = o.view(o.size(0), -1)
        o = self.classifier(o)
        return o
