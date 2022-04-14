#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
from .shufflenetv2.shufflenetv2 import BasicUnit

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % groups == 0)
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class BasicUnit2(nn.Module):
    def __init__(self, inplanes, outplanes, ksize, stride, bias=False, c_tag=0.5, act="silu", SE=False, residual=False, groups=2):
        super(BasicUnit2, self).__init__()
        self.ca = ChannelAttention(inplanes)
        self.left_part = round(c_tag * inplanes)
        self.right_part_in = inplanes - self.left_part
        self.right_part_out = outplanes - self.left_part
        self.conv1 = nn.Conv2d(self.right_part_in, self.right_part_out, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.right_part_out)
        self.conv2 = nn.Conv2d(self.right_part_out, self.right_part_out, kernel_size=3, padding=1, bias=False,
                               groups=self.right_part_out)
        self.bn2 = nn.BatchNorm2d(self.right_part_out)
        self.conv3 = nn.Conv2d(self.right_part_out, self.right_part_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.right_part_out)
        self.activation = get_activation(act, inplace=True)

        self.inplanes = inplanes
        self.outplanes = outplanes
        self.residual = residual
        self.groups = groups
        self.SE = SE
        if self.SE:
            self.SELayer = SELayer(self.right_part_out, 2)  # TODO

    def forward(self, x):
        ca = self.ca(x)
        ca_sum = torch.sum(ca, dim=0)
        a, indices = torch.sort(ca_sum, dim=0, descending=False)
        x = torch.index_select(x, 1, indices.view(x.shape[1])) 
        left = x[:, :self.left_part, :, :]
        right = x[:, self.left_part:, :, :]
        out = self.conv1(right)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.activation(out)

        if self.SE:
            out = self.SELayer(out)
        if self.residual and self.inplanes == self.outplanes:
            out += right

        return channel_shuffle(torch.cat((left, out), 1), self.groups)

class BasicUnit(nn.Module):
    def __init__(self, inplanes, outplanes, ksize, stride, bias=False, c_tag=0.5, act="silu", SE=False, residual=False, groups=2):
        super(BasicUnit, self).__init__()
        self.left_part = round(c_tag * inplanes)
        self.right_part_in = inplanes - self.left_part
        self.right_part_out = outplanes - self.left_part
        self.conv1 = nn.Conv2d(self.right_part_in, self.right_part_out, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.right_part_out)
        self.conv2 = nn.Conv2d(self.right_part_out, self.right_part_out, kernel_size=3, padding=1, bias=False,
                               groups=self.right_part_out)
        self.bn2 = nn.BatchNorm2d(self.right_part_out)
        self.conv3 = nn.Conv2d(self.right_part_out, self.right_part_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.right_part_out)
        self.activation = get_activation(act, inplace=True)

        self.inplanes = inplanes
        self.outplanes = outplanes
        self.residual = residual
        self.groups = groups
        self.SE = SE
        if self.SE:
            self.SELayer = SELayer(self.right_part_out, 2)  # TODO

    def forward(self, x):
        left = x[:, :self.left_part, :, :]
        right = x[:, self.left_part:, :, :]
        out = self.conv1(right)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.activation(out)

        if self.SE:
            out = self.SELayer(out)
        if self.residual and self.inplanes == self.outplanes:
            out += right

        return channel_shuffle(torch.cat((left, out), 1), self.groups)

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        shufftenet=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
        if depthwise:
            Conv = DWConv
        elif shufftenet:
            Conv = BasicUnit2
        else:
            Conv = BaseConv

        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        shufftenet=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act, shufftenet=shufftenet
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)
