#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from tkinter.tix import Tree
import torch
import torch.nn as nn

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
 
 
        self.gamma = nn.Parameter(torch.zeros(1))  # β尺度系数初始化为0，并逐渐地学习分配到更大的权重
        self.softmax  = nn.Softmax(dim=-1)  # 对每一行进行softmax
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × C × C
        """
        m_batchsize, C, height, width = x.size()
        # A -> (N,C,HW)
        proj_query = x.view(m_batchsize, C, -1)
        # A -> (N,HW,C)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        # 矩阵乘积，通道注意图：X -> (N,C,C)
        energy = torch.bmm(proj_query, proj_key)
        # 这里实现了softmax用最后一维的最大值减去了原始数据，获得了一个不是太大的值
        # 沿着最后一维的C选择最大值，keepdim保证输出和输入形状一致，除了指定的dim维度大小为1
        # expand_as表示以复制的形式扩展到energy的尺寸
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        
        attention = self.softmax(energy_new)
        # A -> (N,C,HW)
        proj_value = x.view(m_batchsize, C, -1)
        # XA -> （N,C,HW）
        out = torch.bmm(attention, proj_value)
        # output -> (N,C,H,W)
        out = out.view(m_batchsize, C, height, width)
        
        out = self.gamma*out + x
        return out


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
        #max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        #out = avg_out + max_out
        return self.sigmoid(avg_out)

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


class BaseConv2(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu", c_tag=0.5
    ):
        super().__init__()
        self.ca = ChannelAttention(in_channels*2)
        self.left_part = round(in_channels)
        self.right_part_in = 2*in_channels - self.left_part
        self.right_part_out = out_channels - self.left_part

        # same padding
        pad = (ksize - 1) // 2

        self.conv0 = nn.Conv2d(
            in_channels,
            2*in_channels,
            kernel_size=1
        )
        self.bn0 = nn.BatchNorm2d(2*in_channels)
        self.act = get_activation(act, inplace=True)

        self.conv = nn.Conv2d(
             self.right_part_in,
             self.right_part_in,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(self.right_part_in)
        self.act = get_activation(act, inplace=True)

        self.conv2 = nn.Conv2d(
            2*in_channels,
            out_channels,
            kernel_size=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act =get_activation(act, inplace=True)

    def forward(self, x):

        x = self.act(self.bn0(self.conv0(x)))

        ca = self.ca(x)
        ca_sum = torch.sum(ca, dim=0)
        a, indices = torch.sort(ca_sum, dim=0, descending=False)
        x = torch.index_select(x, 1, indices.view(x.shape[1])) 
        left = x[:, :self.left_part, :, :]
        right = x[:, self.left_part:, :, :]
        out = self.act(self.bn(self.conv(right)))

        out = torch.cat((left, out), 1)
        out = self.act(self.bn2(self.conv2(out)))

        return out


    def fuseforward(self, x):
        return self.act(self.conv(x))


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
        #return self.pconv(x)
        return x

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

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(in_channels*expansion)
        Conv = DWConv if depthwise else BaseConv

        self.ca = ChannelAttention(hidden_channels)
        #self.ca = CAM_Module(hidden_channels)
        self.left_part = round(in_channels*0.5)
        self.right_part_in = hidden_channels - self.left_part
        self.right_part_out = out_channels - self.left_part

        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)

        self.use_add = shortcut and in_channels == out_channels
        
        self.left_conv = nn.Sequential(
                Conv(self.left_part, self.left_part, 3, stride=1, act=act),
                Conv(self.left_part, self.left_part, 3, stride=1, act=act)
                )
        
        #self.conv3 = Conv(hidden_channels, out_channels, 1, stride=1, act=act)
        self.groups = out_channels
    def forward(self, x):
        x = self.conv1(x)
        # ca = self.ca(x)
        # ca_sum = torch.sum(ca, dim=0)
        # a, indices = torch.sort(ca_sum, dim=0, descending=False)
        # x = torch.index_select(x, 1, indices.view(x.shape[1])) 
        # left = x[:, :self.left_part, :, :]
        # right = x[:, self.left_part:, :, :]     
        # out_left = self.left_conv(left)
        # out_right = self.conv2(right)

        # if self.use_add:
        #     out_right = out_right + right
            # out_left = out_left + left
        y = self.conv2(x)
        if self.use_add:
            y = y + x
        return y
        #return channel_shuffle(torch.cat((left, out_right), 1), self.groups)

# class Bottleneck(nn.Module):
#     # Standard bottleneck
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         shortcut=True,
#         expansion=0.5,
#         depthwise=False,
#         act="silu",
#     ):
#         super().__init__()
#         hidden_channels = int(in_channels*expansion)
#         Conv = DWConv if depthwise else BaseConv

#         self.ca = ChannelAttention(hidden_channels)
#         self.left_part = round(in_channels*0.5)
#         self.right_part_in = hidden_channels - self.left_part
#         self.right_part_out = out_channels - self.left_part

#         self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
#         self.conv2 = Conv(self.right_part_in, self.right_part_in, 3, stride=1, act=act)
#         self.use_add = shortcut and in_channels == out_channels
        
#         # self.left_conv = nn.Sequential(
#         #         Conv(self.left_part, self.left_part, 3, stride=1, act=act),
#         #         Conv(self.left_part, self.left_part, 3, stride=1, act=act)
#         #         )
#         self.left_conv = Conv(self.left_part, self.left_part, 5, stride=1, act=act)
        
#         self.conv3 = BaseConv(hidden_channels, out_channels, 1, stride=1, act=act)
#         self.bn = nn.BatchNorm2d(out_channels)
#         #self.groups = out_channels
#     def forward(self, x):
#         x = self.conv1(x)
#         left = x[:, :self.left_part, :, :]
#         right = x[:, self.left_part:, :, :]     
#         out_left = self.left_conv(left)
#         out_right = self.conv2(right)

#         if self.use_add:
#             out_right = out_right + right
#             out_left = out_left + left

#         x = torch.cat((out_left, out_right), 1)
#         ca = self.ca(x)
#         ca_sum = torch.sum(ca, dim=0)
#         a, indices = torch.sort(ca_sum, dim=0, descending=False)
#         x = torch.index_select(x, 1, indices.view(x.shape[1])) 
#         x = self.bn(x)
#         return x

# class Bottleneck(nn.Module):
#     # Standard bottleneck
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         shortcut=True,
#         expansion=0.5,
#         depthwise=False,
#         act="silu",
#     ):
#         super().__init__()
#         hidden_channels = int(out_channels * expansion)
#         Conv = DWConv if depthwise else BaseConv2
#         self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
#         self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
#         self.use_add = shortcut and in_channels == out_channels

#     def forward(self, x):
#         y = self.conv2(self.conv1(x))
#         if self.use_add:
#             y = y + x
#         return y


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
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
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
