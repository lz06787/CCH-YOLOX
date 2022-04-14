#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
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
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu",pad=None
    ):
        super().__init__()
        if pad == None:
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

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Inverted_DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, padding, stride=1, act="silu"):
        super().__init__()

        self.dconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=None,
        )
        self.pconv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            bias=None,
        )
        self.norm = LayerNorm(in_channels, eps=1e-6)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)
        #self.act = nn.GELU()
        # self.pconv = BaseConv(
        #     in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        # )
        #self.pconv = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.dconv(x)
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        #x = self.act(x)
        x = x.permute(0,3,1,2)
        x = self.pconv(x)
        #x = self.bn2(x)
        #x = self.act(x)
        
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
        hidden_channels = int(out_channels * expansion)
        hidden_channels2 = int(out_channels // 0.5)
        Conv = DWConv if depthwise else BaseConv

        self.dconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act = get_activation(act, inplace=True)

        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.dconv(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv2(self.conv1(y))

        if self.use_add:
            y = y + x
        return y

# class Inverted_Bottleneck(nn.Module):
#     # Standard bottleneck
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         #Inverted_ksize,
#         shortcut=True,
#         expansion=0.5,
#         depthwise=True,
#         act="silu",
#     ):
#         super().__init__()
#         #Inverted_padding = (Inverted_ksize-1) // 2
#         hidden_channels = int(out_channels * expansion)
#         hidden_channels2 = int(out_channels // 0.5)

#         self.dconv = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, groups=in_channels, bias=False, padding=2)
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         self.act = get_activation(act, inplace=True)

#         self.pconv = nn.Conv2d(
#             in_channels,
#             hidden_channels,
#             kernel_size=1,
#             stride=1,
#             groups=1,
#             bias=None,
#         )
#         self.bn2 = nn.BatchNorm2d(hidden_channels)

#         self.conv1 = nn.Conv2d(
#             hidden_channels,
#             hidden_channels2,
#             kernel_size=1,
#             stride=1,
#             groups=1,
#             bias=None,
#         )
#         self.bn3 = nn.BatchNorm2d(hidden_channels2)
#         self.act = get_activation(act, inplace=True)

#         self.conv2 = nn.Conv2d(
#             hidden_channels2,
#             out_channels,
#             kernel_size=3,
#             stride=1,
#             groups=1,
#             bias=None,
#             padding=1
#         )
#         self.bn4 = nn.BatchNorm2d(out_channels)

#         self.use_add = shortcut and in_channels == out_channels

        
#     def forward(self, x):
#         y = self.dconv(x)
#         y = self.bn1(y)
#         y = self.act(y)

#         y = self.pconv(y)
#         y = self.bn2(y)

#         y = self.conv1(y)
#         y = self.bn3(y)

#         y = self.conv2(y)
#         y = self.bn4(y)
#         y = self.act(y)

#         if self.use_add:
#             y = y + x
#         return y
        
# class Inverted_Bottleneck(nn.Module):
#     # Standard bottleneck
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         Inverted_ksize,
#         shortcut=True,
#         expansion=4,
#         depthwise=True,
#         act="silu",
#     ):
#         super().__init__()
#         Inverted_padding = (Inverted_ksize-1) // 2
#         hidden_channels = int(out_channels * expansion)
#         Conv = Inverted_DWConv if depthwise else BaseConv
#         #self.conv1 = BaseConv(hidden_channels, out_channels, 1, stride=1, act=act)

#         self.conv1 = nn.Conv2d(
#             hidden_channels,
#             out_channels,
#             kernel_size=1,
#             stride=1,
#             groups=1,
#             bias=None,
#         )

#         self.conv2 = Conv(in_channels, hidden_channels, Inverted_ksize, stride=1, padding=Inverted_padding, act=act)
#         # self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
#         # self.conv2 = Conv(hidden_channels, out_channels, Inverted_ksize, stride=1, pad=Inverted_padding, act=act)
#         self.use_add = shortcut and in_channels == out_channels

#         self.act = get_activation(act, inplace=True)
#     def forward(self, x):
#         y = self.conv2(x)
#         y = self.act(y)
#         y = self.conv1(y)
#         # y = self.conv2(self.conv1(x))
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


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5, e=0.5,activation="silu"):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        # c_ = c1 // 2  # hidden channels
        c_ = int(c1*e)
        self.cv1 = BaseConv(c1, c_, 1, stride=1, act=activation)
        self.cv2 = BaseConv(c_ * 4, c2, 1, stride=1, act=activation)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))



class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=1,
        depthwise=True,
        act="silu",
        Inverted_ksize = 5
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
        hidden_channels2 = int(out_channels // expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels2, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels2, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels2, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels2, hidden_channels2, shortcut, 0.5 ,depthwise, act=act
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

# class CSPLayer(nn.Module):
#     """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         n=1,
#         shortcut=True,
#         expansion=0.5,
#         depthwise=False,
#         act="silu",
#     ):
#         """
#         Args:
#             in_channels (int): input channels.
#             out_channels (int): output channels.
#             n (int): number of Bottlenecks. Default value: 1.
#         """
#         # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         hidden_channels = int(out_channels * expansion)  # hidden channels
#         self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
#         self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
#         self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
#         module_list = [
#             Bottleneck(
#                 hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
#             )
#             for _ in range(n)
#         ]
#         self.m = nn.Sequential(*module_list)

#     def forward(self, x):
#         x_1 = self.conv1(x)
#         x_2 = self.conv2(x)
#         x_1 = self.m(x_1)
#         x = torch.cat((x_1, x_2), dim=1)
#         return self.conv3(x)


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
