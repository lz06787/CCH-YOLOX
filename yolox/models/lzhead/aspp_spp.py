import torch
import numpy as np
import torch.nn as nn
import warnings
from copy import deepcopy
from torch.nn import functional as F
from ..Group_Normalization import GroupNorm
from ..network_blocks import BaseConv


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

class DilateConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu", dilation=2
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2 + dilation - 1
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
            dilation=dilation
        )
        #self.bn = nn.BatchNorm2d(out_channels)
        self.bn = GroupNorm(out_channels, num_groups=32)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class ASPP_SPPF(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_channels (int): number of input channels for ASPP.
            out_channels (int): number of output channels.
            dilations (list): a list of 3 dilations in ASPP.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format. norm is
                applied to all conv layers except the conv following
                global average pooling.
            activation (callable): activation function.
            pool_kernel_size (tuple, list): the average pooling size (kh, kw)
                for image pooling layer in ASPP. If set to None, it always
                performs global average pooling. If not None, it must be
                divisible by the shape of inputs in forward(). It is recommended
                to use a fixed input feature size in training, and set this
                option to match this size, so that it performs global average
                pooling in training, and the size of the pooling window stays
                consistent in inference.
            dropout (float): apply dropout on the output of ASPP. It is used in
                the official DeepLab implementation with a rate of 0.1:
                https://github.com/tensorflow/models/blob/21b73d22f3ed05b650e85ac50849408dd36de32e/research/deeplab/model.py#L532  # noqa
        """
        super(ASPP_SPPF, self).__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        # conv 1x1
        self.convs.append(
            BaseConv(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=1,
                stride=1,
            )
        )
        # atrous convs
        for dilation in [2,3]:
            self.convs.append(
                DilateConv(
                    in_channels,
                    out_channels,
                    ksize=3,
                    stride=1,
                    dilation=dilation,
                )
            )


        image_pooling = SPPF(in_channels, out_channels)

        #self.convs.append(image_pooling)

        self.project = BaseConv(
            3 * out_channels,
            out_channels,
            ksize=1,
            stride=1,
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        res = self.project(res)
        res = F.dropout(res, self.dropout, training=self.training) if self.dropout > 0 else res
        return res