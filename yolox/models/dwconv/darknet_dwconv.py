#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from random import shuffle
from torch import nn

from .network_blocks_dwconv import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck, ChannelAttention, BasicUnit2
from .shufflenetv2.shufflenetv2 import BasicUnit

class CSPDarknet_DWCONV(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark2", "dark3", "dark4", "dark5"),
        depthwise=False,
        shufftenet=True,
        act="silu",
        same = False
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features

        if depthwise:
            Conv = DWConv
        elif shufftenet:
            Conv = BasicUnit2
        else:
            Conv = BaseConv

        if same:
            Conv2 = Conv
        else:
            Conv2 = BaseConv

        base_channels = int(wid_mul *64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv2(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                shufftenet=shufftenet,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv2(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                shufftenet=shufftenet,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv2(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                shufftenet=shufftenet,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv2(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                shufftenet=shufftenet,
                act=act,
            ),
        )

    def forward(self, x):
        
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
