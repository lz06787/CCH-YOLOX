#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from torch import nn

from .network_blocks_lz import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck


class CSPDarknet_LZ(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=True,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        # self.stem = Focus(3, base_channels, ksize=3, act=act)

        dims=[base_channels*2, base_channels*4, base_channels*8, base_channels*16]
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            # LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            nn.BatchNorm2d(dims[0])
        )
        self.downsample_layers.append(self.stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    #LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.BatchNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # dark2
        self.dark2 = nn.Sequential(
            #Conv(base_channels, base_channels * 2, 3, 2, act=act),
            self.downsample_layers[0],
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            #Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            self.downsample_layers[1],
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            #Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            self.downsample_layers[2],
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 9,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            #Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            self.downsample_layers[3],
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):
        outputs = {}
        # x = self.stem(x)
        # outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
