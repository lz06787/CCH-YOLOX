#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv
from .WSA import SpatialAttentionWSA

class YOLONOFPN11_WSA(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.upsample2 = nn.Upsample(scale_factor=4, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(3 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[2] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv

        self.C3_n4 = CSPLayer(
            int(2 * in_channels[2] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # Sub-pixel convolution layer
        # upscale_factor = 2
        # upscale_factor2 = 4
        # upscale_factor3 = 8
        # self.sub_pixel0 = nn.Sequential(
        #     nn.Conv2d(in_channels[1], in_channels[1] * (upscale_factor ** 2), 1),
        #     nn.PixelShuffle(upscale_factor),
        # )
        # self.sub_pixel1 = nn.Sequential(
        #     nn.Conv2d(in_channels[0], in_channels[0] * (upscale_factor2 ** 2), 1),
        #     nn.PixelShuffle(upscale_factor2))

        # self.sub_pixel3 = nn.Sequential(
        #     nn.Conv2d(in_channels[0], in_channels[0] * (upscale_factor3 ** 2), 1),
        #     nn.PixelShuffle(upscale_factor3))

        self.buconv1 = BaseConv(int(in_channels[1]* width), int(in_channels[2]* width), 3, stride=2, act=act)
        self.buconv2 = BaseConv(int(in_channels[0]* width), int(in_channels[1]* width), 3, stride=2, act=act)

        self.wsa = SpatialAttentionWSA()
    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out2 = self.buconv2(x2)
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        #f_out0 = self.sub_pixel0(fpn_out0)  # 512/16
        f_out0 = self.upsample(fpn_out0)
        f_out0 = torch.cat([f_out0, x1, fpn_out2], 1)  # 512->1024/16
        pan_out1 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(x0)  # 1024->256/16
        #f_out1 = self.sub_pixel1(fpn_out1)  # 256/8
        f_out1 = self.upsample2(fpn_out1)
        x2_wsa = self.wsa(x2)
        f_out1 = torch.cat([f_out1, x2_wsa], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        #f_out3 = self.maxpool(x2)
        f_out3 = self.buconv1(x1)
        f_out3 = torch.cat([f_out3, x0], 1)
        pan_out0 = self.C3_n4(f_out3)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
