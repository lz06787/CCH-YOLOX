#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


class YOLOPAFPN_NOFPN3(nn.Module):
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

        self.C3_p3 = CSPLayer(
            int(in_channels[2] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[1] * width), 3, 2, act=act
        )

        self.bu_conv3 = Conv(
            int(in_channels[1] * width), int(in_channels[2] * width), 3, 2, act=act
        )

        self.C3_n3 = CSPLayer(
            int(in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.C3_n4 = CSPLayer(
            int(in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # Sub-pixel convolution layer
        upscale_factor = 2
        upscale_factor2 = 2
        self.sub_pixel0 = nn.Sequential(
            nn.Conv2d(int(in_channels[2] * width), int(in_channels[1] * width * (upscale_factor ** 2)), 1),
            nn.PixelShuffle(upscale_factor),
        )
        self.sub_pixel1 = nn.Sequential(
            nn.Conv2d(int(in_channels[1]* width), int(in_channels[0]* width * (upscale_factor2 ** 2)), 1),
            nn.PixelShuffle(upscale_factor2))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.glb_conv = BaseConv(int(in_channels[2]* width), int(in_channels[1]* width), 1, 1, act=act)

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

        x2_ds_dr = self.bu_conv2(x2)

        x0_us_dd = self.sub_pixel0(x0) 
        
        glb = self.avgpool(x0)
        glb = self.glb_conv(glb)
        
        fuse_layer = x2_ds_dr + x1 + x0_us_dd + glb

        f_x0 = self.bu_conv3(fuse_layer)
        pan_out0 = self.C3_p3(f_x0)

        f_x1 = fuse_layer
        pan_out1 = self.C3_n3(f_x1)

        f_x2 = self.sub_pixel1(fuse_layer)
        pan_out2 = self.C3_n4(f_x2)

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
