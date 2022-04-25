#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet_nospp import CSPDarknet_NOSPP
from .network_blocks import BaseConv, CSPLayer, DWConv, SPPBottleneck
from ..ASPP import ASPP

class YOLOASPPNOFPN(nn.Module):
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
        self.backbone = CSPDarknet_NOSPP(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.upsample2 = nn.Upsample(scale_factor=4, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.lateral_conv1 =  BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )

        self.C3_p4 = CSPLayer(
            int(in_channels[1] * width),
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
            int(in_channels[2] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.buconv = BaseConv(            
                int(in_channels[0] * width),
                int(in_channels[1] * width),
                3,
                2,
                act=act,)

        self.buconv2 = BaseConv(            
                int(in_channels[1] * width),
                int(in_channels[2] * width),
                3,
                2,
                act=act,)

        self.aspp = ASPP(
            3*int(in_channels[1] * width),
            int(in_channels[1] * width),
        )

        self.spp = SPPBottleneck(            
            3*int(in_channels[1] * width),
            int(in_channels[1] * width),)

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
        y2 = self.buconv(x2)
        y1 = x1
        y0 = self.lateral_conv0(x0)
        y0 = self.upsample(y0)

        f_out = torch.cat([y2,y1,y0], 1)
        f_out = self.spp(f_out)

        f_out2 = self.upsample(f_out)
        pan_out2 = self.C3_p3(f_out2)


        pan_out1 = self.C3_p4(f_out)

        f_out0 = self.buconv2(f_out)
        pan_out0 = self.C3_n4(f_out0)

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
