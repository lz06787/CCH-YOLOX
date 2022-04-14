#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from yolox.models.dwconv.darknet_dwconv import CSPDarknet_DWCONV
from .network_blocks import BaseConv, CSPLayer, DWConv


class YOLOPAFPN_DWCONV(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark2","dark3", "dark4", "dark5"),
        in_channels=[128, 256, 512, 1024],
        depthwise=True,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet_DWCONV(depth, width, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[3] * width), int(in_channels[2] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[2] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )

        self.reduce_conv2 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )

        self.C3_p3 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[2] * width), int(in_channels[2] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[2] * width),
            int(in_channels[3] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.c3_p2 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.bu_conv3 =  Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )

        self.C3_n2 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

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
        [x3, x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 512/13->256/13
        f_out0 = self.upsample(fpn_out0)  # 256/13->256/26
        f_out0 = torch.cat([f_out0, x1], 1)  # 256/26->512/26
        f_out0 = self.C3_p4(f_out0)       # 512/26->256/26

        fpn_out1 = self.reduce_conv1(f_out0)  # 256/26->128/26
        f_out1 = self.upsample(fpn_out1)  # 128/26->128/52
        f_out1 = torch.cat([f_out1, x2], 1)  # 128/52->256/52
        f_out1 = self.C3_p3(f_out1)     # 256/52->128/52
        fpn_out2 = self.reduce_conv2(f_out1) # 128/52->64/52
        f_out2 = self.upsample(fpn_out2)    # 64/104->64/104
        f_out2 = torch.cat([f_out2, x3], 1) # 64/104->128/104
        pan_out3 = self.c3_p2(f_out2)       # 128/104->64/104

        p_out2 = self.bu_conv3(pan_out3)    # 64/104->64/52
        p_out2 = torch.cat([p_out2, fpn_out2], 1)   # 64/52->128/52
        pan_out2 = self.C3_n2(p_out2)   # 128/52->128/52


        p_out1 = self.bu_conv2(pan_out2)  # 128/52->128/26
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  #128/26->256/26
        pan_out1 = self.C3_n3(p_out1)  # 256/26->256/26

        p_out0 = self.bu_conv1(pan_out1)  #256/26->256/13
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 256/13->512/13
        pan_out0 = self.C3_n4(p_out0)  # 512/13->512/13

        outputs = (pan_out3, pan_out2, pan_out1, pan_out0)
        return outputs