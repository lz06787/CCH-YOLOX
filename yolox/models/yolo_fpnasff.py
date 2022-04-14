#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv
from .asff import ASFFmobile

class YOLOFPNASFF(nn.Module):
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
        self.asff_level0 = ASFFmobile(level=0,rfb=False,vis=False)
        self.asff_level1 = ASFFmobile(level=1 ,rfb=False,vis=False)
        self.asff_level2 = ASFFmobile(level=2,rfb=False,vis=False)

        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
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
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
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
        [x2, x1, x0] = features

        asff_input0 = self.C3_n4(x0)    # 512/13->512/13

        fpn_out0 = self.lateral_conv0(x0)  # 512/13->256/13
        f_out0 = self.upsample(fpn_out0)  # 256/13->256/26
        f_out0 = torch.cat([f_out0, x1], 1)  # 256/26->512/26
        f_out0 = self.C3_p4(f_out0)       # 512/26->256/26
        asff_input1 = f_out0

        fpn_out1 = self.reduce_conv1(f_out0)  # 256/26->128/26
        f_out1 = self.upsample(fpn_out1)  # 128/26->128/52
        f_out1 = torch.cat([f_out1, x2], 1)  # 128/52->256/52
        asff_input2 = self.C3_p3(f_out1)    #256/52->128/52
        
        #pan_out2 = self.C3_p3(f_out1)  #256/52->128/52

        # p_out1 = self.bu_conv2(pan_out2)  # 128/52->128/26
        # p_out1 = torch.cat([p_out1, fpn_out1], 1)  #128/26->256/26
        # pan_out1 = self.C3_n3(p_out1)  # 256/26->256/26

        # p_out0 = self.bu_conv1(pan_out1)  #256/26->256/13
        # p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 256/13->512/13
        # pan_out0 = self.C3_n4(p_out0)  # 512/13->512/13

        outputs0 = self.asff_level0(asff_input0, asff_input1, asff_input2)
        outputs1 = self.asff_level1(asff_input0, asff_input1, asff_input2)
        outputs2 = self.asff_level2(asff_input0, asff_input1, asff_input2)
        #outputs = (pan_out2, pan_out1, pan_out0)
        return (outputs2,outputs1,outputs0)
