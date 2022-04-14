#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""
在pannet后面加bfp
需要搭配yolox/models/yolo_head_bfp2.py使用
因为bfp需要输入的input具有channel相同的特点，
所以在pannet输出三个向量（channel分别为128,256,512），用1*1的卷积进行处理，将三个向量的通道都变成128，在输入到bfp中
此时yolox_head文件的输入就变了，原三个向量通道为128,256,512，现在都为128
"""
import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv
from yolox.models.mmdt_models.necks.bfp import BFP

class YOLOPAFPN_BFP2(nn.Module):
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

        self.BFP = BFP(in_channels=in_channels,num_levels=3)
        self.reduce_panout1 = BaseConv(int(in_channels[1]*width),int(in_channels[0] * width),1,1,act=act)
        self.reduce_panout0=BaseConv(int(in_channels[2]*width),int(in_channels[0] * width),1,1,act=act)

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
     
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        
        # bfp_x0 = fpn_out0
        # bfp_x1 = f_out0
        # bfp_x2 = f_out1
        # bfp_input = [bfp_x2, bfp_x1, bfp_x0]
        # bfp_out = self.BFP(bfp_input)
        # [bfp_out2, bfp_out1, bfp_out0] = bfp_out

        # bfp_out1 = self.reduce_conv1(bfp_out1)


        pan_out2 = self.C3_p3(f_out1)  # 512->256/8
        # bu_conv2进行降采样，不改变通道数
        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        # yoloxs 模型下
        bfp_x0 = self.reduce_panout0(pan_out0) # 512->128
        bfp_x1 = self.reduce_panout1(pan_out1) # 256->128
        bfp_x2 = pan_out2 # 128
        bfp_input = [bfp_x2, bfp_x1, bfp_x0]
        bfp_out = self.BFP(bfp_input)
        [bfp_out2, bfp_out1, bfp_out0] = bfp_out

        outputs = (bfp_out2, bfp_out1, bfp_out0)
        return outputs
