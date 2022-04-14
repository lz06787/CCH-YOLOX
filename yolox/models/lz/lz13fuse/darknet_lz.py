#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from torch import nn

from .network_blocks_lz import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck, BaseConv2
import torch
import numpy as np

def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    "3x3 convolution with padding"
    
    kernel_size = np.asarray((3, 3))
    
    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
    
    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2
    
    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)
    
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                     padding=full_padding, dilation=dilation, bias=False)


class ASPP(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels_per_branch=256,
                 branch_dilations=(6, 12, 18)):
        
        super(ASPP, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels_per_branch,
                                  kernel_size=1,
                                  bias=False)
        
        self.conv_1x1_bn = torch.nn.BatchNorm2d(num_features=out_channels_per_branch)
        
        self.conv_3x3_first = conv3x3(in_channels, out_channels_per_branch, dilation=branch_dilations[0])
        self.conv_3x3_first_bn = torch.nn.BatchNorm2d(num_features=out_channels_per_branch)
        
        
        self.conv_3x3_second = conv3x3(in_channels, out_channels_per_branch, dilation=branch_dilations[1])
        self.conv_3x3_second_bn = torch.nn.BatchNorm2d(num_features=out_channels_per_branch)
        
        
        self.conv_3x3_third = conv3x3(in_channels, out_channels_per_branch, dilation=branch_dilations[2])
        self.conv_3x3_third_bn = torch.nn.BatchNorm2d(num_features=out_channels_per_branch)
        
        self.conv_1x1_pool = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels_per_branch,
                                       kernel_size=1,
                                       bias=False)
        self.conv_1x1_pool_bn = torch.nn.BatchNorm2d(num_features=out_channels_per_branch)
        
        self.conv_1x1_final = nn.Conv2d(in_channels=out_channels_per_branch * 5,
                                        out_channels=out_channels_per_branch,
                                        kernel_size=1,
                                        bias=False)
        self.conv_1x1_final_bn = torch.nn.BatchNorm2d(num_features=out_channels_per_branch)
        
    
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        conv_1x1_branch = self.relu(self.conv_1x1_bn(self.conv_1x1(x)))
        conv_3x3_first_branch = self.relu(self.conv_3x3_first_bn(self.conv_3x3_first(x)))
        conv_3x3_second_branch = self.relu(self.conv_3x3_second_bn(self.conv_3x3_second(x)))
        conv_3x3_third_branch = self.relu(self.conv_3x3_third_bn(self.conv_3x3_third(x)))
        
        global_pool_branch = self.relu(self.conv_1x1_pool_bn(self.conv_1x1_pool(nn.functional.adaptive_avg_pool2d(x, 1))))
        global_pool_branch = nn.functional.upsample_bilinear(input=global_pool_branch,
                                                             size=input_spatial_dim)
        
        features_concatenated = torch.cat([conv_1x1_branch,
                                           conv_3x3_first_branch,
                                           conv_3x3_second_branch,
                                           conv_3x3_third_branch,
                                           global_pool_branch],
                                          dim=1)
        
        features_fused = self.relu(self.conv_1x1_final_bn(self.conv_1x1_final(features_concatenated)))
        
        return features_fused

class CSPDarknet_LZ(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark2","dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",

        same = False
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

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
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv2(base_channels * 8, base_channels * 16, 3, 2, act=act),
            ASPP(base_channels * 16, base_channels * 16),
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




