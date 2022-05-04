import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import DeformConv2d
from .aspp_spp import ASPP_SPPF



class Conv3x3Norm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Conv3x3Norm, self).__init__()

        self.conv = DeformConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        #self.conv = ModulatedDeformConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        
        self.bn = nn.GroupNorm(num_groups=32, num_channels=out_channels)

    def forward(self, input, **kwargs):
        x = self.conv(input.contiguous(), **kwargs)
        x = self.bn(x)
        return x


class DyConv(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, conv_func=Conv3x3Norm):
        super(DyConv, self).__init__()

        self.DyConv = nn.ModuleList()
        self.DyConv.append(ASPP_SPPF(in_channels, out_channels))
        self.DyConv.append(ASPP_SPPF(in_channels, out_channels))
        self.DyConv.append(ASPP_SPPF(in_channels, out_channels))


        self.down_sample=nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
       

        self.AttnConv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.ReLU(inplace=True))

        self.h_sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        self.init_weights()

    def init_weights(self):
        for m in self.DyConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.AttnConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        next_x = {}
        feature_names = list(x.keys())
        for level, name in enumerate(feature_names):

            feature = x[name]

            temp_fea = [self.DyConv[1](feature)]

            if level > 0:
                y = self.down_sample(x[feature_names[level - 1]])
                temp_fea.append(self.DyConv[2](y))
            
            if level < len(x) - 1:
                z = F.upsample_bilinear(x[feature_names[level + 1]], size=[feature.size(2), feature.size(3)])
                temp_fea.append(self.DyConv[0](z))

            attn_fea = []
            res_fea = []
            for fea in temp_fea:
                res_fea.append(fea)
                attn_fea.append(self.AttnConv(fea))

            res_fea = torch.stack(res_fea)
            spa_pyr_attn = self.h_sigmoid(torch.stack(attn_fea))
            mean_fea = torch.mean(res_fea * spa_pyr_attn, dim=0, keepdim=False)
            next_x[name] = self.relu(mean_fea)

        return next_x


class LZHEAD_ASPP(nn.Module):
    def __init__(self, in_channels):
        super(LZHEAD_ASPP, self).__init__()

        #in_channels = cfg.MODEL.FPN.OUT_CHANNELS
        self.in_channels = in_channels
        channels = 128
        NUM_CONVS=1

        dyhead_tower = []
        for i in range(NUM_CONVS): # NUM_CONVS=6
            dyhead_tower.append(
                DyConv(
                    in_channels if i == 0 else channels,
                    channels,
                    conv_func=Conv3x3Norm,
                )
            )

        self.add_module('dyhead_tower', nn.Sequential(*dyhead_tower))



    def forward(self, x):
        #x = self.backbone(x)
        dyhead_tower = self.dyhead_tower(x)
        return dyhead_tower
