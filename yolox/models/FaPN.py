from .network_blocks import BaseConv
import torch
import torch.nn.functional as F
from torch import nn 
from torchvision.ops import DeformConv2d

class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = BaseConv(in_chan, in_chan, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.conv = BaseConv(in_chan, out_chan, 1, 1)
        # weight_init.c2_xavier_fill(self.conv_atten)
        # weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:]))) #fm 就是激活加卷积，我觉得这平均池化用的贼巧妙
        feat = torch.mul(x, atten) #相乘，得到重要特征
        x = x + feat #再加上
        feat = self.conv(x) #最后一层 1*1 的卷积
        return feat

class FeatureAlignModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureAlignModule, self).__init__()
        self.offset = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1)
        self.dcn = DeformConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, feat_arm, feat_up):
        
        x = torch.cat([feat_arm, feat_up], dim=1)

        offset = self.offset(x)

        x = self.dcn(feat_up, offset)

        out = self.silu(x)
        
        return out

