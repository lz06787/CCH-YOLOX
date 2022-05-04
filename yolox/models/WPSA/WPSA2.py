import numpy as np
import torch
from torch import nn
from torch.nn import init

class SpatialAttentionWSA(nn.Module):
    def __init__(self,kernel_size=7,  side_num=2):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)

        return output


class ChannelAttentionPSA(nn.Module):
    def __init__(self,in_channel, reduction=16, side_num=2):
        super().__init__()
        channel = in_channel * side_num * side_num
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
        self.side_num=side_num
    def forward(self, x) :
        
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        
        return output


def window_partition(x, side_num: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    window_size = H // side_num
    x = x.view(B, C, side_num, window_size, side_num, window_size)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, window_size, window_size)
    return windows



def window_reverse(windows, side_num: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    window_size = H // side_num
    C = int(windows.shape[1] / (side_num*side_num))
    B = int(windows.shape[0])
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, C, side_num, side_num, window_size, window_size)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)

    return x


class WPSABlock2(nn.Module):

    def __init__(self, in_channel=512,reduction=16,kernel_size=7,side_num=2):
        super().__init__()
        self.ca=ChannelAttentionPSA(in_channel, reduction=reduction, side_num=side_num)
        self.sa=SpatialAttentionWSA(kernel_size=kernel_size, side_num=side_num)
        self.side_num=side_num

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    # 带残差边
    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        x1 = window_partition(x, side_num=self.side_num)
        out1 = self.ca(x1)
        out2 = self.sa(x1)
        out = out1 + out2
        x1 = x1 * out
        x2 = window_reverse(x1, side_num=self.side_num, H=x.shape[2], W=x.shape[3]) 
        return x2 + residual
    
    #不带残差边
    # def forward(self, x):
    #     b, c, _, _ = x.size()
    #     out=self.ca(x)
    #     out=self.sa(out)
    #     return out