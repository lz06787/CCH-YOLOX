


import torch
import torch.nn as nn

class SpatialAttentionWSA(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x) :
        x1 = window_partition(x, window_num=4)
        max_result,_=torch.max(x1,dim=1,keepdim=True)
        avg_result=torch.mean(x1,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        output=window_reverse(output, window_num=4, H=x.shape[2], W=x.shape[3])
        return output*x


# def window_partition(x, window_size: int):
#     """
#     将feature map按照window_size划分成一个个没有重叠的window
#     Args:
#         x: (B, H, W, C)
#         window_size (int): window size(M)

#     Returns:
#         windows: (num_windows*B, window_size, window_size, C)
#     """
#     x = x.permute(0, 2, 3, 1)
#     B, H, W, C = x.shape
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
#     # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
#     # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
#     windows = windows.permute(0, 3, 1, 2)
#     return windows 


def window_partition(x, window_num: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    window_num = window_num
    window_size = H // window_num
    x = x.view(B, C, window_num, window_size, window_num, window_size)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


# def window_reverse(windows, window_size: int, H: int, W: int):
#     """
#     将一个个window还原成一个feature map
#     Args:
#         windows: (num_windows*B, window_size, window_size, C)
#         window_size (int): Window size(M)
#         H (int): Height of image
#         W (int): Width of image

#     Returns:
#         x: (B, H, W, C)
#     """
#     windows = windows.permute(0, 2, 3, 1)
#     B = int(windows.shape[0] / (H * W / window_size / window_size))
#     # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
#     x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
#     # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
#     # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     x = x.permute(0, 3, 1, 2)
#     return x

def window_reverse(windows, window_num: int, H: int, W: int):
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
    window_num = window_num
    window_size = H // window_num
    C = windows.shape[1]
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, C, window_size, window_size)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)

    return x