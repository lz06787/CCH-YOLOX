import torch
import numpy as np
import torch.nn as nn

from copy import deepcopy
import fvcore.nn.weight_init as weight_init
from torch.nn import functional as F
from .Group_Normalization import GroupNorm

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class DilateConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu", dilation=2
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2 + dilation - 1
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
            dilation=dilation
        )
        #self.bn = nn.BatchNorm2d(out_channels)
        self.bn = GroupNorm(out_channels, num_groups=32)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


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


class ASPP2(nn.Module):
    """ASPP (Atrous Spatial Pyramid Pooling)
    This is an implementation of the ASPP module used in DetectoRS
    (https://arxiv.org/pdf/2006.02334.pdf)
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by this module
        dilations (tuple[int]): Dilations of the four branches.
            Default: (1, 3, 6, 1)
    """

    def __init__(self, in_channels, out_channels, dilations=(1, 2, 5, 1)):
        super().__init__()
        assert dilations[-1] == 1
        self.aspp = nn.ModuleList()
        for dilation in dilations:
            kernel_size = 3 if dilation > 1 else 1
            padding = dilation if dilation > 1 else 0
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                bias=True)
            self.aspp.append(conv)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(len(self.aspp)):
            inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        return out


class ASPP3(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dilations,
        *,
        norm,
        activation,
        pool_kernel_size=None,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_channels (int): number of input channels for ASPP.
            out_channels (int): number of output channels.
            dilations (list): a list of 3 dilations in ASPP.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format. norm is
                applied to all conv layers except the conv following
                global average pooling.
            activation (callable): activation function.
            pool_kernel_size (tuple, list): the average pooling size (kh, kw)
                for image pooling layer in ASPP. If set to None, it always
                performs global average pooling. If not None, it must be
                divisible by the shape of inputs in forward(). It is recommended
                to use a fixed input feature size in training, and set this
                option to match this size, so that it performs global average
                pooling in training, and the size of the pooling window stays
                consistent in inference.
            dropout (float): apply dropout on the output of ASPP. It is used in
                the official DeepLab implementation with a rate of 0.1:
                https://github.com/tensorflow/models/blob/21b73d22f3ed05b650e85ac50849408dd36de32e/research/deeplab/model.py#L532  # noqa
        """
        super(ASPP, self).__init__()
        assert len(dilations) == 3, "ASPP expects 3 dilations, got {}".format(len(dilations))
        self.pool_kernel_size = pool_kernel_size
        self.dropout = dropout
        use_bias = norm == ""
        self.convs = nn.ModuleList()
        # conv 1x1
        self.convs.append(
            DilateConv(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=1,
                stride=1,
                dilation=1
            )
        )
        weight_init.c2_xavier_fill(self.convs[-1])
        # atrous convs
        for dilation in dilations:
            self.convs.append(
                Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    bias=use_bias,
                    norm=get_norm(norm, out_channels),
                    activation=deepcopy(activation),
                )
            )
            weight_init.c2_xavier_fill(self.convs[-1])
        # image pooling
        # We do not add BatchNorm because the spatial resolution is 1x1,
        # the original TF implementation has BatchNorm.
        if pool_kernel_size is None:
            image_pooling = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv2d(in_channels, out_channels, 1, bias=True, activation=deepcopy(activation)),
            )
        else:
            image_pooling = nn.Sequential(
                nn.AvgPool2d(kernel_size=pool_kernel_size, stride=1),
                Conv2d(in_channels, out_channels, 1, bias=True, activation=deepcopy(activation)),
            )
        weight_init.c2_xavier_fill(image_pooling[1])
        self.convs.append(image_pooling)

        self.project = Conv2d(
            5 * out_channels,
            out_channels,
            kernel_size=1,
            bias=use_bias,
            norm=get_norm(norm, out_channels),
            activation=deepcopy(activation),
        )
        weight_init.c2_xavier_fill(self.project)

    def forward(self, x):
        size = x.shape[-2:]
        if self.pool_kernel_size is not None:
            if size[0] % self.pool_kernel_size[0] or size[1] % self.pool_kernel_size[1]:
                raise ValueError(
                    "`pool_kernel_size` must be divisible by the shape of inputs. "
                    "Input size: {} `pool_kernel_size`: {}".format(size, self.pool_kernel_size)
                )
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res[-1] = F.interpolate(res[-1], size=size, mode="bilinear", align_corners=False)
        res = torch.cat(res, dim=1)
        res = self.project(res)
        res = F.dropout(res, self.dropout, training=self.training) if self.dropout > 0 else res
        return res