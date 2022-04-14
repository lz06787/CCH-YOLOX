import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
import torch
from ..utils import ConvModule


class HighFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 pool_ratios=[0.1,0.2,0.3],
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 normalize=None,
                 activation=None):
        super(HighFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.with_bias = normalize is None

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                padding=0,
                norm_cfg=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)

            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                norm_cfg= normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add lateral conv for features generated by rato-invariant scale adaptive pooling
        self.adaptive_pool_output_ratio = pool_ratios
        self.high_lateral_conv = nn.ModuleList()
        self.high_lateral_conv.extend([nn.Conv2d(in_channels[-1], out_channels, 1) for k in range(len(self.adaptive_pool_output_ratio))])
        self.high_lateral_conv_attention = nn.Sequential(nn.Conv2d(out_channels*(len(self.adaptive_pool_output_ratio)), out_channels, 1),nn.ReLU(), nn.Conv2d(out_channels,len(self.adaptive_pool_output_ratio),3,padding=1))

        # add extra conv layers (e.g., RetinaNet
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_channels = (self.in_channels[self.backbone_end_level - 1]
                               if i == 0 else out_channels)
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    normalize=normalize,
                    bias=self.with_bias,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        for m in self.high_lateral_conv_attention.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)


        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        #Residual Feature Augmentation
        h, w = inputs[-1].size(2), inputs[-1].size(3)
        #Ratio Invariant Adaptive Pooling
        AdapPool_Features = [F.upsample(self.high_lateral_conv[j](F.adaptive_avg_pool2d(inputs[-1],output_size=(max(1,int(h*self.adaptive_pool_output_ratio[j])), max(1,int(w*self.adaptive_pool_output_ratio[j]))))), size=(h,w), mode='bilinear', align_corners=True) for j in range(len(self.adaptive_pool_output_ratio))]
        Concat_AdapPool_Features = torch.cat(AdapPool_Features, dim=1)
        fusion_weights = self.high_lateral_conv_attention(Concat_AdapPool_Features)
        fusion_weights = F.sigmoid(fusion_weights)
        adap_pool_fusion = 0
        for i in range(len(self.adaptive_pool_output_ratio)):
            adap_pool_fusion += torch.unsqueeze(fusion_weights[:,i, :,:], dim=1) * AdapPool_Features[i]

        # for Consistent Supervision 
        raw_laternals = [laterals[i].clone() for i in range(len(laterals))]

        # build top-down path
        laterals[-1] += adap_pool_fusion
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')
        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                orig = inputs[self.backbone_end_level - 1]
                outs.append(self.fpn_convs[used_backbone_levels](orig))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    # BUG: we should add relu before each extra conv
                    outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs), tuple(raw_laternals)