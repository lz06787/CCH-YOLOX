import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import DeformConv2d

class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6

class AFT_FULL(nn.Module):

    def __init__(self, d_model,simple=False):

        super(AFT_FULL, self).__init__()
        self.inter_channel = d_model // 2
        # n = 256
        self.fc_q = nn.Conv2d(in_channels=d_model, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.fc_k = nn.Conv2d(in_channels=d_model, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.fc_v = nn.Conv2d(in_channels=d_model, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        # if(simple):
        #     self.position_biases=torch.zeros((n,n))
        # else:
        #     self.position_biases=nn.Parameter(torch.ones((n,n)))
        self.d_model = d_model
        # self.n=n
        # self.sigmoid=nn.Sigmoid()
        self.position_biases=nn.Parameter(torch.ones((self.inter_channel,self.inter_channel)))
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=d_model, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, input):

        bs, c, h , w = input.shape

        q = self.fc_q(input) #bs,c,h*w
        k = self.fc_k(input).view(1,bs,c//2,h*w) #1,bs,c,h*w
        v = self.fc_v(input).view(1,bs,c//2,h*w) #1,bs,c,h*w
        
        numerator=torch.sum(torch.exp(k+self.position_biases.view(c//2,1,-1,1))*v,dim=2) #c,bs,h*w
        denominator=torch.sum(torch.exp(k+self.position_biases.view(c//2,1,-1,1)),dim=2) #c,bs,h*w

        out=(numerator/denominator) #c,bs,h*w
        out=q*(out.permute(1,0,2).view(bs,c//2,h,w)) #bs,c,h,w
        out = self.conv_mask(out) #bs,c,h,w
        return out

class Conv3x3Norm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Conv3x3Norm, self).__init__()

        self.conv = DeformConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        #self.conv = ModulatedDeformConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        
        self.bn = nn.GroupNorm(num_groups=16, num_channels=out_channels)

    def forward(self, input, **kwargs):
        x = self.conv(input.contiguous(), **kwargs)
        x = self.bn(x)
        return x

class Attention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.att = AFT_FULL(channel)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.att(max_result)
        avg_out=self.att(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class DyConv(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, conv_func=Conv3x3Norm):
        super(DyConv, self).__init__()

        self.DyConv = nn.ModuleList()
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 2))


        self.AttnConv = nn.ModuleList()
        self.AttnConv.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Attention(256*2)
            ))
        self.AttnConv.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Attention(256*3)
        ))
        self.AttnConv.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Attention(256*3)
        ))
        self.AttnConv.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Attention(256*2)
        ))

        self.conv = nn.ModuleList()
        self.conv.append(BaseConv(256*2, 256, 1, 1))
        self.conv.append(BaseConv(256*3, 256, 1, 1))
        self.conv.append(BaseConv(256*2, 256, 1, 1))

        self.h_sigmoid = h_sigmoid()
        self.offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        next_x = {}
        feature_names = list(x.keys())
        for level, name in enumerate(feature_names):

            feature = x[name]

            offset_mask = self.offset(feature)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, 18:, :, :].sigmoid()
            conv_args = dict(offset=offset, mask=mask)

            
            temp_fea = [self.DyConv[1](feature, **conv_args)]

            if level > 0:
                temp_fea.append(self.DyConv[2](x[feature_names[level - 1]], **conv_args))
            
            if level < len(x) - 1:
                temp_fea.append(self.DyConv[0](F.upsample_bilinear(x[feature_names[level + 1]], size=[feature.size(2), feature.size(3)]), **conv_args)
                                                    )

            if level == 0 or level == 3:
                res_fea = torch.cat((temp_fea[0], temp_fea[1]), 1)

            else:
                res_fea = torch.cat((temp_fea[0], temp_fea[1], temp_fea[2]), 1)

            attn_fea = self.AttnConv[level](res_fea)
            mean_fea = attn_fea * res_fea + res_fea
            if level == 0 or level == 3:
                B,C,H,W = mean_fea.size()
                mean_fea = mean_fea.view(B,2,C//2,H,W)
                mean_fea = torch.mean(mean_fea, dim=1, keepdim=False)

            else:
                B,C,H,W = mean_fea.size()
                mean_fea = mean_fea.view(B,3,C//3,H,W)
                mean_fea = torch.mean(mean_fea, dim=1, keepdim=False)
            #mean_fea = self.conv[level](mean_fea)
            next_x[name] = mean_fea

        return next_x


class AFT_DCN(nn.Module):
    def __init__(self, in_channels):
        super(AFT_DCN, self).__init__()

        #in_channels = cfg.MODEL.FPN.OUT_CHANNELS
        self.in_channels = in_channels
        channels = 256
        NUM_CONVS = 3

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

if __name__ == "__main__":
    x = {}
    a = torch.ones(1,256,80,80).cuda()
    b = torch.ones(1,256,40,40).cuda()
    c = torch.ones(1,256,20,20).cuda()
    x["x2"] = a
    x["x1"] = b
    x["x0"] = c
    
    dyhead_out = AFT_DCN(256).cuda()
    out = dyhead_out(x)