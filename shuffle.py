
from sre_constants import SUCCESS
import torch
from torch import nn

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % groups == 0)
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class ChannelAttention_Group(nn.Module):
    def __init__(self, in_planes, ratio=3, channels_per_group=1):
        super(ChannelAttention_Group, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 3, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 3, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.channels_per_group = channels_per_group
        self.groups = in_planes // self.channels_per_group

        self.avg_pool1d = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out

        batchsize, num_channels, height, width = out.data.size()
        out = out.view(batchsize, self.groups, self.channels_per_group, height*width)
        out = self.avg_pool(out)
        #out = torch.sum(out, 2)

        return self.sigmoid(out)

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.channels_per_group = 2
        self.channelattention = ChannelAttention_Group(6, channels_per_group=self.channels_per_group)
        self.groups = 6//self.channels_per_group
        self.left_part = 3
    def forward(self,x):
        batchsize, num_channels, height, width = x.data.size()
        ca = self.channelattention(x)
        ca_sum = torch.sum(ca, dim=0)
        a, indices = torch.sort(ca_sum, dim=0, descending=False)

        x = x.view(batchsize, self.groups, self.channels_per_group, height, width)
        x = torch.index_select(x, 1, indices.view(x.shape[1])) 
        x = x.view(batchsize, num_channels, height, width)
        left = x[:, :self.left_part, :, :]
        right = x[:, self.left_part:, :, :]
        # left = x[:, :self.left_part, :, :]
        # right = x[:, self.left_part:, :, :]
        # #out_left = self.left_conv(left)
        # out_right = self.conv2(right)

a = torch.rand(2,6,4,4)
# b = torch.rand(2,3,1,1)
# ca_sum = torch.sum(b, dim=0)
# c, indices = torch.sort(ca_sum, dim=0, descending=False)
# x = torch.index_select(a, 1, indices.view(a.shape[1]))
model0 = model()
y = model0(a)

# shuffle = channel_shuffle(a, groups=2)



