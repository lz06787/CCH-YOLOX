
import torch.nn as nn

def GroupNorm(num_features, num_groups=64, eps=1e-5, affine=True, *args, **kwargs):
    if num_groups > num_features:
        print('------arrive maxum groub numbers of:', num_features)
        num_groups = num_features
    return nn.GroupNorm(num_groups, num_features, eps=eps, affine=affine)