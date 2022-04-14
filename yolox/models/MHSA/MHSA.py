
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x
def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, h, l, _, device, dtype = *x.shape, x.device, x.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((x, col_pad), dim = 3)
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l-1):]
    return final_x

def relative_logits_1d(q, rel_k):
    b, heads, h, w, dim = q.shape
    logits = einsum('b h x y d, r d -> b h x y r', q, rel_k)
    logits = rearrange(logits, 'b h x y r -> b (h x) y r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, heads, h, w, w)
    logits = expand_dim(logits, dim = 3, k = h)
    return logits

# positional embeddings
class AbsPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, ' h w d -> (h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb)
        return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.fmap_size = fmap_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        h, w = self.fmap_size

        q = rearrange(q, 'b h (x y) d -> b h x y d', x = h, y = w)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b h x i y j-> b h (x y) (i j)')

        q = rearrange(q, 'b h x y d -> b h y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b h x i y j -> b h (y x) (j i)')
        return rel_logits_w + rel_logits_h

class MHSA_V1(nn.Module):
    def __init__(self, n_dims, heads=4,width=13,height=13):
        super(MHSA_V1, self).__init__()
        # self.width = int(input_size[0]/(2**downsample_times))
        # self.height= int(input_size[1]/(2**downsample_times))

        self.heads = heads
        self.n_dims = n_dims
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.rel_h = nn.Parameter(torch.randn([1, self.heads, self.n_dims // self.heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, self.heads, self.n_dims // self.heads, width, 1]), requires_grad=True)


        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        n_batch, C, width, height = x.size()

        #self.rel_h = nn.Parameter(torch.randn([1, self.heads, self.n_dims // self.heads, 1, height]), requires_grad=True)
        #self.rel_w = nn.Parameter(torch.randn([1, self.heads, self.n_dims // self.heads, width, 1]), requires_grad=True)

        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1).cuda()
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1).cuda()
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1).cuda()

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k).cuda()

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2).cuda()
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out

class MHSA_V2(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb
        self.pos_emb = rel_pos_class(fmap_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k, v = self.to_qkv(fmap).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), (q, k, v))

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb(q)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return out

class MHSA_V3(nn.Module):
    def __init__(
        self, dim, fmap_size, heads=4, dim_qk=128, dim_v=128, rel_pos_emb=False
    ):
        """
        dim: number of channels of feature map
        fmap_size: [H, W]
        dim_qk: vector dimension for q, k
        dim_v: vector dimension for v (not necessarily the same with q, k)
        """
        super().__init__()
        self.scale = dim_qk ** -0.5
        self.heads = heads
        out_channels_qk = heads * dim_qk
        out_channels_v = heads * dim_v

        self.to_qk = nn.Conv2d(
            dim, out_channels_qk * 2, 1, bias=False
        )  # 1*1 conv to compute q, k
        self.to_v = nn.Conv2d(
            dim, out_channels_v, 1, bias=False
        )  # 1*1 conv to compute v
        self.softmax = nn.Softmax(dim=-1)

        height, width = fmap_size
        if rel_pos_emb:
            self.pos_emb = RelPosEmb(height, width, dim_qk)
        else:
            self.pos_emb = AbsPosEmb(height, width, dim_qk)

    def forward(self, featuremap):
        """
        featuremap: [B, d_in, H, W]
        Output: [B, H, W, head * d_v]
        """
        heads = self.heads
        B, C, H, W = featuremap.shape
        q, k = self.to_qk(featuremap).chunk(2, dim=1)
        v = self.to_v(featuremap)
        q, k, v = map(
            lambda x: rearrange(x, "B (h d) H W -> B h (H W) d", h=heads), (q, k, v)
        )

        q *= self.scale

        logits = einsum("b h x d, b h y d -> b h x y", q, k)
        logits += self.pos_emb(q)

        weights = self.softmax(logits)
        attn_out = einsum("b h x y, b h y d -> b h x d", weights, v)
        attn_out = rearrange(attn_out, "B h (H W) d -> B (h d) H W", H=H)

        return attn_out