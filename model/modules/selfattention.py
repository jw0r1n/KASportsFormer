import torch
from torch import nn

class Attention(nn.Module):
    # simplified version of attention, with spatial and temporal reshape. x tensor (B, T, J, C), not (B * T, J, C)
    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,mode='spatial'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # 规范化因子 d_K

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.mode = mode
        self.qkv = nn.Linear(dim_in, dim_in * 3 ,bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward_spatial(self, q, k, v):
        B, H, T, J, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, J, J)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v # (B, H, T, J, C)  # 算出来的是矩阵b
        x = x.permute(0, 2, 3, 1, 4).reshape(B, T, J, C * self.num_heads) # (B, T, J, H, C) -> (B, T, J, H, C)
        return x


    def forward_temporal(self, q, k, v):
        B, H, T, J, C = q.shape
        qt = q.transpose(2, 3)  # (B, H, J, T, C)
        kt = k.transpose(2, 3)
        vt = v.transpose(2, 3)

        attn = (qt @ kt.transpose(-2, -1)) * self.scale # (B, H, J, T, T)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ vt  # (B, H, J, T, C)
        x = x.permute(0, 3, 2, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x   # (B, T, J, C)


    def forward(self, x):
        B, T, J, C = x.shape

        qkv = self.qkv(x).reshape(B, T, J, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5) # (3, B, H, T, J, C)

        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.mode == 'temporal':
            x = self.forward_temporal(q, k, v)
        elif self.mode == "spatial":
            x = self.forward_spatial(q, k, v)
        else:
            raise NotImplementedError(self.mode)

        x = self.proj(x)  # 做完qkv是不改变dim的，最后输出前的pro可以改变维度dim
        x = self.proj_drop(x)

        return x


if __name__ == '__main__':
    dummy_x1 = torch.rand((3, 128, 8, 27, 17, 64))
    print(dummy_x1.shape)
    dummy_x2 = dummy_x1.transpose(-1, -2)
    print(dummy_x2.shape)

    res = dummy_x1 @ dummy_x2
    print(res.shape)
    res = nn.Parameter(res)
    res = res.softmax(-1)
    print(res.shape)

    dummy_x3 = torch.rand((3, 128, 8, 27, 17, 64))
    dummy_x3 = dummy_x3.transpose(2, 3)
    print(dummy_x3.shape)


    temp_layer = Attention(512,512, 8)
    dummyX = torch.rand((128, 27, 17, 512))
    dummyY = temp_layer(dummyX)
    print(dummyY.shape)







