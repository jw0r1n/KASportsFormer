import torch
from torch import nn

class BoneCrossAttention(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mode='spatial'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.mode = mode
        # self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.qkv_q = nn.Linear(dim_in, dim_in, bias=qkv_bias)
        self.qkv_kv = nn.Linear(dim_in, dim_in * 2, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.bone_refusion = BoneRefusion()

    def forward_spatial(self, q, k, v):
        B, H, T, J, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # TODO: 跑一下看一下这边的size跟原始输入对不对的上, 不对好像用的不是这个，用的一直是原始的一个输入
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.permute(0, 2, 3, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x

    def forward_temporal(self, q, k, v):
        B, H, T, J, C = q.shape
        qt = q.transpose(2, 3)
        kt = k.transpose(2, 3)
        vt = v.transpose(2, 3)

        attn = (qt @ kt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ vt
        x = x.permute(0, 3, 2, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x

    def forward(self, x, x_limb_comb):
        # input x: (B, T, J, C)  (B, 27, 17, C)  BONE
        B, T, J, C = x.shape
        # x_limb = self.bone_refusion(x)
        q = self.qkv_q(x).reshape(B, T, J, 1, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        kv = self.qkv_kv(x_limb_comb).reshape(B, T, J, 2, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q = q[0]
        k = kv[0]
        v = kv[1]
        if self.mode == 'temporal':
            x = self.forward_temporal(q, k, v)
        elif self.mode == 'spatial':
            x = self.forward_spatial(q, k, v)
        else:
            raise NotImplementedError(self.mode)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


if __name__ == '__main__':
    dummy_x = torch.rand((1, 27, 17, 128))
    dummy_x_limb = torch.rand((1, 27, 17, 128))

    bone_cross_attention = BoneCrossAttention(128, 128)

    dummy_output = bone_cross_attention(dummy_x, dummy_x_limb)
    print(dummy_output.shape)