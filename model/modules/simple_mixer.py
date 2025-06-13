import torch
from torch import nn

class SimpleFusion(nn.Module):
    def __init__(self, dim, mode='spatial'):
        super().__init__()
        self.dim_in = dim
        self.mode = mode
        self.softmax_fusion_linear = nn.Linear(dim * 2, 2)

    def forward_temporal(self, x1, x2):
        x1t = x1.transpose(1, 2)
        x2t = x2.transpose(1, 2)
        factor = torch.cat((x1t, x2t), dim=-1)
        factor = self.softmax_fusion_linear(factor)
        factor = factor.softmax(dim=-1)
        x = x1t * factor[..., 0:1] + x2t * factor[..., 1:2]
        x = x.transpose(1, 2)
        return x

    def forward_spatial(self, x1, x2):
        factor = torch.cat((x1, x2), dim=-1)
        factor = self.softmax_fusion_linear(factor)
        factor = factor.softmax(dim=-1)
        x = x1 * factor[..., 0:1] + x2 * factor[..., 1:2]
        return x


    def forward(self, x1, x2):
        if self.mode == 'temporal':
            x = self.forward_temporal(x1, x2)
        elif self.mode == 'spatial':
            x = self.forward_spatial(x1, x2)
        else:
            raise NotImplementedError(self.mode)

        return x



if __name__ == '__main__':
    dummy_x = torch.rand((1, 27, 17, 128))
    dummy_x_limb = torch.rand((1, 27, 17, 128))
    # d1 = dummy_x.transpose(1, 2)
    # print(d1.shape)
    # d2 = d1.transpose(1, 2)
    # print(d2.shape)
    simple_fusion = SimpleFusion(128)
    dummy_output = simple_fusion(dummy_x, dummy_x_limb)
    print(dummy_output.shape)

