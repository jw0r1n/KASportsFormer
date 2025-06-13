import torch
from torch import nn
from model.modules.bone_MLP import BoneMLP

"""
肢干
0-1 + 1-2 + 2-3     0 1 2
0-4 + 4-5 + 5-6     3 4 5
0-7 + 7-8           6 7
8-9 + 9-10          8 9
8-11 + 11-12 + 12-13    10 11 12
8-14 + 14-15 + 15-16    13 14 15

肢干 对 脊柱  相对运动
0-7 + 7-8   +   1-2 + 2-3   6 7 1 2
0-7 +7-8   +   4-5 + 5-6    6 7 4 5
0-7 + 7-8   +   11-12 + 12-13   6 7 11 12
0-7 + 7-8   +   14-15 + 15-16   6 7 14 15
0-7 + 7-8 + 9-10    6 7 9

双手 双脚
14-15 + 15-16   +   11-12 + 12-13   14 15 11 12
1-2 + 2-3   +   4-5 + 5-6   1 2 4 5

左右协调问题
14-15 + 15-16   +   4-5 + 5-6   14 15 4 5
11-12 + 12-13   +   1-2 + 2-3   11 12 4 5

肩部与胯部的组合
8-11   +   0-1  10 0
8-14   +   0-4  13 3

"""
predefined_limb_combine = [
    [0, 1, 2], [3, 4, 5], [6, 7], [8, 9], [10, 11, 12], [13, 14, 15],
    [6, 7, 1, 2], [6, 7, 4, 5], [6, 7, 11, 12], [6, 7, 14, 15], [6, 7, 9],
    [14, 15, 11, 12], [1, 2, 4, 5],
    [14, 15, 4, 5], [11, 12, 4, 5],
    [10, 0], [13, 3]
]


class BoneRefusion(nn.Module):
    def __init__(self, limb_combine=None, mlp_hidden_dim=None):
        super().__init__()
        self.limb_combine = limb_combine or predefined_limb_combine
        self.mlp_hidden_dim = mlp_hidden_dim or 16
        if len(self.limb_combine) != 17:
            raise ValueError("The length of limb_combine should be 17")
        self.mlp_layers = self.create_mlp_layers()
        if len(self.mlp_layers) != 17:
            raise ValueError("The length of mlp_layers should be 17")

    def create_mlp_layers(self):
        mlp_layers = []
        for i in self.limb_combine:
            mlp_layers.append(BoneMLP(len(i), self.mlp_hidden_dim))
        layers = nn.Sequential(*mlp_layers)
        return layers

    def forward(self, x):
        # input x: (B, T, J, C) (B, 27, 17, C)
        mlp_res_list = []
        for idx, layer in enumerate(self.mlp_layers):
            limbs = self.limb_combine[idx]
            limb_input = x[:, :, limbs, :]
            limb_output = layer(limb_input)
            mlp_res_list.append(limb_output)
        out = torch.cat(mlp_res_list, dim=-2)
        return out


if __name__ == '__main__':
    bone_refusion = BoneRefusion()
    dummy_input = torch.randn(1, 27, 17, 3)
    dummy_output = bone_refusion(dummy_input)
    print(dummy_output.shape)
