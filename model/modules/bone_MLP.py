import torch
from torch import nn
from model.modules.mlp import MLP


class BoneMLP(nn.Module):
    def __init__(self, composed_bone_num, mlp_hidden_dim=None):
        # input channel should be 3 (x_dir, y_dir, length)
        super().__init__()
        self.bone_mlp_hidden_dim = mlp_hidden_dim or 16
        self.bone_num = composed_bone_num
        self.mlp_dir_x = MLP(self.bone_num, self.bone_mlp_hidden_dim, 1)
        self.mlp_dir_y = MLP(self.bone_num, self.bone_mlp_hidden_dim, 1)
        self.mlp_len = MLP(self.bone_num, self.bone_mlp_hidden_dim, 1)

    def forward(self, x):
        # (B, T, J, C)  B, 27, 3, 3
        input_dir_x = x[:, :, :, 0]  # (1, 27, 3
        input_dir_y = x[:, :, :, 1]
        input_len = x[:, :, :, 2]
        output_dir_x = self.mlp_dir_x(input_dir_x)
        output_dir_y = self.mlp_dir_y(input_dir_y)
        output_len = self.mlp_len(input_len)

        output = torch.cat((output_dir_x, output_dir_y, output_len), dim=-1)
        output = output.unsqueeze(-2)
        return output


