from collections import OrderedDict

import torch
# import numpy as np
from torch import nn
from model.modules.mlp import MLP
from model.modules.selfattention import Attention
from model.modules.graph import GCN
# from model.modules.tcn import MultiScaleTCN
from model.modules.bone_refusion import BoneRefusion
from model.modules.bone_crossattention import BoneCrossAttention
from timm.models.layers import DropPath


def create_layers(dim=128, n_layers=26, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop_rate=0., drop_path_rate=0.,
                  num_heads=8, use_layer_scale=True, qkv_bias=False, qkv_scale=None, layer_scale_init_value=1e-5,
                  use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True, temporal_connection_len=1,
                  use_tcn=False, graph_only=False, neighbour_num=4, n_frames=27, with_bone=False):
    layers = []
    for _ in range(n_layers):
        if not with_bone:
            layers.append(RepeatFormerPart(dim=dim, mlp_ratio=mlp_ratio, act_layer=act_layer, attn_drop=attn_drop, drop=drop_rate,
                                           num_heads=num_heads, use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value,
                                           qkv_bias=qkv_bias, qkv_scale=qkv_scale, use_adaptive_fusion=use_adaptive_fusion,
                                           hierarchical=hierarchical, use_temporal_similarity=use_temporal_similarity,
                                           temporal_connection_len=temporal_connection_len, use_tcn=use_tcn, graph_only=graph_only,
                                           neighbour_num=neighbour_num, n_frames=n_frames))
        elif with_bone:
            layers.append(RepeatFormerPartWithBone(dim=dim, mlp_ratio=mlp_ratio, act_layer=act_layer, attn_drop=attn_drop, drop=drop_rate,
                                               num_heads=num_heads, use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value,
                                               qkv_bias=qkv_bias, qkv_scale=qkv_scale, use_adaptive_fusion=use_adaptive_fusion,
                                               hierarchical=hierarchical, use_temporal_similarity=use_temporal_similarity,
                                               temporal_connection_len=temporal_connection_len, use_tcn=use_tcn, graph_only=graph_only,
                                               neighbour_num=neighbour_num, n_frames=n_frames))
        else:
            raise NotImplementedError("unrecognized layer name")

    layers = nn.Sequential(*layers)
    return layers


def bone_decomposer(input_x: torch.Tensor) -> torch.Tensor:
    # x: [B, 27, 17, 3]
    input_x = input_x[..., :2]

    bone_child = [0,1,2, 0,4,5, 0,7,8,9, 8,11,12, 8,14,15]
    bone_parent = [1,2,3, 4,5,6, 7,8,9,10, 11,12,13, 14,15,16]
    bone_directions = input_x[:, :, bone_child] - input_x[:, :, bone_parent] # [B, 27, 17, 3]
    bone_lengths = torch.norm(bone_directions, dim=-1)
    bone_lengths = bone_lengths.unsqueeze(-1)
    bone_lengths[bone_lengths == 0] = 1
    bone_directions = bone_directions / bone_lengths

    bone_directions_mean = torch.mean(bone_directions, dim=-2, keepdim=True)
    bone_lengths_mean = torch.mean(bone_lengths, dim=-2, keepdim=True)

    bone_directions = torch.concatenate((bone_directions, bone_directions_mean), dim=-2)
    bone_lengths = torch.concatenate((bone_lengths, bone_lengths_mean), dim=-2)

    bone_info = torch.concatenate((bone_directions, bone_lengths), dim=-1)

    return bone_info


class FormerModule(nn.Module):
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., mlp_drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, layer_scale_init_value=1e-5,
                 mode='spatial', mixer_type='attention', use_temporal_similarity=True,
                 temporal_connection_len=1, neighbour_num=4, n_frames=27):
        super().__init__()
        self.mixer_type = mixer_type
        self.norm1 = nn.LayerNorm(dim)
        self.norm1_limb = nn.LayerNorm(dim)

        if mixer_type == 'attention':
            self.mixer = Attention(dim_in=dim, dim_out=dim, num_heads=num_heads,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, mode=mode)
        elif mixer_type == 'graph':
            self.mixer = GCN(dim, dim, num_nodes=17 if mode == 'spatial' else n_frames,
                             neighbour_num=neighbour_num, mode=mode, use_temporal_similarity=use_temporal_similarity,
                             temporal_connection_len=temporal_connection_len)
        # elif mixer_type == "ms-tcn":
        #     self.mixer = MultiScaleTCN(in_channels=dim, out_channels=dim)
        elif mixer_type == "bone":
            self.mixer = BoneCrossAttention(dim_in=dim, dim_out=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            attn_drop=attn_drop, mode=mode)
        else:
            raise NotImplementedError("unrecognized mixer type, check your stupid code. ")

        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=mlp_drop)

        # The following two techniques are useful to train deep GraphFormers
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x, x_limb=None):
        # x tensor [B, T, J, C]
        if self.use_layer_scale:
            if self.mixer_type == "bone":
                x = x + self.drop_path(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * self.mixer(self.norm1(x), self.norm1_limb(x_limb)))
            else:
                x = x + self.drop_path(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * self.mixer(
                    self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(self.norm2(x)))
        else:
            if self.mixer_type == "bone":
                x = x + self.drop_path(self.mixer(self.norm1(x), self.norm1_limb(x_limb)))
            else:
                x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class RepeatFormerPart(nn.Module):
    def __init__(self, dim=128, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, use_layer_scale=True, qkv_bias=False, qkv_scale=None, layer_scale_init_value=1e-5,
                 use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True, temporal_connection_len=1,
                 use_tcn=False, graph_only=False, neighbour_num=4, n_frames=27):
        super().__init__()
        self.hierarchical = hierarchical
        dim = dim // 2 if hierarchical else dim

        # Spatial Temporal Attention Branch
        self.att_spatial = FormerModule(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                            qkv_bias, qkv_scale, use_layer_scale, layer_scale_init_value,
                                            mode='spatial', mixer_type="attention",
                                            use_temporal_similarity=use_temporal_similarity,
                                            neighbour_num=neighbour_num,
                                            n_frames=n_frames)
        self.att_temporal = FormerModule(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                            qkv_bias, qkv_scale, use_layer_scale, layer_scale_init_value,
                                            mode='temporal', mixer_type="attention",
                                            use_temporal_similarity=use_temporal_similarity,
                                            neighbour_num=neighbour_num,
                                            n_frames=n_frames)


        self.graph_spatial = FormerModule(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                              qkv_bias, qkv_scale, use_layer_scale, layer_scale_init_value,
                                              mode='spatial', mixer_type='graph',
                                              use_temporal_similarity=use_temporal_similarity,
                                              temporal_connection_len=temporal_connection_len,
                                              neighbour_num=neighbour_num,
                                              n_frames=n_frames)
        self.graph_temporal = FormerModule(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                               qkv_bias, qkv_scale, use_layer_scale, layer_scale_init_value,
                                               mode='temporal', mixer_type='graph',
                                               use_temporal_similarity=use_temporal_similarity,
                                               temporal_connection_len=temporal_connection_len,
                                               neighbour_num=neighbour_num,
                                               n_frames=n_frames)


        # Spatial Temporal Bone Branch
        self.bone_spatial = FormerModule(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                             qkv_bias, qkv_scale, use_layer_scale, layer_scale_init_value,
                                             mode='spatial', mixer_type='bone',
                                             use_temporal_similarity=use_temporal_similarity,
                                             temporal_connection_len=temporal_connection_len,
                                             neighbour_num=neighbour_num,
                                             n_frames=n_frames)
        self.bone_temporal = FormerModule(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                              qkv_bias, qkv_scale, use_layer_scale, layer_scale_init_value,
                                              mode='temporal', mixer_type='bone',
                                              use_temporal_similarity=use_temporal_similarity,
                                              temporal_connection_len=temporal_connection_len,
                                              neighbour_num=neighbour_num,
                                              n_frames=n_frames)


        self.use_adaptive_fusion = use_adaptive_fusion
        self.fusion_three_channel = nn.Linear(dim * 3, 3)
        self._init_fusion_three()


    def _init_fusion_three(self):
        self.fusion_three_channel.weight.data.fill_(0)
        self.fusion_three_channel.bias.data.fill_(1 / 3)

    def forward(self, x):
        x_attn = self.att_temporal(self.att_spatial(x))
        x_graph = self.graph_temporal(self.graph_spatial(x))
        x_bone = self.bone_temporal(self.bone_spatial(x))

        if self.use_adaptive_fusion:
            alpha = torch.cat((x_attn, x_graph, x_bone), dim=-1)
            alpha = self.fusion_three_channel(alpha)
            alpha = alpha.softmax(dim=-1)
            x = x_attn * alpha[..., 0:1] + x_graph * alpha[..., 1:2] + x_bone * alpha[..., 2:3]
        else:
            x = (x_attn + x_graph + x_bone) / 3

        return x


class RepeatFormerPartWithBone(nn.Module):
    def __init__(self, dim=128, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, use_layer_scale=True, qkv_bias=False, qkv_scale=None, layer_scale_init_value=1e-5,
                 use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True, temporal_connection_len=1,
                 use_tcn=False, graph_only=False, neighbour_num=4, n_frames=27):
        super().__init__()
        self.hierarchical = hierarchical
        dim = dim // 2 if hierarchical else dim

        # Spatial Temporal Attention Branch
        self.att_spatial = FormerModule(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                            qkv_bias, qkv_scale, use_layer_scale, layer_scale_init_value,
                                            mode='spatial', mixer_type="attention",
                                            use_temporal_similarity=use_temporal_similarity,
                                            neighbour_num=neighbour_num,
                                            n_frames=n_frames)
        self.att_temporal = FormerModule(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                             qkv_bias, qkv_scale, use_layer_scale, layer_scale_init_value,
                                             mode='temporal', mixer_type="attention",
                                             use_temporal_similarity=use_temporal_similarity,
                                             neighbour_num=neighbour_num,
                                             n_frames=n_frames)


        self.graph_spatial = FormerModule(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                              qkv_bias, qkv_scale, use_layer_scale, layer_scale_init_value,
                                              mode='spatial', mixer_type='graph',
                                              use_temporal_similarity=use_temporal_similarity,
                                              temporal_connection_len=temporal_connection_len,
                                              neighbour_num=neighbour_num,
                                              n_frames=n_frames)
        self.graph_temporal = FormerModule(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                               qkv_bias, qkv_scale, use_layer_scale, layer_scale_init_value,
                                               mode='temporal', mixer_type='graph',
                                               use_temporal_similarity=use_temporal_similarity,
                                               temporal_connection_len=temporal_connection_len,
                                               neighbour_num=neighbour_num,
                                               n_frames=n_frames)


        # Spatial Temporal Bone Branch
        self.bone_spatial = FormerModule(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                             qkv_bias, qkv_scale, use_layer_scale, layer_scale_init_value,
                                             mode='spatial', mixer_type='bone',
                                             use_temporal_similarity=use_temporal_similarity,
                                             temporal_connection_len=temporal_connection_len,
                                             neighbour_num=neighbour_num,
                                             n_frames=n_frames)
        self.bone_temporal = FormerModule(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                              qkv_bias, qkv_scale, use_layer_scale, layer_scale_init_value,
                                              mode='temporal', mixer_type='bone',
                                              use_temporal_similarity=use_temporal_similarity,
                                              temporal_connection_len=temporal_connection_len,
                                              neighbour_num=neighbour_num,
                                              n_frames=n_frames)

        self.use_adaptive_fusion = use_adaptive_fusion
        self.fusion_three_channel = nn.Linear(dim * 3, 3)
        self._init_fusion_three()

    def _init_fusion_three(self):
        self.fusion_three_channel.weight.data.fill_(0)
        self.fusion_three_channel.bias.data.fill_(1 / 3)

    def forward(self, x, x_bone=None, x_limb=None):

        x_attn = self.att_temporal(self.att_spatial(x))
        x_graph = self.graph_temporal(self.graph_spatial(x))
        if x_bone is None:
            x_bone = self.bone_temporal(self.bone_spatial(x, x_limb), x_limb)
        else:
            x_bone = self.bone_temporal(self.bone_spatial(x_bone, x_limb), x_limb)


        if self.use_adaptive_fusion:
            alpha = torch.cat((x_attn, x_graph, x_bone), dim=-1)
            alpha = self.fusion_three_channel(alpha)
            alpha = alpha.softmax(dim=-1)
            x = x_attn * alpha[..., 0:1] + x_graph * alpha[..., 1:2] + x_bone * alpha[..., 2:3]
        else:
            x = (x_attn + x_graph + x_bone) / 3

        return x



class KASportsFormer(nn.Module):
    def __init__(self, n_layers=26, dim_in=3, dim_feat=128, dim_rep=512, dim_out=3, mlp_ratio=4, act_layer=nn.GELU, attn_drop=0.,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5, use_adaptive_fusion=True,
                 num_heads=4, qkv_bias=False, qkv_scale=None, hierarchical=False, num_joints=17,
                 use_temporal_similarity=True, temporal_connection_len=1, use_tcn=False, graph_only=False,
                 neighbour_num=4, n_frames=27):
        super().__init__()
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.bone_embed = nn.Linear(dim_in, dim_feat)
        self.limb_embed = nn.Linear(dim_in, dim_feat)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.bone_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.limb_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.norm = nn.LayerNorm(dim_feat)
        self.bone_refusion = BoneRefusion()


        self.layers_with_bone = create_layers(dim=dim_feat, n_layers=n_layers, mlp_ratio=mlp_ratio, act_layer=act_layer, attn_drop=attn_drop,
                                    drop_rate=drop, drop_path_rate=drop_path, num_heads=num_heads, use_layer_scale=use_layer_scale,
                                    qkv_bias=qkv_bias, qkv_scale=qkv_scale, layer_scale_init_value=layer_scale_init_value, use_adaptive_fusion=use_adaptive_fusion,
                                    hierarchical=hierarchical, use_temporal_similarity=use_temporal_similarity, temporal_connection_len=temporal_connection_len,
                                    use_tcn=use_tcn, graph_only=graph_only, neighbour_num=neighbour_num, n_frames=n_frames, with_bone=True)

        self.rep_logit = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.Tanh())
        ]))

        self.head = nn.Linear(dim_rep, dim_out)

    def forward(self, x, return_rep=False):
        # x [B, T, J, C] (1, 27, 17, 3)
        # x_bone [B, T, J, C] (1, 27, 17, 3)
        x_bone = bone_decomposer(x)
        x_limb = self.bone_refusion(x)
        x = self.joints_embed(x)
        x = x + self.pos_embed
        x_bone = self.bone_embed(x_bone)
        x_bone = x_bone + self.bone_pos_embed
        x_limb = self.limb_embed(x_limb)
        x_limb = x_limb + self.limb_pos_embed

        for layer_idx, layer in enumerate(self.layers_with_bone):
            if layer_idx > 0:
                x = layer(x=x, x_limb=x_limb)
            else:
                x = layer(x=x, x_bone=x_bone, x_limb=x_limb)


        x = self.norm(x)
        x = self.rep_logit(x)

        if return_rep:
            return x

        x = self.head(x)

        return x


def main() -> None:
    # temp = nn.Parameter(torch.ones(1, 17, 128))
    # temp2 = torch.zeros(1, 27, 17, 128)
    # print(temp.shape)
    # print(temp2)

    # templayer = nn.Linear(128, 3)
    # y = templayer(temp)
    # print(y.shape)
    #
    # y = y.softmax(dim=-1)
    # print(y.shape)
    #
    # templayer2 = nn.Identity()
    # y = templayer2(y)
    # print(y.shape)
    #
    # templayer3 = nn.Parameter(1e-5 * torch.ones(512), requires_grad=True)
    # templayer3 = templayer3.unsqueeze(0).unsqueeze(0)
    # print(templayer3.shape)
    #
    # dummy = torch.randn((128, 27, 17, 3))
    #
    # res = bone_decomposer(dummy)
    # print(res.shape)


    b, c, t, j = 8, 3, 27, 17
    random_x = torch.randn((b, t, j, c)).to('cuda')
    model = KASportsFormer().to('cuda')
    model.eval()

    res = model(random_x)
    print(res.shape)



if __name__ == '__main__':
    main()

