import torch
import sys
sys.path.append("..")
# sys.path.append(".")
from model.MotionAGFormer import MotionAGFormer
from model.KTPFormer import KTPFormer
from model.MixSTE import MixSTE2
from model.DSTFormer import DSTformer
from model.STCFormer import Model as STCFormer
from model.HDFormer.vertex_model import Model as HDFormer
from model.HDFormer.skeleton import get_skeleton
from model.diffusionpose import D3DP
from model.KASportsFormer import KASportsFormer
from functools import partial
from torch import nn
# from torchstat import stat
# from torchsummary import summary
# from thop import profile
from torchprofile import profile_macs
import numpy as np
import scipy.sparse as sp





def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx)  # + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)

    adj_mx = adj_mx * (1 - torch.eye(adj_mx.shape[0])) + torch.eye(adj_mx.shape[0])

    return adj_mx



def adj_mx_from_skeleton_temporal(num_frame, parents):
    num_joints = num_frame
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), parents)))
    return adj_mx_from_edges(num_joints, edges, sparse=False)


def adj_mx_from_skeleton(num_joints):
    # num_joints = skeleton.num_joints()
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]))))
    return adj_mx_from_edges(num_joints, edges, sparse=False)



def load_model(args) -> torch.nn.Module:
    act_mapper = {
        "gelu": nn.GELU,
        'relu': nn.ReLU
    }

    if args.model_name == "KASportsFormer":
        model = KASportsFormer(n_layers=args.n_layers, dim_in=args.dim_in, dim_feat=args.dim_feat, dim_rep=args.dim_rep,
                                 dim_out=args.dim_out, mlp_ratio=args.mlp_ratio, act_layer=act_mapper[args.act_layer], attn_drop=args.attn_drop,
                                 drop=args.drop, drop_path=args.drop_path, use_layer_scale=args.use_layer_scale, layer_scale_init_value=args.layer_scale_init_value,
                                 use_adaptive_fusion=args.use_adaptive_fusion, num_heads=args.num_heads, qkv_bias=args.qkv_bias, qkv_scale=args.qkv_scale,
                                 hierarchical=args.hierarchical, num_joints=args.num_joints, use_temporal_similarity=args.use_temporal_similarity,
                                 temporal_connection_len=args.temporal_connection_len, use_tcn=args.use_tcn, graph_only=args.graph_only, neighbour_num=args.neighbour_num,
                                 n_frames=args.n_frames)
    else:
        raise Exception("Unexpected model name")

    return model



def total_parameters_count(model: torch.nn.Module):
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    return model_params




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="../configs/sportspose-kasportsformer.yaml", help="path to the config file")

    cli_args = parser.parse_args()
    from utils.utilities import yaml_config_reader
    args_dict = yaml_config_reader(cli_args.config_path)
    # print(args_dict.norm_layer)
    model = load_model(args_dict)


    b, f, n, c = 1, 27, 17, 3
    random_x = torch.randn((b, f, n, c)).to('cuda')
    # random_x2 = torch.randn((b, f, n ,3)).to('cuda')
    # random_x = torch.randn(b, c, f, n).to('cuda')
    model = model.to('cuda')
    # random_x = torch.randn(b, f, n, c).to('CPU')
    

    model.eval()

    print("input: ")
    print(random_x.shape)
    print("output: ")
    out = model(random_x)
    print(out.shape)

    for _ in range(10):
        _ = model(random_x)


    params_count = total_parameters_count(model)
    print(f'The total parameter numbers of this model: {params_count:,}')


    print(f"Model FLOPS #: {profile_macs(model, (random_x)):,}")

    import time
    num_iterations = 100
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(random_x)
    end_time = time.time()

    averate_inference_time = (end_time - start_time) / num_iterations
    fps = 1.0 / averate_inference_time
    print(f"FPS: {fps}")














