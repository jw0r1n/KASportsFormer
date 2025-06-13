import math
import torch
from torch import nn

"""
    :param dim_in: Channel input dimension
    :param dim_out: Channel output dimension
    :param num_nodes: Number of nodes
    :param neighbour_num: Neighbor numbers. Used in temporal GCN to create edges
    :param mode: Either 'spatial' or 'temporal'
    :param use_temporal_similarity: If true, for temporal GCN uses top-k similarity between nodes
    :param temporal_connection_len: Connects joint to itself within next `temporal_connection_len` frames
    :param connections: Spatial connections for graph edges (Optional)
"""

CONNECTIONS = {10: [9], 9: [8, 10], 8: [7, 9, 11, 14], 14: [15, 8], 15: [16, 14], 11: [12, 8], 12: [13, 11],
               7: [0, 8], 0: [1, 7, 4], 1: [2, 0], 2: [3, 1], 4: [5, 0], 5: [6, 4], 16: [15], 13: [12], 3: [2], 6: [5]}

class GCN(nn.Module):
    # GCN部分的参数不是很多，所以处理的效果可能不是很好
    def __init__(self, dim_in, dim_out, num_nodes, neighbour_num=4, mode='spatial', use_temporal_similarity=True,
                 temporal_connection_len=1, connections=None):
        super().__init__()
        # self.nodes_ = ""
        assert mode in ['spatial', 'temporal'], "Mode is undefined"
        self.relu = nn.ReLU()
        self.neighbour_num = neighbour_num
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.mode = mode
        self.use_temporal_similarity = use_temporal_similarity
        self.num_nodes = num_nodes
        self.connections = connections

        self.U = nn.Linear(self.dim_in, self.dim_out)
        self.V = nn.Linear(self.dim_in, self.dim_out)
        self.batch_norm = nn.BatchNorm1d(self.num_nodes)

        self._init_gcn()

        if mode == 'spatial':
            self.adj = self._init_spatial_adj()
        elif mode == 'temporal' and not self.use_temporal_similarity:
            self.adj = self._init_temporal_adj(temporal_connection_len)

    def _init_gcn(self):  # 这个init的作用不是很明确
        self.U.weight.data.normal_(0, math.sqrt(2. / self.dim_in))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.dim_in))
        self.batch_norm.weight.data.fill_(1)
        self.batch_norm.bias.data.zero_()

    def _init_spatial_adj(self): # spatial
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        connections = self.connections if self.connections is not None else CONNECTIONS

        for i in range(self.num_nodes):  # 对的,这是个字典，取的是键，python的语法还是不太熟悉
            connected_nodes = connections[i]
            for j in connected_nodes:
                adj[i, j] = 1

        return adj

    def _init_temporal_adj(self, connection_length): # temporal 但是实际上没有用到这边构建的这个矩阵
        # 构建 temporal 矩阵
        # connects each joint to itself and the same joint within next connection_length frames
        # 有没有可能这个不是num nodes来决定的，frame_length能不能用来决定，我主要是不确定这个adj能不能不是方阵
        adj = torch.zeros((self.num_nodes, self.num_nodes))

        for i in range(self.num_nodes):
            try:
                for j in range(connection_length + 1):
                    adj[i, i + j] = 1
            except IndexError:
                pass
        return adj

    @staticmethod
    def normalize_digraph(adj):
        b, n, c = adj.shape  # 这个地方写的不好，c只能和n一样大

        node_degrees = adj.detach().sum(dim=-1)  # (B, N)
        deg_inv_sqrt = node_degrees ** -0.5  # 类似于Attention里面做规范化因子，保持graph分支和attention分支的值相差不会太大
        norm_deg_matrix = torch.eye(n)
        dev = adj.get_device()
        if dev >= 0:
            norm_deg_matrix = norm_deg_matrix.to(dev)
        norm_deg_matrix = norm_deg_matrix.view(1, n, n) * deg_inv_sqrt.view(b, n, 1)  # 这边用的是广播机制的乘法 结果(b, n, n)
        norm_adj = torch.bmm(torch.bmm(norm_deg_matrix, adj), norm_deg_matrix)  # -> (b, n, c) -> 若c = n, 输出 (B, J, J)

        return norm_adj

    def change_adj_device_to_cuda(self, adj):
        dev = self.V.weight.get_device()
        if dev >= 0 and adj.get_device() < 0:
            adj = adj.to(dev)
        return adj


    def forward(self, x):
        # tensor x: B, T, J, C

        b, t, j, c = x.shape

        if self.mode == 'temporal':
            x = x.transpose(1, 2)  # (B, T, J, C) -> (B, J, T, C)
            x = x.reshape(-1, t, c)
            if self.use_temporal_similarity:  # 在这里会构建一个新的矩阵
                similarity = x @ x.transpose(1, 2)  # (B * J, T, C) @ (B * J, C, T) -> (B * J, T, T) 只剩T在这边
                threshold = similarity.topk(k=self.neighbour_num, dim=-1, largest=True)[0][..., -1].view(b * j, t, 1)
                # topk (B * J, 27, 4)  选出第4小的作为阈值 (B * J, 27)
                adj = (similarity >= threshold).float() # (B * J, T, T)
                adj = self.change_adj_device_to_cuda(adj)
            else:
                adj = self.adj
                adj = self.change_adj_device_to_cuda(adj)
                adj = adj.repeat(b * j, 1, 1)
        elif self.mode == 'spatial':
            x = x.reshape(-1, j, c)  # x 的 (B, T) 融合成同一个维度
            adj = self.adj
            adj = self.change_adj_device_to_cuda(adj)
            adj = adj.repeat(b * t, 1, 1)  # 把 adj 也重复成BT融合的维度, (B*T, 17, 17)
        else:
            raise NotImplementedError(self.mode)

        norm_adj = self.normalize_digraph(adj)
        aggregate = norm_adj @ self.V(x)  # (B * T, J, J) @ (B * T, J, C) -> (B * T, J, C)

        if self.dim_in == self.dim_out:
            x = self.relu(x + self.batch_norm(aggregate + self.U(x)))
        else:
            x = self.relu(self.batch_norm(aggregate + self.U(x)))  # 取消了残差连接

        x = x.reshape(-1, t, j, self.dim_out) if self.mode == 'spatial' else x.reshape(-1, j, t, self.dim_out).transpose(1, 2)
        return x





if __name__ == '__main__':
    # for i in range(17):
    #     connected_nodes = CONNECTIONS[i]
    #     print(i)
    #     print(connected_nodes)
    #     for j in connected_nodes:
    #         print(j)

    dummy_x = nn.Parameter(torch.rand((128, 27, 17, 512))).cuda()
    dev = dummy_x.get_device()
    print(dev)


    dummy_x2 = torch.rand((128, 17, 512))
    node_degrees = dummy_x2.detach().sum(dim=-1)
    deg_inv_sqrt = node_degrees = node_degrees ** -0.5 # (128, 17)
    norm_deg_matrix = torch.eye(17)
    temp1 = norm_deg_matrix.view(1, 17, 17)
    temp2 = deg_inv_sqrt.view(128, 17, 1) # (128, 17, 17)
    multitemp = temp1 * temp2
    temp3 = multitemp @ dummy_x2
    # temp3 = torch.bmm(multitemp, dummy_x2)
    # temp4 = torch.bmm(temp3, multitemp)



    print(node_degrees.shape)
    print(deg_inv_sqrt.shape)
    print(norm_deg_matrix.shape)
    print(temp1.shape)
    print(temp2.shape)
    print(multitemp.shape)
    print(temp3.shape)
    # print(temp4.shape)


    dummy_adj = torch.rand((17, 17))
    print(dummy_adj.shape)
    dummy_adj = dummy_adj.repeat(128 * 27, 1, 1)
    print(dummy_adj.shape)


    temp1 = torch.rand((128, 17, 17))
    temp2 = torch.rand((128 * 27, 17, 17))
    temp3 = torch.rand((27, 17, 17)) @ torch.rand((27, 17, 64))
    print(temp3.shape)

    temp1 = torch.rand((2176, 27, 512))
    temp2 = torch.rand((2176, 512, 27))
    temp3 = temp1 @ temp2
    # print(temp3.shape)
    # threshold = temp3.topk(k=4, dim=-1, largest=True)[0][..., -1]
    threshold = temp3.topk(k=4, dim=-1, largest=True)[0][..., -1].view(2176, 27, 1)
    print(threshold.shape)
    # print(threhold.shape)
    res = (temp3 >= threshold)
    print(res.shape)


    templayer = GCN(512, 512, 17)
    dummy_input = torch.rand((128, 27, 17, 512))
    dummy_output = templayer(dummy_input)
    print(dummy_output.shape)






