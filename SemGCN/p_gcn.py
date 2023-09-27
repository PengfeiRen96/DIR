from __future__ import absolute_import
import torch
import torch.nn as nn
from .p_graph_conv import PGraphConv


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = PGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class SimplePGCN(nn.Module):
    def __init__(self, adj, in_dim, hidden_dim, out_dim, num_layers=4):
        super(SimplePGCN, self).__init__()

        _gconv_layers = []
        for i in range(num_layers):
            _gconv_layers.append(_GraphConv(adj, hidden_dim, hidden_dim))

        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_in = _GraphConv(adj, in_dim, hidden_dim)
        self.gconv_out = _GraphConv(adj, hidden_dim, out_dim)

    def forward(self, x):
        out = self.gconv_in(x)
        out = self.gconv_layers(out)
        out = self.gconv_out(out)
        return out


class ResSimplePGCN(nn.Module):
    def __init__(self, adj, hidden_dim, num_layers=4):
        super(ResSimplePGCN, self).__init__()
        _gconv_layers = []
        for i in range(num_layers):
            _gconv_layers.append(_GraphConv(adj, hidden_dim, hidden_dim))
        self.gconv_layers = nn.Sequential(*_gconv_layers)

    def forward(self, x):
        out = self.gconv_layers(x)
        return out


class SimplePoolGCN(nn.Module):
    def __init__(self, adjs, node_maps, in_dim, hid_dim, out_dim, p_dropout=None):
        super(SimplePoolGCN, self).__init__()
        self.in_dim = in_dim
        self.node_maps = node_maps
        self.node_mats = [self.map2mat(node_map) for node_map in node_maps]

        self.gconv_layers_in_0 = _GraphConv(adjs[0], in_dim, hid_dim[0], p_dropout)
        self.gconv_layers_in_1 = _GraphConv(adjs[1], hid_dim[0], hid_dim[1], p_dropout)
        self.gconv_layers_in_2 = _GraphConv(adjs[2], hid_dim[1], hid_dim[2], p_dropout)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mean_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hid_dim[2], out_dim)

        self.ReLU = nn.ReLU()

    def forward(self, x):
        c0 = self.gconv_layers_in_0(x)
        c0_d = self.pool(c0, self.node_maps[0], 'mean')

        c1 = self.gconv_layers_in_1(c0_d)
        c1_d = self.pool(c1, self.node_maps[1], 'mean')

        c2 = self.gconv_layers_in_2(c1_d)
        c2_d = self.pool(c2, self.node_maps[2], 'mean')

        out = self.fc(c2_d).squeeze(1)
        return out

    # x: B X N X C
    def pool(self, x, pool_map, pool_type):
        out = []
        for joint_index in pool_map:
            if pool_type == 'mean':
                out.append(self.mean_pool(x[:, joint_index, :].permute(0, 2, 1)).squeeze(-1))
            else:
                out.append(self.max_pool(x[:, joint_index, :].permute(0, 2, 1)).squeeze(-1))
        return torch.stack(out, dim=1)

    def unpool(self, x, unpool_mat):
        return torch.matmul(x.permute(0, 2, 1), unpool_mat).permute(0, 2, 1)

    def map2mat(self, pool_map):
        M = len(pool_map)
        N = 0
        for node in pool_map:
            N = max(N, max(node)+1)
        mat = torch.zeros([M, N])
        for index, node in enumerate(pool_map):
            mat[index, [node]] = 1
        return mat

    def _get_decoder_input(self, X_e, X_d):
        return X_e+X_d