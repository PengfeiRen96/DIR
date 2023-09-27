from __future__ import division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PGraphConv(nn.Module):
    """
    High-order graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(PGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, adj.size(0), in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj_0 = torch.eye(adj.size(0), dtype=torch.float)  # declare self-connections
        self.m_0 = (self.adj_0 > 0)
        self.e_0 = nn.Parameter(torch.zeros(1, len(self.m_0.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_0.data, 1)

        self.adj_1 = adj  # one_hop neighbors
        self.m_1 = (self.adj_1 > 0)
        self.e_1 = nn.Parameter(torch.zeros(1, len(self.m_1.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_1.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(1))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input.unsqueeze(-2), self.W[0]).squeeze(-2)
        h1 = torch.matmul(input.unsqueeze(-2), self.W[1]).squeeze(-2)

        A_0 = -9e15 * torch.ones_like(self.adj_0).to(input.device)
        A_1 = -9e15 * torch.ones_like(self.adj_1).to(input.device)  # without self-connection

        A_0[self.m_0] = self.e_0
        A_1[self.m_1] = self.e_1

        A_0 = F.softmax(A_0, dim=1)
        A_1 = F.softmax(A_1, dim=1)

        output_0 = torch.matmul(A_0, h0)
        output_1 = torch.matmul(A_1, h1)

        output = output_0 + output_1

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'