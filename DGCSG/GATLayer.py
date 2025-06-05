import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from opt import args

class GATLayer(nn.Module):

    def __init__(self, in_features, out_features, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        if isinstance(input, np.ndarray):
            input = torch.tensor(input, dtype=torch.float32)
        if isinstance(adj, np.ndarray):
            adj = torch.tensor(adj, dtype=torch.float32)
        h = torch.mm(input, self.W)

        attn_for_self = torch.mm(h, self.a_self)  # (N,1)
        attn_for_neighs = torch.mm(h, self.a_neighs)  # (N,1)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)
        attn_dense = self.leakyrelu(attn_dense)  # (N,N)
        L=normalized_adjacency_matrix(adj)
        adj_dense = adj
        zero_vec = -9e15 * torch.ones_like(adj_dense)
        adj_dense = torch.where(adj_dense > 0, attn_dense, zero_vec)
        attention = F.softmax(adj_dense, dim=1)
        h_prime = torch.matmul(attention, h)

        lossd = _attn_s_smoothing_ortho(h_prime,h_prime,L)
        return F.elu(h_prime),lossd

def compute_degree_matrix(adj):
    degree_values = torch.sum(adj, dim=1)
    degree_matrix = torch.diag(degree_values)
    return degree_matrix

def normalized_adjacency_matrix(adj):
    D = compute_degree_matrix(adj)
    D_inv_sqrt = torch.diag(torch.pow(torch.clamp(D.diag(), min=1e-10), -0.5))
    normalized_adj = torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
    return normalized_adj
def _attn_s_smoothing(features: torch.Tensor, L: torch.Tensor,n_nodes) -> torch.Tensor:
    tmp = torch.matmul(L,features)
    trace = torch.einsum('ij,ij->i', features, tmp)
    return trace.sum() / (n_nodes*args.n_clusters)

def _attn_s_smoothing_ortho(z: torch.Tensor,features,L) -> torch.Tensor:
    n_nodes, n_dim = z.size()
    smooth_loss = _attn_s_smoothing(features,L,n_nodes)
    ortho_loss = torch.sum((torch.mm(z.T, z)/ n_nodes  - torch.eye(n_dim, device=z.device)) ** 2)
    return  ortho_loss-smooth_loss


