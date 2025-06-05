import torch
from torch import nn
from torch.nn import Parameter
from AE import AE
from GATLayer import GATLayer
from opt import args

class DGCSG(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2,  ae_n_dec_1, ae_n_dec_2,
                  hidden_size1, hidden_size2, alpha,
                 n_input, n_z, n_clusters, v):
        super(DGCSG, self).__init__()

        self.ae = AE(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            n_input=n_input,
            n_z=n_z)

        ae_pre = './ae_pretrain/{}.pkl'.format(args.name)
        self.ae.load_state_dict(torch.load(ae_pre, map_location='cpu'),strict=False)
        print('Loading AE pretrain model:', ae_pre)

        self.gate_1 = GATLayer(n_input, hidden_size1,alpha)
        self.gate_2 = GATLayer(hidden_size1, hidden_size2,alpha)
        self.gate_3 = GATLayer(hidden_size2, n_z,alpha)
        self.a = 0.5
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.s = nn.Sigmoid()
        self.v = v
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, adj):
        z_ae, x_bar, enc_h1, enc_h2 = self.ae(x)
        gate_enc1 ,lossd0= self.gate_1(x, adj)
        gate_enc2 ,lossd1= self.gate_2((1 - self.a) * gate_enc1 + self.a * enc_h1, adj)
        z_gate ,lossd2= self.gate_3((1 - self.a) * gate_enc2 + self.a * enc_h2, adj)
        z_i = (1 - self.a) * z_gate + self.a * z_ae

        z_l = torch.spmm(adj, z_i)


        adj_hat = self.s(torch.matmul(z_gate, z_gate.t()))

        q = 1.0 / (1.0 + torch.sum(torch.pow(z_l.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        q1 = 1.0 / (1.0 + torch.sum(torch.pow(z_ae.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q1 = q1.pow((self.v + 1.0) / 2.0)
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        total_lossd = (lossd0 + lossd1 + lossd2)

        return x_bar,  adj_hat, z_ae, q, q1, z_l,total_lossd
