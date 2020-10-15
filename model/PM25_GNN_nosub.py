import torch
from torch import nn
from model.cells import GRUCell
from torch.nn import Sequential, Linear, Sigmoid
import numpy as np
from torch_scatter import scatter_add, scatter_sub
from torch.nn import functional as F
from torch.nn import Parameter


class GraphGNN(nn.Module):
    def __init__(self, device, edge_index, edge_attr, in_dim, out_dim, wind_mean, wind_std):
        super(GraphGNN, self).__init__()
        self.device = device
        self.edge_index = torch.LongTensor(edge_index).to(self.device)
        self.edge_attr = torch.Tensor(np.float32(edge_attr))
        self.edge_attr_norm = (self.edge_attr - self.edge_attr.mean(dim=0)) / self.edge_attr.std(dim=0)
        self.w = Parameter(torch.rand([1]))
        self.b = Parameter(torch.rand([1]))
        self.wind_mean = torch.Tensor(np.float32(wind_mean)).to(self.device)
        self.wind_std = torch.Tensor(np.float32(wind_std)).to(self.device)
        e_h = 32
        e_out = 30
        n_out = out_dim
        self.edge_mlp = Sequential(Linear(in_dim * 2 + 2 + 1, e_h),
                                   Sigmoid(),
                                   Linear(e_h, e_out),
                                   Sigmoid(),
                                   )
        self.node_mlp = Sequential(Linear(e_out, n_out),
                                   Sigmoid(),
                                   )

    def forward(self, x):
        self.edge_index = self.edge_index.to(self.device)
        self.edge_attr = self.edge_attr.to(self.device)
        self.w = self.w.to(self.device)
        self.b = self.b.to(self.device)

        edge_src, edge_target = self.edge_index
        node_src = x[:, edge_src]
        node_target = x[:, edge_target]

        src_wind = node_src[:,:,-2:] * self.wind_std[None,None,:] + self.wind_mean[None,None,:]
        src_wind_speed = src_wind[:, :, 0]
        src_wind_direc = src_wind[:,:,1]
        self.edge_attr_ = self.edge_attr[None, :, :].repeat(node_src.size(0), 1, 1)
        city_dist = self.edge_attr_[:,:,0]
        city_direc = self.edge_attr_[:,:,1]

        theta = torch.abs(city_direc - src_wind_direc)
        edge_weight = F.relu(3 * src_wind_speed * torch.cos(theta) / city_dist)
        edge_weight = edge_weight.to(self.device)
        edge_attr_norm = self.edge_attr_norm[None, :, :].repeat(node_src.size(0), 1, 1).to(self.device)
        out = torch.cat([node_src, node_target, edge_attr_norm, edge_weight[:,:,None]], dim=-1)

        out = self.edge_mlp(out)
        out_add = scatter_add(out, edge_target, dim=1, dim_size=x.size(1))
        #out_sub = scatter_sub(out, edge_src, dim=1, dim_size=x.size(1))

        out = out_add# + out_sub
        out = self.node_mlp(out)

        return out


class PM25_GNN_nosub(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, edge_index, edge_attr, wind_mean, wind_std):
        super(PM25_GNN_nosub, self).__init__()

        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size

        self.in_dim = in_dim
        self.hid_dim = 64
        self.out_dim = 1
        self.gnn_out = 13

        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.graph_gnn = GraphGNN(self.device, edge_index, edge_attr, self.in_dim, self.gnn_out, wind_mean, wind_std)
        self.gru_cell = GRUCell(self.in_dim + self.gnn_out, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, pm25_hist, feature):
        pm25_pred = []
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        xn = pm25_hist[:, -1]
        for i in range(self.pred_len):
            x = torch.cat((xn, feature[:, self.hist_len + i]), dim=-1)

            xn_gnn = x
            xn_gnn = xn_gnn.contiguous()
            xn_gnn = self.graph_gnn(xn_gnn)
            x = torch.cat([xn_gnn, x], dim=-1)

            hn = self.gru_cell(x, hn)
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)
            pm25_pred.append(xn)

        pm25_pred = torch.stack(pm25_pred, dim=1)

        return pm25_pred
