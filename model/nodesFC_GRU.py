import torch
from torch import nn
from model.cells import GRUCell
from torch.nn import Sequential, Linear, Sigmoid


class nodesFC_GRU(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device):
        super(nodesFC_GRU, self).__init__()
        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.hid_dim = 32
        self.out_dim = 1
        self.graph_mlp_out = 1
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)
        self.gru_cell = GRUCell(self.in_dim + self.graph_mlp_out, self.hid_dim)
        self.graph_mlp = Sequential(Linear(self.city_num * self.in_dim, self.city_num * self.graph_mlp_out),
                                   Sigmoid())

    def forward(self, pm25_hist, feature):
        pm25_pred = []
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        xn = pm25_hist[:, -1]
        for i in range(self.pred_len):
            x = torch.cat((xn, feature[:, self.hist_len+i]), dim=-1)
            # nodes FC
            xn_gnn = x
            xn_gnn = xn_gnn.contiguous()
            xn_gnn = xn_gnn.view(self.batch_size, -1)
            xn_gnn = self.graph_mlp(xn_gnn)
            xn_gnn = xn_gnn.view(self.batch_size, self.city_num, 1)
            x = torch.cat([xn_gnn, x], dim=-1)
            # nodes FC
            hn = self.gru_cell(x, hn)
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)
            pm25_pred.append(xn)

        pm25_pred = torch.stack(pm25_pred, dim=1)
        return pm25_pred
