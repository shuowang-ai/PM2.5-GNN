import torch
from torch import nn
from torch.nn import Sequential, Linear, Sigmoid


class MLP(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim):
        super(MLP, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.in_dim = in_dim
        self.hid_dim = 16
        self.out_dim = 1
        self.graph_mlp_out = 1
        self.graph_mlp_hid = 1
        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)
        self.mlp = Sequential(Linear(self.hid_dim, self.hid_dim),
                                   Sigmoid(),
                                    Linear(self.hid_dim, self.hid_dim),
                                    Sigmoid()
                                    )

    def forward(self, pm25_hist, feature):
        pm25_pred = []
        xn = pm25_hist[:, -1]
        for i in range(self.pred_len):
            x = torch.cat((xn, feature[:, self.hist_len+i]), dim=-1)
            x = self.fc_in(x)
            x = self.mlp(x)
            xn = self.fc_out(x)
            pm25_pred.append(xn)
        pm25_pred = torch.stack(pm25_pred, dim=1)

        return pm25_pred
