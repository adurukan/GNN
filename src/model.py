import torch
from torch.nn import Linear, LayerNorm, ReLU, Dropout, Parameter
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import yaml
import os

params = yaml.safe_load(open("params.yaml"))["model"]


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, dropout, use_skip=False):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0], 2)
        self.dropout = dropout
        self.use_skip = use_skip
        if self.use_skip:
            self.weight = torch.nn.init.xavier_normal_(
                Parameter(torch.Tensor(num_node_features, 2))
            )

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, data.edge_index)
        if self.use_skip:
            x = F.softmax(x + torch.matmul(data.x, self.weight), dim=-1)
        else:
            x = F.softmax(x, dim=-1)
        return x

    def embed(self, data):
        x = self.conv1(data.x, data.edge_index)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: \n {device}")
model = GCN(
    params["num_node_features"], params["hidden_channels"], params["dropout"]
).to(device)
# model.double()
