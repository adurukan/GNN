import torch
from torch.nn import Linear, LayerNorm, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import (
    ChebConv,
    NNConv,
    DeepGCNLayer,
    GATConv,
    DenseGCNConv,
    GCNConv,
    GraphConv,
)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = GCNConv(165, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(64, 64)
        self.conv4 = GCNConv(128, 1)

    def forward(self, data, adj=None):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv4(x, edge_index)

        return torch.sigmoid(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: \n {device}")
model = Net().to(device)
model.double()
