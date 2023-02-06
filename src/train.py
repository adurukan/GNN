import numpy as np
import pandas as pd
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
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import scipy.sparse as sp
import yaml
import os
import json
from model import model, device

params = yaml.safe_load(open("params.yaml"))["train"]
data_train, train_idx, valid_idx, X_train, X_valid, y_train = (
    torch.load(params["data_train"]),
    torch.load(params["train_idx"]),
    torch.load(params["valid_idx"]),
    torch.load(params["X_train"]),
    torch.load(params["X_valid"]),
    torch.load(params["y_train"]),
)

data_train = data_train.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
criterion = torch.nn.BCELoss()

training_dict = []
model.train()
for epoch in range(70):
    optimizer.zero_grad()
    out = model(data_train)
    out = out.reshape((data_train.x.shape[0]))
    loss = criterion(out[train_idx], data_train.y[train_idx])
    auc = roc_auc_score(
        data_train.y.detach().cpu().numpy()[train_idx],
        out.detach().cpu().numpy()[train_idx],
    )
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        training_dict.append({"epoch": epoch, "loss": loss.item(), "roc": auc})

with open("training_loss_acc.json", "w") as f:
    json.dump(
        {"result": {"loss": i["loss"], "roc": i["roc"]} for i in training_dict},
        f,
        indent=4,
    )
os.makedirs(os.path.join("src", "model"), exist_ok=True)
torch.save(model.state_dict(), "src/model/model_state.pt")
