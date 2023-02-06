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

params = yaml.safe_load(open("params.yaml"))["evaluate"]
data_train, train_idx, valid_idx = (
    torch.load(params["data_train"]),
    torch.load(params["train_idx"]),
    torch.load(params["valid_idx"]),
)

model.load_state_dict(torch.load("src/model/model_state.pt"))
model.eval()

preds = model(data_train)
preds = preds.detach().cpu().numpy()
