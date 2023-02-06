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

params = yaml.safe_load(open("params.yaml"))["generate"]


def get_dfs(params):
    df_features = pd.read_csv(params["path_fetures"], index_col=0)
    df_edges = pd.read_csv(params["path_edges"], index_col=0)
    df_classes = pd.read_csv(params["path_classes"], index_col=0)
    return df_features, df_edges, df_classes


# def export_df(df_features, df_edges, df_classes, df_features_classes):
#     os.makedirs(os.path.join("src", "generated"), exist_ok=True)
#     df_features.to_csv("src/generated/df_features.csv")
#     df_edges.to_csv("src/generated/df_edges.csv")
#     df_classes.to_csv("src/generated/df_classes.csv")
#     df_features_classes.to_csv("src/generated/df_features_classes.csv")


df_features, df_edges, df_classes = get_dfs(params)

df_features_classes = pd.merge(
    df_features, df_classes, how="left", left_on="txId", right_on="txId"
)

df_features_classes.sort_values("txId").reset_index(drop=True)

classified = df_features_classes.loc[
    df_features_classes["class"].loc[df_features_classes["class"] != 2].index
].drop("txId", axis=1)

unclassified = df_features_classes.loc[
    df_features_classes["class"].loc[df_features_classes["class"] == 2].index
].drop("txId", axis=1)

nodes = df_features_classes["txId"].values
map_id = {j: int(i) for i, j in enumerate(nodes)}

edges = df_edges.copy()
edges.txId1 = edges.txId1.map(map_id)
edges.txId2 = edges.txId2.map(map_id)
edge_index = np.array(edges.values).T
edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
weights = torch.tensor([1] * edge_index.shape[1], dtype=torch.double)

node_features = df_features_classes.copy()

node_features["txId"] = node_features["txId"].map(map_id)

classified_idx = node_features["class"].loc[node_features["class"] != 2].index
unclassified_idx = node_features["class"].loc[node_features["class"] == 2].index
node_features["class"] = node_features["class"].replace(2, 0)

labels = node_features["class"].values
labels = torch.tensor(labels, dtype=torch.double)
node_features = torch.tensor(
    np.array(
        node_features.drop(["Time step", "class", "txId"], axis=1).values,
        dtype=np.double,
    ),
    dtype=torch.double,
)
print(f"node_features: \n {node_features.shape}")
print(f"edge_index: \n {edge_index.shape}")
print(f"weights: \n {weights.shape}")
print(f"labels: \n {labels.shape}")
data_train = Data(
    x=node_features,
    edge_index=edge_index,
    edge_attr=weights,
    y=torch.tensor(labels, dtype=torch.double),
)
y_train = labels[classified_idx]

X_train, X_valid, y_train, y_valid, train_idx, valid_idx = train_test_split(
    node_features[classified_idx],
    y_train,
    classified_idx,
    test_size=0.15,
    random_state=42,
    stratify=y_train,
)

os.makedirs(os.path.join("src", "generated"), exist_ok=True)
torch.save(data_train, "src/generated/data_train.pt")
torch.save(X_train, "src/generated/X_train.pt")
torch.save(X_valid, "src/generated/X_valid.pt")
torch.save(y_train, "src/generated/y_train.pt")
torch.save(train_idx, "src/generated/train_idx.pt")
torch.save(valid_idx, "src/generated/valid_idx.pt")
# export_df(df_features, df_edges, df_classes, df_features_classes)
