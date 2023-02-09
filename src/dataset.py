import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.nn import Parameter
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

params = yaml.safe_load(open("params.yaml"))["dataset"]

df_class = pd.read_csv(params["path_classes"])
df_features = pd.read_csv(params["path_fetures"])
df_edge_time_fin = pd.read_csv(params["path_edgelist_timed"])
node_label = pd.read_csv(params["path_node_label"])
df_node_time = pd.read_csv(params["path_nodetime"])
merged_nodes_df = pd.read_csv(params["path_merged_nodes"])

print(f"Shape of df_class: \n {df_class.shape} \n df_class: \n {df_class.head(3)}")
print(
    f"Shape of df_features: \n {df_features.shape} \n df_features: \n {df_features.head(3)}"
)
print(
    f"Shape of df_edge_time_fin: \n {df_edge_time_fin.shape} \n df_edge_time_fin: \n {df_edge_time_fin.head(3)}"
)
print(
    f"Shape of node_label: \n {node_label.shape} \n node_label: \n {node_label.head(3)}"
)
print(
    f"Shape of df_node_time: \n {df_node_time.shape} \n df_node_time: \n {df_node_time.head(3)}"
)
print(
    f"Shape of merged_nodes_df: \n {merged_nodes_df.shape} \n merged_nodes_df: \n {merged_nodes_df.head(3)}"
)

train_dataset = []
test_dataset = []

# Filling train_dataset and test_dataset
for i in range(49):
    nodes_df_tmp = merged_nodes_df[merged_nodes_df["time"] == i + 1].reset_index()
    nodes_df_tmp["index"] = nodes_df_tmp.index
    df_edge_tmp = (
        df_edge_time_fin.join(
            nodes_df_tmp.rename(columns={"nid": "source"})[
                ["source", "index"]
            ].set_index("source"),
            on="source",
            how="inner",
        )
        .join(
            nodes_df_tmp.rename(columns={"nid": "target"})[
                ["target", "index"]
            ].set_index("target"),
            on="target",
            how="inner",
            rsuffix="2",
        )
        .drop(columns=["source", "target"])
        .rename(columns={"index": "source", "index2": "target"})
    )
    x = torch.tensor(
        np.array(
            nodes_df_tmp.sort_values(by="index").drop(columns=["index", "nid", "label"])
        ),
        dtype=torch.float,
    )
    edge_index = torch.tensor(
        np.array(df_edge_tmp[["source", "target"]]).T, dtype=torch.long
    )
    edge_index = to_undirected(edge_index)
    mask = nodes_df_tmp["label"] != 2
    y = torch.tensor(np.array(nodes_df_tmp["label"]))

    if i + 1 < 35:
        data = Data(x=x, edge_index=edge_index, train_mask=mask, y=y)
        train_dataset.append(data)
    else:
        data = Data(x=x, edge_index=edge_index, test_mask=mask, y=y)
        test_dataset.append(data)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Saving the DataLoader objects
os.makedirs(os.path.join("src", "dataset"), exist_ok=True)
torch.save(train_loader, "src/dataset/train_loader.pth")
torch.save(test_loader, "src/dataset/test_loader.pth")
