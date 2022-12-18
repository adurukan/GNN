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

from sklearn.metrics import roc_auc_score
import scipy.sparse as sp
import yaml
import os

params = yaml.safe_load(open("params.yaml"))["prepare"]


def get_dfs(params):
    df_features = pd.read_csv(params["path_fetures"], header=None)
    df_edges = pd.read_csv(params["path_edges"])
    df_classes = pd.read_csv(params["path_classes"])
    return df_features, df_edges, df_classes


def fix_df_classes(df_classes):
    df_classes["class"] = df_classes["class"].map({"unknown": 2, "1": 1, "2": 0})
    return df_classes


def fix_df_features(df_features):
    colNames1 = {"0": "txId", 1: "Time step"}
    colNames2 = {str(ii + 2): "Local_feature_" + str(ii + 1) for ii in range(93)}
    colNames3 = {str(ii + 95): "Aggregate_feature_" + str(ii + 1) for ii in range(72)}

    colNames = dict(colNames1, **colNames2, **colNames3)
    colNames = {int(jj): item_kk for jj, item_kk in colNames.items()}
    df_features = df_features.rename(columns=colNames)
    return df_features


def export_df(df_features, df_edges, df_classes):
    os.makedirs(os.path.join("src", "prepared"), exist_ok=True)
    df_features.to_csv("src/prepared/df_features.csv")
    df_edges.to_csv("src/prepared/df_edges.csv")
    df_classes.to_csv("src/prepared/df_classes.csv")


df_features, df_edges, df_classes = get_dfs(params)
df_classes = fix_df_classes(df_classes)
df_features = fix_df_features(df_features)
export_df(df_features, df_edges, df_classes)
