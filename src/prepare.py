import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os

params = yaml.safe_load(open("params.yaml"))["prepare"]

# Loading the dataframes from csv files
df_edge = pd.read_csv(params["path_edges"])
df_class = pd.read_csv(params["path_classes"])
df_features = pd.read_csv(params["path_fetures"], header=None)

# Setting column names for df_features
df_features.columns = (
    ["id", "time step"]
    + [f"trans_feat_{i}" for i in range(93)]
    + [f"agg_feat_{i}" for i in range(72)]
)

# Printing data frame specifics
print(f"Shape of df_class: \n {df_class.shape} \n df_class: \n {df_class.head(3)}")
print(f"Shape of df_edge: \n {df_edge.shape} \n df_edge: \n {df_edge.head(3)}")
print(
    f"Shape of df_features: \n {df_features.shape} \n df_features: \n {df_features.head(3)}"
)

# Creating nodes dataframe
all_nodes = list(
    set(df_edge["txId1"])
    .union(set(df_edge["txId2"]))
    .union(set(df_class["txId"]))
    .union(set(df_features["id"]))
)
nodes_df = pd.DataFrame(all_nodes, columns=["id"]).reset_index()
print(f"Shape of nodes_df: \n {nodes_df.shape} \n nodes_df: \n {nodes_df.head(3)}")

# Fixing id index in df_edges using nodes_df
df_edge = (
    df_edge.join(
        nodes_df.rename(columns={"id": "txId1"}).set_index("txId1"),
        on="txId1",
        how="inner",
    )
    .join(
        nodes_df.rename(columns={"id": "txId2"}).set_index("txId2"),
        on="txId2",
        how="inner",
        rsuffix="2",
    )
    .drop(columns=["txId1", "txId2"])
    .rename(columns={"index": "txId1", "index2": "txId2"})
)
print(f"Shape of df_edge: \n {df_edge.shape} \n df_edge: \n {df_edge.head(3)}")

# Fixing the txId of df_class using nodes_df
df_class = (
    df_class.join(
        nodes_df.rename(columns={"id": "txId"}).set_index("txId"),
        on="txId",
        how="inner",
    )
    .drop(columns=["txId"])
    .rename(columns={"index": "txId"})[["txId", "class"]]
)
print(f"Shape of df_class: \n {df_class.shape} \n df_class: \n {df_class.head(3)}")

# Fixing the id of df_features using nodes_df
df_features = (
    df_features.join(nodes_df.set_index("id"), on="id", how="inner")
    .drop(columns=["id"])
    .rename(columns={"index": "id"})
)
df_features = df_features[["id"] + list(df_features.drop(columns=["id"]).columns)]
print(
    f"Shape of df_features: \n {df_features.shape} \n df_features: \n {df_features.head(3)}"
)

# Creating df_edge_time_fin
df_edge_time = df_edge.join(
    df_features[["id", "time step"]].rename(columns={"id": "txId1"}).set_index("txId1"),
    on="txId1",
    how="left",
    rsuffix="1",
).join(
    df_features[["id", "time step"]].rename(columns={"id": "txId2"}).set_index("txId2"),
    on="txId2",
    how="left",
    rsuffix="2",
)
df_edge_time["is_time_same"] = df_edge_time["time step"] == df_edge_time["time step2"]
df_edge_time_fin = df_edge_time[["txId1", "txId2", "time step"]].rename(
    columns={"txId1": "source", "txId2": "target", "time step": "time"}
)
print(
    f"Shape of df_edge_time_fin: \n {df_edge_time_fin.shape} \n df_edge_time_fin: \n {df_edge_time_fin.head(3)}"
)

# Creating node labels
node_label = (
    df_class.rename(columns={"txId": "nid", "class": "label"})[["nid", "label"]]
    .sort_values(by="nid")
    .merge(
        df_features[["id", "time step"]].rename(
            columns={"id": "nid", "time step": "time"}
        ),
        on="nid",
        how="left",
    )
)
node_label["label"] = (
    node_label["label"].apply(lambda x: "3" if x == "unknown" else x).astype(int) - 1
)
print(
    f"Shape of node_label: \n {node_label.shape} \n node_label: \n {node_label.head(3)}"
)

# Merging node label with df_features
merged_nodes_df = node_label.merge(
    df_features.rename(columns={"id": "nid", "time step": "time"}).drop(
        columns=["time"]
    ),
    on="nid",
    how="left",
)
print(
    f"Shape of merged_nodes_df: \n {merged_nodes_df.shape} \n merged_nodes_df: \n {merged_nodes_df.head(3)}"
)

# Exporting the data frames
os.makedirs(os.path.join("src", "prepared"), exist_ok=True)
df_features.drop(columns=["time step"]).to_csv(
    "src/prepared/elliptic_txs_features.csv", index=False
)
df_class.rename(columns={"txId": "nid", "class": "label"})[
    ["nid", "label"]
].sort_values(by="nid").to_csv("src/prepared/elliptic_txs_classes.csv", index=False)
df_features[["id", "time step"]].rename(columns={"id": "nid", "time step": "time"})[
    ["nid", "time"]
].sort_values(by="nid").to_csv("src/prepared/elliptic_txs_nodetime.csv", index=False)
df_edge_time_fin[["source", "target", "time"]].to_csv(
    "src/prepared/elliptic_txs_edgelist_timed.csv", index=False
)

node_label.to_csv("src/prepared/node_label.csv", index=False)
merged_nodes_df.to_csv("src/prepared/merged_nodes_df.csv", index=False)
