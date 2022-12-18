import yaml
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

params = yaml.safe_load(open("params.yaml"))["eda"]


def get_dfs(params):
    df_features = pd.read_csv(params["path_fetures"], index_col=0)
    df_edges = pd.read_csv(params["path_edges"], index_col=0)
    df_classes = pd.read_csv(params["path_classes"], index_col=0)
    return df_features, df_edges, df_classes


df_features, df_edges, df_classes = get_dfs(params)

# Merging features and classes
df_features_classes = pd.merge(
    df_features, df_classes, how="left", left_on="txId", right_on="txId"
)


def number_of_transactions_line(df_features_classes):
    plt.figure(figsize=(12, 8))
    grouped = (
        df_features_classes.groupby(["Time step", "class"])["txId"]
        .count()
        .reset_index()
        .rename(columns={"txId": "count"})
    )
    sns.lineplot(
        x="Time step",
        y="count",
        hue="class",
        style="class",
        data=grouped,
        palette=["g", "b", "r"],
    )
    plt.legend()
    plt.title("# Transactions by in each time step per class")
    plt.savefig("src/figures/num_transactions.png")


def ilicit_licit_at_timestep(df_features_classes):
    ilicit = df_features_classes[df_features_classes["class"] == 1][
        df_features_classes["Time step"] == 20
    ]
    licit = df_features_classes[df_features_classes["class"] == 0][
        df_features_classes["Time step"] == 20
    ]
    ilicit_licit = ilicit.append(licit).sample(frac=1).reset_index(drop=True)
    X = ilicit_licit.drop(["class"], axis=1).values
    y = ilicit_licit["class"].values
    tsne = TSNE(n_components=2, random_state=0)
    X_t = tsne.fit_transform(X)
    plt.figure(figsize=(12, 8))
    plt.scatter(
        X_t[np.where(y == 1), 0],
        X_t[np.where(y == 1), 1],
        marker="o",
        color="r",
        linewidth=1,
        alpha=0.8,
        label="Ilicit",
    )
    plt.scatter(
        X_t[np.where(y == 0), 0],
        X_t[np.where(y == 0), 1],
        marker="o",
        color="g",
        linewidth=1,
        alpha=0.8,
        label="Licit",
    )
    plt.legend(loc="best")
    plt.title("Ilicit vs Licit Transactions at Time=20")
    plt.savefig("src/figures/ilicit_licit.png")


os.makedirs(os.path.join("src", "figures"), exist_ok=True)
number_of_transactions_line(df_features_classes)
ilicit_licit_at_timestep(df_features_classes)
