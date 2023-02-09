import numpy as np
import pandas as pd
import torch
from torch.nn import Linear, LayerNorm, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import scipy.sparse as sp
import yaml
import os
import json
from model import model, device

params = yaml.safe_load(open("params.yaml"))["train"]
# Loading patience, lr, epoches
patience = params["patience"]
lr = params["lr"]
epoches = params["epoches"]
# Loading train_loader, test_loader
train_loader = torch.load(params["path_train_loader"])
test_loader = torch.load(params["path_test_loader"])
# Creating optimizer, criterion
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.7, 0.3]).to(device))

# Creating lists to append to later
train_losses = []
val_losses = []
accuracies = []
if1 = []
precisions = []
recalls = []
iterations = []
results_ = []
for epoch in range(epoches):

    model.train()
    train_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        _, pred = out[data.train_mask].max(dim=1)
        loss.backward()
        train_loss += loss.item() * data.num_graphs
        optimizer.step()
    train_loss /= len(train_loader.dataset)

    if (epoch + 1) % 50 == 0:
        model.eval()
        ys, preds = [], []
        val_loss = 0
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out[data.test_mask], data.y[data.test_mask])
            val_loss += loss.item() * data.num_graphs
            _, pred = out[data.test_mask].max(dim=1)
            ys.append(data.y[data.test_mask].cpu())
            preds.append(pred.cpu())

        y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
        val_loss /= len(test_loader.dataset)
        f1 = f1_score(y, pred, average=None)
        mf1 = f1_score(y, pred, average="micro")
        precision = precision_score(y, pred, average=None)
        recall = recall_score(y, pred, average=None)

        iterations.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if1.append(f1[0])
        accuracies.append(mf1)
        precisions.append(precision[0])
        recalls.append(recall[0])
        results_.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "if1": f1[0],
                "mf1": mf1,
                "precision": precision[0],
                "recall": recall[0],
            }
        )
print(results_)
with open("training_loss_acc.json", "w") as f:
    json.dump(
        [i for i in results_],
        f,
        indent=4,
    )
os.makedirs(os.path.join("src", "model"), exist_ok=True)
torch.save(model.state_dict(), "src/model/model_state.pt")
