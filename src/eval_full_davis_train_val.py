import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv, ChebConv, global_mean_pool
from davis_full_loader import load_davis_full

# ------------------------
# Load Full Davis (scaled)
# ------------------------
drug_graphs, protein_x, y = load_davis_full(max_pairs=5000)

indices = np.arange(len(y))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_graphs = [drug_graphs[i] for i in train_idx]
val_graphs = [drug_graphs[i] for i in val_idx]
y_train = y[train_idx]
y_val = y[val_idx]

# ------------------------
# Models
# ------------------------
class DrugGCN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.c1 = GCNConv(in_dim, 32)
        self.c2 = GCNConv(32, 32)
        self.fc = nn.Linear(32, 1)

    def forward(self, graphs):
        xs, ei, b = [], [], []
        offset = 0
        for i, g in enumerate(graphs):
            xs.append(g.x)
            ei.append(g.edge_index + offset)
            b.append(torch.full((g.x.size(0),), i))
            offset += g.x.size(0)
        x = torch.cat(xs)
        ei = torch.cat(ei, 1)
        b = torch.cat(b)
        x = F.relu(self.c1(x, ei))
        x = F.relu(self.c2(x, ei))
        x = global_mean_pool(x, b)
        return self.fc(x).squeeze()


class DrugSpectral(nn.Module):
    def __init__(self, in_dim, K=3):
        super().__init__()
        self.c1 = ChebConv(in_dim, 32, K)
        self.c2 = ChebConv(32, 32, K)
        self.fc = nn.Linear(32, 1)

    def forward(self, graphs):
        xs, ei, b = [], [], []
        offset = 0
        for i, g in enumerate(graphs):
            xs.append(g.x)
            ei.append(g.edge_index + offset)
            b.append(torch.full((g.x.size(0),), i))
            offset += g.x.size(0)
        x = torch.cat(xs)
        ei = torch.cat(ei, 1)
        b = torch.cat(b)
        x = F.relu(self.c1(x, ei))
        x = F.relu(self.c2(x, ei))
        x = global_mean_pool(x, b)
        return self.fc(x).squeeze()


# ------------------------
# Training + Evaluation
# ------------------------
def train_and_eval(model, name, epochs=50):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses, val_losses = [], []

    for _ in range(epochs):
        opt.zero_grad()
        pred = model(train_graphs)
        loss = F.mse_loss(pred, y_train)
        loss.backward()
        opt.step()

        with torch.no_grad():
            val_pred = model(val_graphs)
            val_loss = F.mse_loss(val_pred, y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

    # Metrics
    val_pred = val_pred.numpy()
    y_true = y_val.numpy()

    mse = np.mean((val_pred - y_true) ** 2)
    mae = np.mean(np.abs(val_pred - y_true))
    pearson = pearsonr(val_pred, y_true)[0]
    spearman = spearmanr(val_pred, y_true)[0]

    return train_losses, val_losses, val_pred, mse, mae, pearson, spearman


# ------------------------
# Run models
# ------------------------
results = {}

for Model, name in [(DrugGCN, "GCN"), (DrugSpectral, "Spectral")]:
    model = Model(drug_graphs[0].x.shape[1])
    results[name] = train_and_eval(model, name)

# ------------------------
# FIGURE 2 — Train vs Val Loss
# ------------------------
plt.figure()
for name, (tr, vl, *_ ) in results.items():
    plt.plot(tr, label=f"{name} Train")
    plt.plot(vl, linestyle="--", label=f"{name} Val")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.title("Train vs Validation Loss (Full Davis)")
plt.savefig("figures/train_val_loss.png", dpi=300)
plt.savefig("figures/train_val_loss.pdf")
plt.close()

# ------------------------
# FIGURE 3 — Predicted vs True
# ------------------------
plt.figure()
for name, (_, _, pred, *_ ) in results.items():
    plt.scatter(y_val, pred, label=name, alpha=0.6)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "k--")
plt.xlabel("True Affinity")
plt.ylabel("Predicted Affinity")
plt.legend()
plt.title("Predicted vs True Affinity")
plt.savefig("figures/pred_vs_true.png", dpi=300)
plt.savefig("figures/pred_vs_true.pdf")
plt.close()

# ------------------------
# FIGURE 4 — Correlation Bar Plot
# ------------------------
labels = list(results.keys())
pearsons = [results[m][5] for m in labels]
spearmans = [results[m][6] for m in labels]

x = np.arange(len(labels))
plt.figure()
plt.bar(x - 0.15, pearsons, width=0.3, label="Pearson R")
plt.bar(x + 0.15, spearmans, width=0.3, label="Spearman ρ")
plt.xticks(x, labels)
plt.ylabel("Correlation")
plt.legend()
plt.title("Correlation Comparison (Validation Set)")
plt.savefig("figures/correlation_comparison.png", dpi=300)
plt.savefig("figures/correlation_comparison.pdf")
plt.close()
