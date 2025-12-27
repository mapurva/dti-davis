import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from torch_geometric.nn import ChebConv, global_mean_pool
import torch.nn as nn
from davis_full_loader import load_davis_full

# ------------------------
# Load data
# ------------------------
drug_graphs, protein_x, y = load_davis_full(max_pairs=5000)

N = len(y)
idx = torch.randperm(N)
split = int(0.8 * N)

train_idx = idx[:split]
val_idx = idx[split:]

train_graphs = [drug_graphs[i] for i in train_idx]
val_graphs = [drug_graphs[i] for i in val_idx]

y_train = y[train_idx]
y_val = y[val_idx]

# ------------------------
# Spectral GNN
# ------------------------
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
# TRAIN (Phase 1)
# ------------------------
model = DrugSpectral(train_graphs[0].x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses = []

for epoch in range(50):
    optimizer.zero_grad()
    pred = model(train_graphs)
    loss = F.mse_loss(pred, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

# ------------------------
# FREEZE + EVALUATE (Phase 2 & 3)
# ------------------------
model.eval()
with torch.no_grad():
    val_pred = model(val_graphs)

val_mse = F.mse_loss(val_pred, y_val).item()
val_mae = F.l1_loss(val_pred, y_val).item()

pearson_r = pearsonr(val_pred.numpy(), y_val.numpy())[0]
spearman_r = spearmanr(val_pred.numpy(), y_val.numpy())[0]

print("Clean Evaluation on Validation Set")
print(f"MSE      : {val_mse:.4f}")
print(f"MAE      : {val_mae:.4f}")
print(f"Pearson R: {pearson_r:.4f}")
print(f"Spearman : {spearman_r:.4f}")

# ------------------------
# PLOTS
# ------------------------
plt.figure()
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.title("Training Loss (Spectral GNN)")
plt.savefig("figures/train_loss_clean.png", dpi=300)
plt.close()

plt.figure()
plt.scatter(y_val.numpy(), val_pred.numpy(), alpha=0.7)
plt.plot(
    [y_val.min(), y_val.max()],
    [y_val.min(), y_val.max()],
    linestyle="--"
)
plt.xlabel("True Affinity")
plt.ylabel("Predicted Affinity")
plt.title("Predicted vs True (Validation)")
plt.savefig("figures/pred_vs_true_clean.png", dpi=300)
plt.close()
