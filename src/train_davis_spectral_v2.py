import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from torch_geometric.nn import ChebConv, global_mean_pool
from sklearn.model_selection import train_test_split

from davis_loader_v2 import load_davis_v2

# ------------------------
# Load data
# ------------------------
graphs, protein_x, y = load_davis_v2()

idx = np.arange(len(y))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

train_graphs = [graphs[i] for i in train_idx]
test_graphs = [graphs[i] for i in test_idx]
y_train = y[train_idx]
y_test = y[test_idx]


# ------------------------
# Spectral GNN
# ------------------------
class SpectralGNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.c1 = ChebConv(in_dim, 64, K=3)
        self.c2 = ChebConv(64, 64, K=3)
        self.fc = nn.Linear(64, 1)

    def forward(self, graphs):
        xs, eis, batch = [], [], []
        offset = 0

        for i, g in enumerate(graphs):
            xs.append(g.x)
            eis.append(g.edge_index + offset)
            batch.append(torch.full((g.x.size(0),), i))
            offset += g.x.size(0)

        x = torch.cat(xs)
        ei = torch.cat(eis, dim=1)
        batch = torch.cat(batch)

        x = F.relu(self.c1(x, ei))
        x = F.relu(self.c2(x, ei))
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze()


model = SpectralGNN(train_graphs[0].x.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------
# Train
# ------------------------
train_losses = []

for epoch in range(80):
    opt.zero_grad()
    pred = model(train_graphs)
    loss = F.mse_loss(pred, y_train)
    loss.backward()
    opt.step()
    train_losses.append(loss.item())

# ------------------------
# Evaluate
# ------------------------
with torch.no_grad():
    pred = model(test_graphs).cpu().numpy()
    true = y_test.cpu().numpy()

mse = np.mean((pred - true) ** 2)
mae = np.mean(np.abs(pred - true))
pearson = pearsonr(pred, true)[0]
spearman = spearmanr(pred, true)[0]

print(f"MSE      : {mse:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"PearsonR : {pearson:.4f}")
print(f"Spearman : {spearman:.4f}")

# ------------------------
# Save plot
# ------------------------
plt.figure()
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Train MSE")
plt.title("Davis v2 Spectral GNN Training")
plt.savefig("figures/davis_v2_train_loss.png", dpi=300)
plt.close()
