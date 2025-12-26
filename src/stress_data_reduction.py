import torch
import torch.nn.functional as F
from davis_loader import load_davis
from torch_geometric.nn import GCNConv, ChebConv, global_mean_pool
import torch.nn as nn
import random

# ------------------------
# Load controlled Davis
# ------------------------
drug_graphs, protein_x, y = load_davis()
N = len(y)

# ------------------------
# Models
# ------------------------
class DrugGCN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, 32)
        self.conv2 = GCNConv(32, 32)
        self.fc = nn.Linear(32, 1)

    def forward(self, graphs):
        xs, edge_indices, batches = [], [], []
        offset = 0
        for i, g in enumerate(graphs):
            xs.append(g.x)
            edge_indices.append(g.edge_index + offset)
            batches.append(torch.full((g.x.size(0),), i, dtype=torch.long))
            offset += g.x.size(0)
        x = torch.cat(xs)
        edge_index = torch.cat(edge_indices, dim=1)
        batch = torch.cat(batches)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze()


class DrugSpectral(nn.Module):
    def __init__(self, in_dim, K=3):
        super().__init__()
        self.conv1 = ChebConv(in_dim, 32, K)
        self.conv2 = ChebConv(32, 32, K)
        self.fc = nn.Linear(32, 1)

    def forward(self, graphs):
        xs, edge_indices, batches = [], [], []
        offset = 0
        for i, g in enumerate(graphs):
            xs.append(g.x)
            edge_indices.append(g.edge_index + offset)
            batches.append(torch.full((g.x.size(0),), i, dtype=torch.long))
            offset += g.x.size(0)
        x = torch.cat(xs)
        edge_index = torch.cat(edge_indices, dim=1)
        batch = torch.cat(batches)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze()


# ------------------------
# Experiment
# ------------------------
fractions = [1.0, 0.7, 0.4, 0.2]

for frac in fractions:
    k = int(N * frac)
    idx = random.sample(range(N), k)
    subset_graphs = [drug_graphs[i] for i in idx]
    subset_y = y[idx]

    model = DrugSpectral(subset_graphs[0].x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(30):
        opt.zero_grad()
        pred = model(subset_graphs)
        loss = F.mse_loss(pred, subset_y)
        loss.backward()
        opt.step()

    print(f"[Spectral] Data fraction {frac:.1f} | Loss: {loss.item():.4f}")
