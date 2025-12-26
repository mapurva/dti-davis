import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, global_mean_pool
from davis_loader import load_davis

# ------------------------
# Load controlled Davis
# ------------------------
drug_graphs, _, y = load_davis()

# ------------------------
# Spectral GNN
# ------------------------
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
# Label noise experiment
# ------------------------
sigmas = [0.0, 0.1, 0.3, 0.5, 1.0]

for sigma in sigmas:
    noisy_y = y + sigma * torch.randn_like(y)

    model = DrugSpectral(drug_graphs[0].x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(40):
        optimizer.zero_grad()
        pred = model(drug_graphs)
        loss = F.mse_loss(pred, noisy_y)
        loss.backward()
        optimizer.step()

    # evaluate against clean labels
    with torch.no_grad():
        clean_loss = F.mse_loss(model(drug_graphs), y)

    print(
        f"[Spectral] Label noise σ={sigma:.1f} | "
        f"Train loss (noisy): {loss.item():.4f} | "
        f"Eval loss (clean): {clean_loss.item():.4f}"
    )
