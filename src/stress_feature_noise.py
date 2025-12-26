import torch
import torch.nn.functional as F
from davis_loader import load_davis
from torch_geometric.nn import ChebConv, global_mean_pool
import torch.nn as nn

drug_graphs, _, y = load_davis()

# Add Gaussian noise to atom features
def noisy_graphs(graphs, sigma):
    noisy = []
    for g in graphs:
        g2 = g.clone()
        g2.x = g.x + sigma * torch.randn_like(g.x)
        noisy.append(g2)
    return noisy


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


sigmas = [0.0, 0.05, 0.1, 0.2]

for s in sigmas:
    graphs = noisy_graphs(drug_graphs, s)
    model = DrugSpectral(graphs[0].x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(30):
        opt.zero_grad()
        pred = model(graphs)
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()

    print(f"[Spectral] Noise sigma {s:.2f} | Loss: {loss.item():.4f}")
