import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from davis_loader import load_davis

# ------------------------
# Load controlled Davis
# ------------------------
drug_graphs, protein_x, y = load_davis()

# Create batch indices manually
batch = torch.arange(len(drug_graphs))

# ------------------------
# GNN model
# ------------------------
class DrugGNN(nn.Module):
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

        x = torch.cat(xs, dim=0)
        edge_index = torch.cat(edge_indices, dim=1)
        batch = torch.cat(batches, dim=0)

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)

        return self.fc(x).squeeze()

model = DrugGNN(drug_graphs[0].x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------
# Training loop
# ------------------------
for epoch in range(50):
    optimizer.zero_grad()
    pred = model(drug_graphs)
    loss = F.mse_loss(pred, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"[GNN] Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")
