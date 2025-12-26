import torch
import torch.nn as nn
import torch.nn.functional as F
from davis_loader import load_davis

# ------------------------
# Load controlled Davis
# ------------------------
drug_graphs, protein_x, y = load_davis()

# Simple drug feature: mean of atom features
drug_features = []
for g in drug_graphs:
    drug_features.append(g.x.mean(dim=0))
drug_features = torch.stack(drug_features)

# ------------------------
# Baseline model
# ------------------------
class DrugMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()

model = DrugMLP(drug_features.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------
# Training loop
# ------------------------
for epoch in range(50):
    optimizer.zero_grad()
    pred = model(drug_features)
    loss = F.mse_loss(pred, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"[Baseline] Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")
