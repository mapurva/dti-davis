import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Mock Davis protein features (AA composition)
N = 500
X = torch.rand(N, 20)   # protein features
y = torch.rand(N) * 10  # binding affinity

class ProteinMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()

model = ProteinMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    optimizer.zero_grad()
    pred = model(X)
    loss = F.mse_loss(pred, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
