import torch
import torch.nn as nn
import torch.nn.functional as F

# Mock drug features (fingerprint-like)
N = 500
X = torch.rand(N, 128)
y = torch.rand(N) * 10

class DrugMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()

model = DrugMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    optimizer.zero_grad()
    pred = model(X)
    loss = F.mse_loss(pred, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
