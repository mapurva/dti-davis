import torch
import torch.nn as nn
import torch.nn.functional as F

N = 500
drug = torch.rand(N, 128)
protein = torch.rand(N, 20)
y = torch.rand(N) * 10

class DTIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(148, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, d, p):
        return self.net(torch.cat([d, p], dim=1)).squeeze()

model = DTIModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    optimizer.zero_grad()
    pred = model(drug, protein)
    loss = F.mse_loss(pred, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
