import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from dti_gnn_model import DTI_GNN_Model
import random

# generate mock dataset
def random_graph():
    n = random.randint(10, 30)
    x = torch.rand(n, 10)
    edge_index = torch.randint(0, n, (2, n*2))
    return Data(x=x, edge_index=edge_index)

N = 200
drug_graphs = [random_graph() for _ in range(N)]
protein = torch.rand(N, 20)
y = torch.rand(N) * 10

model = DTI_GNN_Model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    total_loss = 0
    for i in range(N):
        optimizer.zero_grad()
        pred = model(drug_graphs[i], protein[i])
        loss = F.mse_loss(pred, y[i])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss/N:.4f}")
