import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from torch_geometric.nn import ChebConv, global_mean_pool
from davis_full_loader import load_davis_full

# ------------------------
# Load data
# ------------------------
drug_graphs, protein_x, y = load_davis_full(max_pairs=5000)
N = len(y)

# ------------------------
# Spectral GNN
# ------------------------
class DrugSpectral(nn.Module):
    def __init__(self, in_dim, K=3):
        super().__init__()
        self.c1 = ChebConv(in_dim, 32, K)
        self.c2 = ChebConv(32, 32, K)
        self.fc = nn.Linear(32, 1)

    def forward(self, graphs):
        xs, ei, b = [], [], []
        offset = 0
        for i, g in enumerate(graphs):
            xs.append(g.x)
            ei.append(g.edge_index + offset)
            b.append(torch.full((g.x.size(0),), i))
            offset += g.x.size(0)

        x = torch.cat(xs)
        ei = torch.cat(ei, 1)
        b = torch.cat(b)

        x = F.relu(self.c1(x, ei))
        x = F.relu(self.c2(x, ei))
        x = global_mean_pool(x, b)
        return self.fc(x).squeeze()


# ------------------------
# k-fold CV
# ------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mse_list, mae_list, pearson_list, spearman_list = [], [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(range(N))):
    train_graphs = [drug_graphs[i] for i in train_idx]
    val_graphs = [drug_graphs[i] for i in val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]

    model = DrugSpectral(drug_graphs[0].x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train
    for _ in range(40):
        opt.zero_grad()
        pred = model(train_graphs)
        loss = F.mse_loss(pred, y_train)
        loss.backward()
        opt.step()

    # evaluate
    with torch.no_grad():
        val_pred = model(val_graphs)

    mse = F.mse_loss(val_pred, y_val).item()
    mae = F.l1_loss(val_pred, y_val).item()
    p = pearsonr(val_pred.numpy(), y_val.numpy())[0]
    s = spearmanr(val_pred.numpy(), y_val.numpy())[0]

    mse_list.append(mse)
    mae_list.append(mae)
    pearson_list.append(p)
    spearman_list.append(s)

    print(f"Fold {fold+1} | MSE={mse:.3f} | MAE={mae:.3f} | R={p:.3f} | ρ={s:.3f}")

# ------------------------
# Summary
# ------------------------
def summarize(name, values):
    print(f"{name}: {np.mean(values):.3f} ± {np.std(values):.3f}")

print("\n=== Cross-validation summary ===")
summarize("MSE", mse_list)
summarize("MAE", mae_list)
summarize("Pearson R", pearson_list)
summarize("Spearman ρ", spearman_list)

# ------------------------
# Plot metrics
# ------------------------
plt.figure(figsize=(8, 4))
plt.bar(range(5), mse_list)
plt.xlabel("Fold")
plt.ylabel("MSE")
plt.title("5-fold Cross-Validation MSE (Spectral GNN)")
plt.savefig("figures/kfold_mse.png", dpi=300)
plt.savefig("figures/kfold_mse.pdf")
plt.close()
