import torch
import torch.nn as nn
import torch.nn.functional as F
from drug_gnn import DrugGNN

class DTI_GNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug = DrugGNN()
        self.protein = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU()
        )
        self.head = nn.Linear(128, 1)

    def forward(self, drug_graph, protein_x):
        d = self.drug(drug_graph)
        p = self.protein(protein_x)
        z = torch.cat([d, p], dim=-1)
        return self.head(z).squeeze()
