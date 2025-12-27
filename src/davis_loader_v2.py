import torch
import pickle
import json
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data

# ------------------------
# Improved atom features
# ------------------------
def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetMass() * 0.01
    ]


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor(
        [atom_features(a) for a in mol.GetAtoms()],
        dtype=torch.float
    )

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    return Data(x=x, edge_index=edge_index)


def protein_to_feature(seq):
    aa = "ACDEFGHIKLMNPQRSTVWY"
    counts = [seq.count(a) for a in aa]
    return torch.tensor(counts, dtype=torch.float) / len(seq)


# ------------------------
# Controlled Davis (v2)
# ------------------------
def load_davis_v2(path="data/davis/"):
    with open(path + "ligands_can.txt") as f:
        ligands = list(json.load(f).values())

    proteins = open(path + "proteins.txt").read().strip().split("\n")
    fixed_protein = proteins[0]

    with open(path + "Y.pkl", "rb") as f:
        Y = pickle.load(f, encoding="latin1")

    drug_graphs, protein_x, y = [], [], []

    for d_idx, smi in enumerate(ligands):
        g = smiles_to_graph(smi)
        if g is None:
            continue

        drug_graphs.append(g)
        protein_x.append(protein_to_feature(fixed_protein))
        y.append(Y[d_idx, 0])

    y = torch.tensor(y, dtype=torch.float)

    # 🔑 critical: normalize labels
    y = (y - y.mean()) / y.std()

    return drug_graphs, torch.stack(protein_x), y
