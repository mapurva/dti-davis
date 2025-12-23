import torch
import numpy as np

x = torch.randn(10, 5)
w = torch.randn(5, 1)

y = x @ w
print("Output shape:", y.shape)
