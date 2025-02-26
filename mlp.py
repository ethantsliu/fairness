import torch
import torch.nn as nn
import numpy as np

# Check for Metal Performance Shaders on mac
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear()
