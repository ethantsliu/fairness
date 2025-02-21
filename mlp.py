import torch
import torch.nn as nn
import numpy as np

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
