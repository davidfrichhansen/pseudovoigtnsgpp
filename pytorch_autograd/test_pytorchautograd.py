import torch
import numpy as np


x = torch.tensor(2.0, requires_grad=True)
y = x*x
z = y + 3
k = z*2


