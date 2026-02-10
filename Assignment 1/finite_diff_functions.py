import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

def l2_norm(u, uhat) -> float:
    return np.sqrt(np.sum((u - uhat)**2))

def inf_norm(u, uhat) -> float:
    return np.max(np.abs(u - uhat))
