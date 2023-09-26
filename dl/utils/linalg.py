from scipy.linalg import block_diag
import numpy as np


def make_block_diag_from_labels(labels, dtype=bool):
    indices, counts = np.unique(labels, return_counts=True)
    
    diag = block_diag(*[np.ones((count, count), dtype=dtype) for count in counts])
    
    return diag