"""Question: Create a function to perform batch normalization on a feature matrix (normalize each feature across samples to zero mean and unit variance).
Input: data=[[1][2], [2][4], [3][6], [4][8]], epsilon=1e-5
Expected Output: [[-1.342, -1.342], [-0.447, -0.447], [0.447, 0.447], [1.342, 1.342]]
Usage: Preprocessing layers in deep neural networks, stabilizing training in drug discovery models, normalizing multi-omics data integration"""

import numpy as np

def batch_normalization(data, epsilon=1e-5):
    data = np.array(data, dtype=float)
    
    # Calculate mean and std for each feature (column)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Normalize
    normalized = (data - mean) / (std + epsilon)
    
    return np.round(normalized, 3).tolist()

# Test
data = [[1, 2], [2, 4], [3, 6], [4, 8]]
print(batch_normalization(data, epsilon=1e-5))
# Output: [[-1.342, -1.342], [-0.447, -0.447], [0.447, 0.447], [1.342, 1.342]]
