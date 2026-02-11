"""
Question: Write a function to calculate the log-loss (binary cross-entropy) for binary classification predictions.

Input: y_true=[43][43][43], y_pred=[0.9, 0.1, 0.8, 0.7, 0.2]

Expected Output: 0.168

Usage: Training neural networks for image classification, evaluating probabilistic predictions in risk assessment, optimizing logistic regression model
"""

import numpy as np

def log_loss(y_true, y_pred, epsilon=1e-15):
    """
    Calculate binary cross-entropy (log loss).
    Formula: -1/N * Σ[y*log(p) + (1-y)*log(1-p)]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Calculate binary cross-entropy
    n = len(y_true)
    loss = -np.sum(y_true * np.log(y_pred) + 
                   (1 - y_true) * np.log(1 - y_pred)) / n
    
    return round(loss, 3)

# Test
y_true = [1, 0, 1, 1, 0]
y_pred = [0.9, 0.1, 0.8, 0.7, 0.2]
result = log_loss(y_true, y_pred)
print(f"Log Loss: {result}")
