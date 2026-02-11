"""
Question: Implement the silhouette coefficient calculation for evaluating clustering quality by measuring how similar an object is to its own cluster compared to other clusters.

Input: data=[[43][44], [44][45], [51][51], [49][41]], cluster_labels=[43][43]

Expected Output: 0.85 (approximate average silhouette score)

Usage: Determining optimal number of clusters in patient segmentation, validating clustering of cell types, evaluating unsupervised learning results
"""

import numpy as np

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))

def silhouette_coefficient(data, cluster_labels):
    """
    Calculate average silhouette coefficient.
    Formula: s(i) = (b(i) - a(i)) / max(a(i), b(i))
    """
    data = np.array(data)
    cluster_labels = np.array(cluster_labels)
    n_samples = len(data)
    silhouette_scores = []
    
    for i in range(n_samples):
        # Current point and its cluster
        point = data[i]
        own_cluster = cluster_labels[i]
        
        # Calculate a(i): mean distance to points in same cluster
        same_cluster_indices = np.where(cluster_labels == own_cluster)[0]
        same_cluster_indices = same_cluster_indices[same_cluster_indices != i]
        
        if len(same_cluster_indices) == 0:
            a_i = 0
        else:
            distances_same = [euclidean_distance(point, data[j]) 
                            for j in same_cluster_indices]
            a_i = np.mean(distances_same)
        
        # Calculate b(i): minimum mean distance to points in other clusters
        other_clusters = np.unique(cluster_labels[cluster_labels != own_cluster])
        b_i = float('inf')
        
        for cluster in other_clusters:
            other_cluster_indices = np.where(cluster_labels == cluster)[0]
            distances_other = [euclidean_distance(point, data[j]) 
                             for j in other_cluster_indices]
            mean_dist = np.mean(distances_other)
            b_i = min(b_i, mean_dist)
        
        # Calculate silhouette score for this point
        if max(a_i, b_i) == 0:
            s_i = 0
        else:
            s_i = (b_i - a_i) / max(a_i, b_i)
        
        silhouette_scores.append(s_i)
    
    return round(np.mean(silhouette_scores), 2)

# Test
data = [[1,2], [2,3], [8,8], [9,10]]
labels = [0, 0, 1, 1]
result = silhouette_coefficient(data, labels)
print(f"Silhouette Score: {result}")
