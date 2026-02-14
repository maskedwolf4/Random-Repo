"""Question: Write a function to calculate the Area Under the ROC Curve (AUC-ROC) using the trapezoidal rule given true labels and prediction scores.
Input: y_true=[0][0][1][1][1], y_scores=[0.1, 0.4, 0.35, 0.8, 0.9]
Expected Output: 0.833
Usage: Evaluating binary classifiers in disease diagnosis, comparing drug efficacy models, assessing biomarker performance"""

def calculate_auc_roc(y_true, y_scores):
    # Combine and sort by scores in descending order
    combined = sorted(zip(y_scores, y_true), reverse=True)
    
    # Calculate TPR and FPR at each threshold
    total_positives = sum(y_true)
    total_negatives = len(y_true) - total_positives
    
    tpr_list = [0]
    fpr_list = [0]
    tp = fp = 0
    
    for score, label in combined:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / total_positives)
        fpr_list.append(fp / total_negatives)
    
    # Apply trapezoidal rule
    auc = 0
    for i in range(len(fpr_list) - 1):
        auc += (fpr_list[i+1] - fpr_list[i]) * (tpr_list[i+1] + tpr_list[i]) / 2
    
    return round(auc, 3)

# Test
y_true = [0, 0, 1, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8, 0.9]
print(calculate_auc_roc(y_true, y_scores))  # Output: 0.833
