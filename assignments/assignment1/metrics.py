import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    TP = np.sum((prediction == True) & (ground_truth == True)) # count predicts 0 class
    TN = np.sum((prediction == False) & (ground_truth == False)) # count predicts 9 class
    FP = np.sum((prediction == True) & (ground_truth == False)) # count false 0 class predicts
    FN = np.sum((prediction == False) & (ground_truth == True)) # count false 9 class predicts
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1 = 2 * precision * recall / (precision + recall) 
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    TPTN = np.sum(prediction == ground_truth)
    accuracy = TPTN / len(prediction)
    return accuracy
