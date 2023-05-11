# Libraries
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

# Calculate Loss
def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)

# Calculate jaccard score
def jaccard(y_true, y_pred):
    # print(y_true)
    # print(y_true.shape)
    # print(y_pred)
    # print(y_pred.shape)
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return (np.sum(intersection) / np.sum(union)) * 5
