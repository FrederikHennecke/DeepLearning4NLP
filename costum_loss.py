import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
import numpy as np

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, logits, labels):
        # Convert logits to binary predictions
        predicted_labels = (logits > 0.5).int()

        # Initialize tensor to store Matthews Correlation Coefficients (MCCs)
        matthews_coefficients = torch.zeros(labels.size(1), device=logits.device)

        # Compute the MCC for each label
        for label_idx in range(labels.size(1)):
            pred = predicted_labels[:, label_idx]
            true = labels[:, label_idx]

            tp = (pred * true).sum().float()
            tn = ((1 - pred) * (1 - true)).sum().float()
            fp = (pred * (1 - true)).sum().float()
            fn = ((1 - pred) * true).sum().float()

            numerator = (tp * tn) - (fp * fn)
            denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

            # Handle case where denominator is zero
            mcc = numerator / (denominator + 1e-6)
            mcc = mcc.requires_grad_()
            matthews_coefficients[label_idx] = mcc

        # Define weights as a tensor
        weights = torch.tensor([1, 10, 1, 1, 1, 10, 10], dtype=torch.float32, device=logits.device)

        # Calculate the weighted MCC
        weighted_matthews_coefficient = torch.dot(matthews_coefficients, weights) / labels.size(1)

        return -weighted_matthews_coefficient
