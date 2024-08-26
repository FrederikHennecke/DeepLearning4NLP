import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
import numpy as np

class CustomLoss(nn.Module):
    def __init__(self, pos_weight):
        super(CustomLoss, self).__init__()
        self.pos_weight = pos_weight

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
        weights = torch.tensor([2.6425e-01, 1.6713e+00, 2.6901e-01, 2.0537e-01, 3.5322e-01, 2.5138e+00,
        2.3272e+00], dtype=torch.float32, device=logits.device)
        # [2.6425e-01, 4.6713e+00, 2.6901e-01, 2.0537e-01, 3.5322e-01, 2.5138e+02, 4.3272e+00]

        # Calculate the weighted MCC
        weighted_matthews_coefficient = torch.dot(matthews_coefficients, weights) / labels.size(1)

        # loss = nn.BCEWithLogitsLoss()
        # print(f"weighted_matthews_coefficient, {weighted_matthews_coefficient:.3f}")

        # pos_weight = torch.tensor([1., 0.3, 1., 1., 1., 0.3, 0.3], device=device) # [2.3696682, 0.6013229, 2.3364486, 2.9069767, 1.8975332, 0.4972650, 0.6097561]
        # [1., 0.3, 1., 1., 1., 0.3, 0.3]
        
        BCE = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = BCE(logits, labels.float())

        return -weighted_matthews_coefficient  + 0.5 * loss