"""Metrics and losses shared by the WESAD benchmark adapters."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


def classification_metrics(labels, predictions, loss):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    classes = range(3)
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)
        ),
        "kappa": float(cohen_kappa_score(labels, predictions)),
        "recall_per_class": recall_score(
            labels, predictions, labels=classes, average=None, zero_division=0
        ).tolist(),
        "f1_per_class": f1_score(
            labels, predictions, labels=classes, average=None, zero_division=0
        ).tolist(),
        "confusion": confusion_matrix(labels, predictions, labels=classes).tolist(),
        "pred_dist": np.bincount(predictions, minlength=3).tolist(),
    }


def calibration_metrics(probabilities, labels, bins=15):
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    one_hot = np.eye(3)[labels]
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == labels
    expected_calibration_error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            expected_calibration_error += selected.mean() * abs(
                correct[selected].mean() - confidence[selected].mean()
            )
    return {
        "nll": float(
            -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-8, 1.0)).mean()
        ),
        "brier": float(np.square(probabilities - one_hot).sum(1).mean()),
        "ece_15": float(expected_calibration_error),
    }


def fit_temperature(logits, labels):
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long)
    log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


class HusformerFocalLoss(nn.Module):
    """Focal loss used by the public Husformer WESAD training code."""

    def __init__(self, alpha=(0.4, 0.3, 0.3), gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits, target):
        log_probability = F.log_softmax(logits, dim=1).gather(
            1, target[:, None]
        ).squeeze(1)
        probability = log_probability.exp()
        alpha = self.alpha.to(logits.device)[target]
        return (-alpha * (1.0 - probability).pow(self.gamma) * log_probability).mean()
