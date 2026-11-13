import numpy as np
import torch

from benchmarks.wesad.metrics import (
    HusformerFocalLoss,
    calibration_metrics,
    classification_metrics,
)


def test_wesad_metric_contract():
    labels = np.asarray([0, 1, 2, 0])
    predictions = np.asarray([0, 1, 1, 0])
    result = classification_metrics(labels, predictions, loss=0.5)
    assert set(("accuracy", "balanced_accuracy", "macro_f1", "kappa")) <= set(result)
    assert np.asarray(result["confusion"]).shape == (3, 3)
    probabilities = np.asarray(
        [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.5, 0.3], [0.7, 0.2, 0.1]]
    )
    assert set(calibration_metrics(probabilities, labels)) == {"nll", "brier", "ece_15"}


def test_husformer_focal_loss_is_finite():
    logits = torch.randn(5, 3, requires_grad=True)
    target = torch.tensor([0, 1, 2, 1, 0])
    loss = HusformerFocalLoss()(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
