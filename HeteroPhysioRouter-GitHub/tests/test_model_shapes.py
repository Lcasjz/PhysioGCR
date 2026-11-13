import torch

from heterophysiorouter import QualityAwareRouter, corrupt_groups
from physiogcr import PhysioGCR


def test_pamap2_router_forward_and_auxiliary_shapes():
    groups = ((0, 1), (1, 18), (18, 35), (35, 52))
    model = QualityAwareRouter(
        channels=52,
        samples=24,
        patch=4,
        classes=12,
        channel_groups=groups,
        variant="full_no_calibration",
        hidden=32,
        dropout=0.0,
    ).eval()
    signal = torch.randn(2, 24, 52)
    with torch.no_grad():
        logits, auxiliary = model(signal, signal, return_aux=True)
    assert logits.shape == (2, 12)
    assert auxiliary["gates"].shape == (2, 3)
    assert auxiliary["sensor_reliability"].shape == (2, 4)
    assert auxiliary["temporal_reliability"].shape == (2, 6)
    torch.testing.assert_close(auxiliary["gates"].sum(1), torch.ones(2))
    assert PhysioGCR is QualityAwareRouter


def test_group_corruption_preserves_at_least_one_group():
    groups = ((0, 1), (1, 4), (4, 7))
    signal = torch.randn(16, 12, 7)
    corrupted, quality, severity = corrupt_groups(
        signal, signal, probability=1.0, groups=groups, mode="dropout"
    )
    assert corrupted.shape == quality.shape == signal.shape
    assert severity.shape == (16, 3)
    assert torch.all((severity > 0).sum(1) <= 2)
