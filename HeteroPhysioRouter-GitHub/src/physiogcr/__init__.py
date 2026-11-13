"""Public API for PhysioGCR.

The historical ``heterophysiorouter`` namespace remains available so existing
experiment checkpoints and scripts keep working.
"""

from heterophysiorouter import (
    CORRUPTION_MODES,
    GATE_RULES,
    PILOT_MODES,
    VARIANTS,
    GroupTemporalIntegrityRouter,
    QualityAwareRouter,
    TaskRelevanceLocalAdapter,
    balanced_reliability_loss,
    clean_identity_loss,
    combine_joint_gate,
    corrupt_group_time,
    corrupt_groups,
    local_integrity_loss,
    monotonic_calibration_loss,
    sampled_relevance_loss,
)


PhysioGCR = QualityAwareRouter

__all__ = [
    "CORRUPTION_MODES",
    "GATE_RULES",
    "PILOT_MODES",
    "VARIANTS",
    "GroupTemporalIntegrityRouter",
    "PhysioGCR",
    "QualityAwareRouter",
    "TaskRelevanceLocalAdapter",
    "balanced_reliability_loss",
    "clean_identity_loss",
    "combine_joint_gate",
    "corrupt_group_time",
    "corrupt_groups",
    "local_integrity_loss",
    "monotonic_calibration_loss",
    "sampled_relevance_loss",
]
