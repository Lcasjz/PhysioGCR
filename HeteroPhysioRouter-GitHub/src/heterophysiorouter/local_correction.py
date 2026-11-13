import torch
import torch.nn as nn

from .model import QualityAwareRouter, _scale_invariant_quality


PILOT_MODES = (
    "frozen",
    "plain",
    "availability",
    "reliability",
    "relevance",
    "joint",
    "oracle",
)

GATE_RULES = ("product", "geometric", "minimum", "soft_or")


def combine_joint_gate(severity, relevance, rule="product", scale=1.0,
                       temperature=1.0):
    if rule == "product":
        joint = severity * relevance
    elif rule == "geometric":
        joint = torch.sqrt((severity * relevance).clamp_min(0.0))
    elif rule == "minimum":
        joint = torch.minimum(severity, relevance)
    elif rule == "soft_or":
        temperature = max(float(temperature), 1e-4)
        severity_logit = torch.logit(severity.clamp(1e-5, 1.0 - 1e-5))
        relevance_logit = torch.logit(relevance.clamp(1e-5, 1.0 - 1e-5))
        joint = torch.sigmoid((severity_logit + relevance_logit) / temperature)
    else:
        raise ValueError(f"unknown gate rule: {rule}")
    return (float(scale) * joint).clamp(0.0, 1.0)


class LocalSignalStatistics(nn.Module):
    def __init__(self, patch, channel_groups):
        super().__init__()
        self.patch = int(patch)
        self.channel_groups = tuple(
            tuple(range(start, end)) for start, end in channel_groups
        )

    def forward(self, signal):
        outputs = []
        for indices in self.channel_groups:
            patches = (
                signal[:, :, list(indices)].unfold(1, self.patch, self.patch)
                .permute(0, 1, 3, 2).contiguous()
            )
            outputs.append(_scale_invariant_quality(patches))
        return torch.stack(outputs, dim=1)


class ReliabilityHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        width = max(hidden // 4, 16)
        self.network = nn.Sequential(
            nn.LayerNorm(6), nn.Linear(6, width), nn.GELU(), nn.Linear(width, 1),
        )

    def forward(self, statistics):
        availability = statistics[..., -1]
        predicted = torch.sigmoid(self.network(statistics).squeeze(-1))
        return predicted * availability + (1.0 - availability)


class RelevanceHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        width = max(hidden // 2, 32)
        self.network = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, width), nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, group_tokens):
        return torch.sigmoid(self.network(group_tokens).squeeze(-1))


class ContextResidualCorrection(nn.Module):
    def __init__(self, hidden, groups, rank=8):
        super().__init__()
        self.norm = nn.LayerNorm(hidden * 3)
        self.down = nn.Linear(hidden * 3, rank)
        self.activation = nn.GELU()
        self.up = nn.ModuleList([nn.Linear(rank, hidden) for _ in range(groups)])
        for layer in self.up:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def temporal_context(tokens):
        left = torch.cat((tokens[:, :, :1], tokens[:, :, :-1]), dim=2)
        right = torch.cat((tokens[:, :, 1:], tokens[:, :, -1:]), dim=2)
        return 0.5 * (left + right)

    @staticmethod
    def cross_group_context(tokens):
        if tokens.size(1) == 1:
            return tokens
        total = tokens.sum(dim=1, keepdim=True)
        return (total - tokens) / (tokens.size(1) - 1)

    def forward(self, tokens):
        temporal = self.temporal_context(tokens)
        cross_group = self.cross_group_context(tokens)
        features = torch.cat(
            (tokens, temporal - tokens, cross_group - tokens), dim=-1,
        )
        latent = self.activation(self.down(self.norm(features)))
        return torch.stack([
            layer(latent[:, group]) for group, layer in enumerate(self.up)
        ], dim=1)


class TaskRelevanceLocalAdapter(nn.Module):
    """Frozen recognizer with a pre-pooling local residual adapter."""

    supports_local_overrides = True

    def __init__(self, channels, samples, patch, classes, channel_groups, mode,
                 hidden=128, rank=8, dropout=0.2, gate_rule="product",
                 gate_scale=1.0, gate_temperature=1.0):
        super().__init__()
        if mode not in PILOT_MODES:
            raise ValueError(f"unknown pilot mode: {mode}")
        self.mode = mode
        if gate_rule not in GATE_RULES:
            raise ValueError(f"unknown gate rule: {gate_rule}")
        self.gate_rule = gate_rule
        self.gate_scale = float(gate_scale)
        self.gate_temperature = float(gate_temperature)
        self.groups = tuple(
            tuple(range(start, end)) for start, end in channel_groups
        )
        self.base = QualityAwareRouter(
            channels, samples, patch, classes, channel_groups,
            "full_no_calibration", hidden=hidden, dropout=dropout,
        )
        self.statistics = LocalSignalStatistics(patch, channel_groups)
        self.reliability_head = ReliabilityHead(hidden)
        self.relevance_head = RelevanceHead(hidden)
        self.correction = ContextResidualCorrection(
            hidden, len(self.groups), rank=rank,
        )
        self.freeze_base()
        if mode == "frozen":
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def freeze_base(self):
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def group_tokens(self, local_tokens):
        return torch.stack([
            local_tokens[:, :, list(indices)].mean(dim=2)
            for indices in self.groups
        ], dim=1)

    def apply_group_correction(self, local_tokens, correction, gate):
        corrected = local_tokens.clone()
        for group, indices in enumerate(self.groups):
            delta = gate[:, group].unsqueeze(-1).unsqueeze(-1) * correction[:, group].unsqueeze(2)
            corrected[:, :, list(indices)] = corrected[:, :, list(indices)] + delta
        return corrected

    def adapter_gate(self, predicted_severity, predicted_relevance,
                     availability_severity, severity_override,
                     relevance_override, force_gate):
        if force_gate is False or self.mode == "frozen":
            return torch.zeros_like(predicted_severity)
        if self.mode == "plain":
            return torch.ones_like(predicted_severity)
        if self.mode == "availability":
            return availability_severity.detach()
        severity = (
            severity_override.to(predicted_severity.dtype)
            if severity_override is not None else predicted_severity
        )
        relevance = (
            relevance_override.to(predicted_relevance.dtype)
            if relevance_override is not None else predicted_relevance
        )
        if self.mode == "reliability":
            return severity.detach()
        if self.mode == "relevance":
            return relevance.detach()
        if self.mode in ("joint", "oracle"):
            return combine_joint_gate(
                severity, relevance, self.gate_rule,
                self.gate_scale, self.gate_temperature,
            ).detach()
        return torch.zeros_like(predicted_severity)

    def forward(self, x, quality_x=None, return_aux=False,
                severity_override=None, relevance_override=None,
                force_gate=None):
        quality_x = x if quality_x is None else quality_x
        with torch.no_grad():
            local_tokens = self.base.tokenizer.encode_local(x)
        group_tokens = self.group_tokens(local_tokens)
        statistics = self.statistics(quality_x)
        predicted_severity = self.reliability_head(statistics)
        predicted_relevance = self.relevance_head(group_tokens.detach())
        availability_severity = 1.0 - statistics[..., -1]
        gate = self.adapter_gate(
            predicted_severity, predicted_relevance, availability_severity,
            severity_override, relevance_override, force_gate,
        )
        correction = self.correction(group_tokens)
        corrected_local = self.apply_group_correction(
            local_tokens, correction, gate,
        )
        logits = self.base(
            x, quality_x, local_override=corrected_local,
        )
        if not return_aux:
            return logits
        return logits, {
            "local_severity": predicted_severity,
            "local_reliability": 1.0 - predicted_severity,
            "local_relevance": predicted_relevance,
            "local_availability": statistics[..., -1],
            "adapter_gate": gate,
            "adapter_gate_mean": gate.mean(dim=(1, 2)),
            "correction_norm": correction.norm(dim=-1),
            "applied_correction_norm": (correction * gate.unsqueeze(-1)).norm(dim=-1),
            "base_local_tokens": local_tokens,
            "corrected_local_tokens": corrected_local,
        }


def balanced_reliability_loss(predicted_severity, target_severity):
    corrupted = target_severity > 0
    preserved = ~corrupted
    zero = predicted_severity.sum() * 0.0
    corrupted_loss = (
        nn.functional.smooth_l1_loss(
            predicted_severity[corrupted], target_severity[corrupted],
        ) if corrupted.any() else zero
    )
    preserved_loss = (
        nn.functional.smooth_l1_loss(
            predicted_severity[preserved], target_severity[preserved],
        ) if preserved.any() else zero
    )
    return corrupted_loss + preserved_loss


def sampled_relevance_loss(predicted, target, supervised_mask):
    if not supervised_mask.any():
        return predicted.sum() * 0.0
    return nn.functional.smooth_l1_loss(
        predicted[supervised_mask], target[supervised_mask],
    )


def clean_identity_loss(adapted_logits, base_logits, auxiliary):
    teacher = torch.softmax(base_logits.detach(), dim=1)
    consistency = nn.functional.kl_div(
        torch.log_softmax(adapted_logits, dim=1), teacher,
        reduction="batchmean",
    )
    correction = auxiliary["applied_correction_norm"].square().mean()
    return consistency + 0.01 * correction
