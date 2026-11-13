import math

import torch
import torch.nn as nn

from .model import AttentionPool, _scale_invariant_quality


class GroupTemporalTokenizer(nn.Module):
    """Build a sensor-group by temporal-patch token lattice."""

    def __init__(self, channels, samples, patch, channel_groups, hidden, dropout):
        super().__init__()
        if samples % patch:
            raise ValueError("samples must be divisible by patch")
        self.patch = int(patch)
        self.patches = samples // patch
        self.channel_groups = tuple(
            tuple(range(start, end)) for start, end in channel_groups
        )
        flattened = sorted(index for group in self.channel_groups for index in group)
        if flattened != list(range(channels)):
            raise ValueError("channel_groups must partition every input channel")

        self.patch_norm = nn.LayerNorm(patch)
        self.patch_project = nn.Sequential(
            nn.Linear(patch, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.channel_embedding = nn.Parameter(
            torch.randn(1, 1, channels, hidden) * 0.02
        )
        self.time_embedding = nn.Parameter(
            torch.randn(1, self.patches, 1, hidden) * 0.02
        )
        self.group_embedding = nn.Parameter(
            torch.randn(1, len(self.channel_groups), 1, hidden) * 0.02
        )
        self.channel_pool = AttentionPool(hidden)

    def forward(self, x, quality_x):
        local_raw = (
            x.transpose(1, 2).unfold(2, self.patch, self.patch)
            .permute(0, 2, 1, 3)
        )
        local = (
            self.patch_project(self.patch_norm(local_raw))
            + self.channel_embedding
            + self.time_embedding
        )

        group_patch_tokens = []
        group_patch_statistics = []
        for indices in self.channel_groups:
            group_patch_tokens.append(
                self.channel_pool(local[:, :, list(indices)], dim=2)
            )
            patches = (
                quality_x[:, :, list(indices)]
                .unfold(1, self.patch, self.patch)
                .permute(0, 1, 3, 2)
            )
            group_patch_statistics.append(_scale_invariant_quality(patches))

        tokens = torch.stack(group_patch_tokens, dim=1) + self.group_embedding
        statistics = torch.stack(group_patch_statistics, dim=1)
        return tokens, statistics


class LocalIntegrityHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden + 6),
            nn.Linear(hidden + 6, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, tokens, statistics):
        availability = statistics[..., -1]
        values = torch.cat((tokens, statistics), dim=-1)
        return torch.sigmoid(self.network(values).squeeze(-1)) * availability


class GroupTemporalIntegrityRouter(nn.Module):
    """Use local integrity to pool a group-time token lattice.

    The branch router is deliberately static. The experiment therefore tests
    local degradation handling rather than another sample-conditioned router.
    """

    def __init__(self, channels, samples, patch, classes, channel_groups,
                 hidden=128, dropout=0.2, use_integrity_gate=True):
        super().__init__()
        self.use_integrity_gate = bool(use_integrity_gate)
        self.tokenizer = GroupTemporalTokenizer(
            channels, samples, patch, channel_groups, hidden, dropout,
        )
        self.integrity_head = LocalIntegrityHead(hidden)
        self.sensor_time_score = nn.Linear(hidden, 1)
        self.temporal_group_score = nn.Linear(hidden, 1)
        self.global_pool = AttentionPool(hidden)
        self.sensor_branch_pool = AttentionPool(hidden)
        self.temporal_branch_pool = AttentionPool(hidden)
        self.token_types = nn.Parameter(torch.randn(3, 1, hidden) * 0.02)

        sensor_layer = nn.TransformerEncoderLayer(
            hidden, 4, hidden * 4, dropout, batch_first=True, norm_first=True,
        )
        temporal_layer = nn.TransformerEncoderLayer(
            hidden, 4, hidden * 4, dropout, batch_first=True, norm_first=True,
        )
        self.sensor_encoder = nn.TransformerEncoder(
            sensor_layer, 1, norm=nn.LayerNorm(hidden),
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, 1, norm=nn.LayerNorm(hidden),
        )
        self.sensor_cross = nn.MultiheadAttention(
            hidden, 4, dropout=dropout, batch_first=True,
        )
        self.temporal_cross = nn.MultiheadAttention(
            hidden, 4, dropout=dropout, batch_first=True,
        )
        self.sensor_cross_norm = nn.LayerNorm(hidden)
        self.temporal_cross_norm = nn.LayerNorm(hidden)
        self.branch_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])
        self.static_logits = nn.Parameter(torch.zeros(3))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, classes),
        )

    @staticmethod
    def _masked_softmax(logits, availability, dim):
        minimum = torch.finfo(logits.dtype).min
        masked = logits.masked_fill(~availability, minimum)
        weights = torch.softmax(masked, dim=dim)
        weights = weights * availability.to(weights.dtype)
        return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-6)

    def forward(self, x, quality_x=None, return_aux=False,
                integrity_override=None, force_local_gate=None):
        quality_x = x if quality_x is None else quality_x
        lattice, statistics = self.tokenizer(x, quality_x)
        predicted_integrity = self.integrity_head(lattice, statistics)
        availability = statistics[..., -1] > 0.5

        use_gate = self.use_integrity_gate if force_local_gate is None else force_local_gate
        gate_integrity = predicted_integrity
        if integrity_override is not None:
            gate_integrity = integrity_override.to(lattice.dtype) * availability

        sensor_logits = self.sensor_time_score(lattice).squeeze(-1)
        temporal_logits = self.temporal_group_score(lattice).squeeze(-1)
        if use_gate:
            bias = torch.log(gate_integrity.clamp_min(1e-4))
            sensor_logits = sensor_logits + bias
            temporal_logits = temporal_logits + bias

        sensor_weights = self._masked_softmax(sensor_logits, availability, dim=2)
        temporal_weights = self._masked_softmax(temporal_logits, availability, dim=1)
        sensor = torch.sum(lattice * sensor_weights.unsqueeze(-1), dim=2)
        temporal = torch.sum(lattice * temporal_weights.unsqueeze(-1), dim=1)
        sensor = sensor + self.token_types[0]
        temporal = temporal + self.token_types[1]

        sensor_available = availability.any(dim=2)
        temporal_available = availability.any(dim=1)
        sensor = self.sensor_encoder(sensor, src_key_padding_mask=~sensor_available)
        temporal = self.temporal_encoder(
            temporal, src_key_padding_mask=~temporal_available,
        )
        sensor_delta, _ = self.sensor_cross(
            sensor, temporal, temporal, key_padding_mask=~temporal_available,
            need_weights=False,
        )
        temporal_delta, _ = self.temporal_cross(
            temporal, sensor, sensor, key_padding_mask=~sensor_available,
            need_weights=False,
        )
        sensor = self.sensor_cross_norm(sensor + sensor_delta)
        temporal = self.temporal_cross_norm(temporal + temporal_delta)

        sensor_branch_logits = self.sensor_branch_pool.score(sensor).squeeze(-1)
        temporal_branch_logits = self.temporal_branch_pool.score(temporal).squeeze(-1)
        sensor_branch_weights = self._masked_softmax(
            sensor_branch_logits, sensor_available, dim=1,
        )
        temporal_branch_weights = self._masked_softmax(
            temporal_branch_logits, temporal_available, dim=1,
        )
        sensor_repr = self.branch_norms[0](
            torch.sum(sensor * sensor_branch_weights.unsqueeze(-1), dim=1)
        )
        temporal_repr = self.branch_norms[1](
            torch.sum(temporal * temporal_branch_weights.unsqueeze(-1), dim=1)
        )
        global_logits = self.global_pool.score(temporal).squeeze(-1)
        global_weights = self._masked_softmax(
            global_logits, temporal_available, dim=1,
        )
        global_token = (
            torch.sum(temporal * global_weights.unsqueeze(-1), dim=1)
            + self.token_types[2, 0]
        )
        global_repr = self.branch_norms[2](global_token)
        branches = torch.stack((sensor_repr, temporal_repr, global_repr), dim=1)
        branch_gates = torch.softmax(self.static_logits, dim=0)
        embedding = torch.sum(branches * branch_gates.view(1, 3, 1), dim=1)
        output = self.head(embedding)
        if not return_aux:
            return output

        auxiliary = {
            "gates": branch_gates.unsqueeze(0).expand(len(x), -1),
            "local_integrity": predicted_integrity,
            "local_availability": availability,
            "sensor_reliability": predicted_integrity.mean(dim=2),
            "temporal_reliability": predicted_integrity.mean(dim=1),
            "sensor_time_weights": sensor_weights,
            "temporal_group_weights": temporal_weights,
            "embedding": embedding,
        }
        return output, auxiliary


def local_integrity_loss(predicted, severity):
    """Balanced regression to the known synthetic local integrity target."""

    target = (1.0 - severity).clamp(0.0, 1.0)
    corrupted = severity > 0
    preserved = ~corrupted
    zero = predicted.sum() * 0.0
    corrupt_loss = (
        nn.functional.smooth_l1_loss(predicted[corrupted], target[corrupted])
        if corrupted.any() else zero
    )
    clean_loss = (
        nn.functional.smooth_l1_loss(predicted[preserved], target[preserved])
        if preserved.any() else zero
    )
    return corrupt_loss + 0.25 * clean_loss


def _rand(shape, device, generator, dtype=None):
    return torch.rand(shape, device=device, generator=generator, dtype=dtype)


def corrupt_group_time(x, quality_x, groups, patch, probability=0.75,
                       max_span_fraction=0.4, full_span_probability=0.25,
                       kinds=("dropout", "noise", "flatline"), generator=None,
                       fixed_kind=None, fixed_span_fraction=None):
    """Apply contiguous corruption to one physical group per selected sample."""

    x = x.clone()
    quality_x = quality_x.clone()
    batch, samples, _ = x.shape
    patches = samples // patch
    device = x.device
    dtype = x.dtype
    severity = torch.zeros(batch, len(groups), patches, device=device, dtype=dtype)
    selected = _rand((batch,), device, generator) < probability
    rows = selected.nonzero(as_tuple=False).squeeze(1)
    if not len(rows):
        return x, quality_x, severity

    group_index = torch.randint(
        len(groups), (len(rows),), device=device, generator=generator,
    )
    if fixed_kind is None:
        kind_index = torch.randint(
            len(kinds), (len(rows),), device=device, generator=generator,
        )
        chosen_kinds = [kinds[index] for index in kind_index.cpu().tolist()]
    else:
        chosen_kinds = [fixed_kind] * len(rows)

    if fixed_span_fraction is None:
        max_span = max(1, int(math.ceil(patches * max_span_fraction)))
        span = torch.randint(
            1, max_span + 1, (len(rows),), device=device, generator=generator,
        )
        make_full = _rand((len(rows),), device, generator) < full_span_probability
        span[make_full] = patches
    else:
        span = torch.full(
            (len(rows),), max(1, int(round(patches * fixed_span_fraction))),
            device=device, dtype=torch.long,
        ).clamp_max(patches)
    possible = patches - span + 1
    start = torch.floor(_rand((len(rows),), device, generator) * possible).long()
    patch_axis = torch.arange(patches, device=device).unsqueeze(0)
    patch_mask = (patch_axis >= start.unsqueeze(1)) & (
        patch_axis < (start + span).unsqueeze(1)
    )

    amount = torch.empty(len(rows), device=device, dtype=dtype).uniform_(
        0.5, 1.5, generator=generator,
    )
    kind_severity = {
        "dropout": torch.ones_like(amount),
        "noise": amount / 1.5,
        "flatline": torch.full_like(amount, 0.9),
        "drift": amount / 1.5,
        "spike": torch.ones_like(amount),
        "gain": torch.full_like(amount, 0.8),
    }
    selected_severity = torch.stack([
        kind_severity[kind][offset] for offset, kind in enumerate(chosen_kinds)
    ])
    group_indicator = torch.nn.functional.one_hot(
        group_index, num_classes=len(groups),
    ).to(dtype)
    severity[rows] = (
        group_indicator.unsqueeze(-1)
        * patch_mask.unsqueeze(1).to(dtype)
        * selected_severity.view(-1, 1, 1)
    )

    sample_mask = patch_mask.repeat_interleave(patch, dim=1)
    sample_axis = torch.arange(samples, device=device).unsqueeze(0)
    sample_start = (start * patch).unsqueeze(1)
    sample_length = (span * patch).unsqueeze(1).clamp_min(1)
    local_position = ((sample_axis - sample_start) / sample_length).clamp(0, 1)
    local_ramp = 2.0 * local_position - 1.0

    for group, (channel_start, channel_end) in enumerate(groups):
        group_rows = group_index == group
        if not group_rows.any():
            continue
        local_rows = rows[group_rows]
        group_masks = sample_mask[group_rows].unsqueeze(-1)
        group_ramps = local_ramp[group_rows].unsqueeze(-1)
        group_amount = amount[group_rows].view(-1, 1, 1)
        group_kinds = [
            kind for keep, kind in zip(group_rows.cpu().tolist(), chosen_kinds) if keep
        ]
        for kind in set(group_kinds):
            kind_keep = torch.tensor(
                [value == kind for value in group_kinds], device=device,
                dtype=torch.bool,
            )
            kind_rows = local_rows[kind_keep]
            mask = group_masks[kind_keep]
            ramp = group_ramps[kind_keep]
            strength = group_amount[kind_keep]
            for value in (x, quality_x):
                subset = value[kind_rows, :, channel_start:channel_end]
                scale = subset.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
                if kind == "dropout":
                    replacement = torch.zeros_like(subset)
                elif kind == "noise":
                    noise = torch.randn(
                        subset.shape, device=device, dtype=dtype, generator=generator,
                    )
                    replacement = subset + noise * scale * strength
                elif kind == "flatline":
                    replacement = subset.mean(dim=1, keepdim=True).expand_as(subset)
                elif kind == "drift":
                    replacement = subset + ramp * scale * strength
                elif kind == "spike":
                    impulses = (
                        _rand(subset.shape, device, generator, dtype=dtype) < 0.03
                    ).to(dtype)
                    signs = torch.where(
                        _rand(subset.shape, device, generator, dtype=dtype) < 0.5,
                        -torch.ones_like(subset), torch.ones_like(subset),
                    )
                    replacement = subset + impulses * signs * scale * 6.0
                elif kind == "gain":
                    replacement = subset * 3.0
                else:
                    raise ValueError(f"unknown local corruption: {kind}")
                value[kind_rows, :, channel_start:channel_end] = torch.where(
                    mask, replacement, subset,
                )
    return x, quality_x, severity
