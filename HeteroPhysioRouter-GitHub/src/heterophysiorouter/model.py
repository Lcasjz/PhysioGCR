import math

import torch
import torch.nn as nn


VARIANTS = (
    "static",
    "dynamic",
    "quality_proxy",
    "quality_calibrated",
    "cross_dynamic",
    "cross_quality",
    "full",
    "full_no_cross",
    "full_no_calibration",
    "degrade_no_quality",
    "degrade_static_router",
    "no_degradation",
    "degrade_no_sensor_tokens",
    "degrade_no_temporal_tokens",
    "degrade_no_global_token",
    "degrade_dropout_only",
    "degrade_noise_only",
    "degrade_flatline_only",
    "degrade_mixed_reference",
)


class AttentionPool(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x, dim, return_weights=False):
        weight = torch.softmax(self.score(x).squeeze(-1), dim=dim)
        pooled = torch.sum(weight.unsqueeze(-1) * x, dim=dim)
        return (pooled, weight) if return_weights else pooled


def _scale_invariant_quality(x):
    """Return roughness, flatness, outliers, crest, spectral entropy, availability.

    x is [..., time, channels]. Features are computed per channel and averaged,
    so heterogeneous units never get mixed before scale normalization.
    """
    availability = (x.abs().amax(dim=(-2, -1)) > 1e-7).float()
    centered = x - x.mean(dim=-2, keepdim=True)
    scale = centered.square().mean(dim=-2, keepdim=True).sqrt().clamp_min(1e-6)
    z = centered / scale
    difference = z[..., 1:, :] - z[..., :-1, :]
    roughness = difference.square().mean(dim=(-2, -1)).sqrt() / math.sqrt(2.0)
    flatness = (difference.abs() < .02).float().mean(dim=(-2, -1))
    outliers = (z.abs() > 4).float().mean(dim=(-2, -1))
    crest = z.abs().amax(dim=-2).mean(dim=-1).clamp(max=10) / 10
    spectrum = torch.fft.rfft(z, dim=-2).abs().square().clamp_min(1e-10)
    probability = spectrum / spectrum.sum(dim=-2, keepdim=True)
    entropy = -(probability * probability.log()).sum(dim=-2)
    entropy = entropy / math.log(max(spectrum.shape[-2], 2))
    spectral_entropy = entropy.mean(dim=-1)
    return torch.stack((roughness, flatness, outliers, crest, spectral_entropy, availability), dim=-1)


class QualityEstimator(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(6), nn.Linear(6, hidden // 2), nn.GELU(),
                                     nn.Linear(hidden // 2, 1))

    def forward(self, statistics):
        availability = statistics[..., -1]
        return torch.sigmoid(self.network(statistics).squeeze(-1)) * availability


class GroupedTokenizer(nn.Module):
    def __init__(self, channels, samples, patch, channel_groups, hidden, dropout,
                 channel_group_indices=None):
        super().__init__()
        if samples % patch:
            raise ValueError("samples must be divisible by patch")
        self.patch = patch
        self.patches = samples // patch
        self.channel_groups = tuple(tuple(group) for group in channel_groups)
        if channel_group_indices is None:
            channel_group_indices = tuple(
                tuple(range(start, end)) for start, end in self.channel_groups
            )
        self.channel_group_indices = tuple(
            tuple(int(index) for index in group) for group in channel_group_indices
        )
        flattened = [index for group in self.channel_group_indices for index in group]
        if sorted(flattened) != list(range(channels)):
            raise ValueError(
                "channel_group_indices must partition every input channel exactly once"
            )
        self.patch_norm = nn.LayerNorm(patch)
        self.patch_project = nn.Sequential(nn.Linear(patch, hidden), nn.GELU(), nn.Dropout(dropout))
        self.channel_embedding = nn.Parameter(torch.randn(1, 1, channels, hidden) * .02)
        self.time_embedding = nn.Parameter(torch.randn(1, self.patches, 1, hidden) * .02)
        self.group_embedding = nn.Parameter(
            torch.randn(1, len(self.channel_group_indices), hidden) * .02
        )
        self.temporal_pool = AttentionPool(hidden)
        self.group_pool = AttentionPool(hidden)
        self.global_pool = AttentionPool(hidden)
        self.token_types = nn.Parameter(torch.randn(3, 1, hidden) * .02)

    def encode_local(self, x):
        local_raw = x.transpose(1, 2).unfold(2, self.patch, self.patch).permute(0, 2, 1, 3)
        return (
            self.patch_project(self.patch_norm(local_raw))
            + self.channel_embedding + self.time_embedding
        )

    def aggregate_local(self, local, quality_x, return_diagnostics=False):
        temporal = self.temporal_pool(local, dim=2) + self.token_types[0]
        sensor = []
        sensor_pool_weights = []
        group_statistics = []
        for indices in self.channel_group_indices:
            grouped = local[:, :, list(indices)].reshape(
                len(local), -1, local.shape[-1],
            )
            if return_diagnostics:
                pooled, weights = self.group_pool(grouped, dim=1, return_weights=True)
                sensor_pool_weights.append(weights)
            else:
                pooled = self.group_pool(grouped, dim=1)
            sensor.append(pooled)
            group_statistics.append(
                _scale_invariant_quality(quality_x[:, :, list(indices)])
            )
        sensor = torch.stack(sensor, dim=1) + self.group_embedding + self.token_types[1]
        group_statistics = torch.stack(group_statistics, dim=1)
        if return_diagnostics:
            global_pooled, global_pool_weights = self.global_pool(
                temporal, dim=1, return_weights=True,
            )
        else:
            global_pooled = self.global_pool(temporal, dim=1)
        global_token = global_pooled.unsqueeze(1) + self.token_types[2]

        quality_patches = quality_x.unfold(1, self.patch, self.patch).permute(0, 1, 3, 2)
        temporal_statistics = _scale_invariant_quality(quality_patches)
        outputs = (temporal, sensor, global_token, temporal_statistics, group_statistics)
        if not return_diagnostics:
            return outputs
        diagnostics = {
            "global_pool_weights": global_pool_weights,
            "sensor_pool_weights": torch.stack(sensor_pool_weights, dim=1),
        }
        return outputs + (diagnostics,)

    def forward(self, x, quality_x, return_diagnostics=False, local_override=None):
        local = self.encode_local(x) if local_override is None else local_override
        return self.aggregate_local(
            local, quality_x, return_diagnostics=return_diagnostics,
        )


class QualityAwareRouter(nn.Module):
    def __init__(self, channels, samples, patch, classes, channel_groups, variant,
                 hidden=128, dropout=.2, channel_group_indices=None,
                 sensor_token_mask_probability=0.0, sensor_fusion="weighted"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        self.variant = variant
        final_family = variant.startswith("degrade_") or variant == "no_degradation"
        self.use_cross = (
            variant.startswith("cross_")
            or variant in ("full", "full_no_calibration")
            or final_family
        )
        if variant in ("degrade_no_sensor_tokens", "degrade_no_temporal_tokens"):
            self.use_cross = False
        self.use_quality = variant in (
            "quality_proxy", "quality_calibrated", "cross_quality", "full", "full_no_cross",
            "full_no_calibration",
        ) or (final_family and variant != "degrade_no_quality")
        self.static_router = variant in ("static", "degrade_static_router")
        self.use_sensor_tokens = variant != "degrade_no_sensor_tokens"
        self.use_temporal_tokens = variant != "degrade_no_temporal_tokens"
        self.use_global_token = variant != "degrade_no_global_token"
        self.sensor_token_mask_probability = float(sensor_token_mask_probability)
        if sensor_fusion not in ("weighted", "embrace"):
            raise ValueError(f"unknown sensor fusion: {sensor_fusion}")
        self.sensor_fusion = sensor_fusion
        self.tokenizer = GroupedTokenizer(
            channels, samples, patch, channel_groups, hidden, dropout,
            channel_group_indices=channel_group_indices,
        )
        self.sensor_quality = QualityEstimator(hidden)
        self.temporal_quality = QualityEstimator(hidden)

        sensor_layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 4, dropout,
                                                  batch_first=True, norm_first=True)
        temporal_layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 4, dropout,
                                                    batch_first=True, norm_first=True)
        self.sensor_encoder = nn.TransformerEncoder(sensor_layer, 1, norm=nn.LayerNorm(hidden))
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, 1, norm=nn.LayerNorm(hidden))
        self.sensor_cross = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
        self.temporal_cross = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
        self.sensor_cross_norm = nn.LayerNorm(hidden)
        self.temporal_cross_norm = nn.LayerNorm(hidden)

        self.branch_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])
        self.static_logits = nn.Parameter(torch.zeros(3))
        self.router = nn.Sequential(nn.LayerNorm(hidden * 3 + 10), nn.Linear(hidden * 3 + 10, hidden),
                                    nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 3))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, classes))

    @staticmethod
    def _weighted_pool(tokens, reliability):
        weight = reliability / reliability.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.sum(tokens * weight.unsqueeze(-1), dim=1)

    @staticmethod
    def _summary(sensor_reliability, temporal_reliability):
        def describe(value):
            return torch.stack((value.mean(1), value.std(1, unbiased=False), value.amin(1), value.amax(1)), dim=1)
        sensor_missing = (sensor_reliability <= 1e-7).float().mean(1, keepdim=True)
        temporal_missing = (temporal_reliability <= 1e-7).float().mean(1, keepdim=True)
        return torch.cat((describe(sensor_reliability), describe(temporal_reliability),
                          sensor_missing, temporal_missing), dim=1)

    def forward(self, x, quality_x=None, return_aux=False, return_diagnostics=False,
                local_override=None):
        quality_x = x if quality_x is None else quality_x
        tokenizer_output = self.tokenizer(
            x, quality_x, return_diagnostics=return_diagnostics,
            local_override=local_override,
        )
        temporal, sensor, global_token, temporal_stats, sensor_stats = tokenizer_output[:5]
        diagnostics = tokenizer_output[5] if return_diagnostics else None
        if self.use_quality:
            sensor_reliability = self.sensor_quality(sensor_stats)
            temporal_reliability = self.temporal_quality(temporal_stats)
        else:
            sensor_reliability = sensor_stats[..., -1]
            temporal_reliability = temporal_stats[..., -1]

        if self.training and self.sensor_token_mask_probability > 0:
            token_mask = torch.rand(
                sensor.shape[:2], device=sensor.device,
            ) < self.sensor_token_mask_probability
            all_masked = token_mask.all(1)
            if all_masked.any():
                rows = all_masked.nonzero(as_tuple=False).squeeze(1)
                keep = torch.randint(sensor.shape[1], (len(rows),), device=sensor.device)
                token_mask[rows, keep] = False
            sensor = sensor.masked_fill(token_mask.unsqueeze(-1), 0)
            sensor_reliability = sensor_reliability.masked_fill(token_mask, 0)

        sensor_missing = sensor_reliability <= 1e-7
        temporal_missing = temporal_reliability <= 1e-7
        sensor = self.sensor_encoder(sensor, src_key_padding_mask=sensor_missing)
        temporal = self.temporal_encoder(temporal, src_key_padding_mask=temporal_missing)
        if self.use_cross:
            sensor_delta, sensor_cross_weights = self.sensor_cross(
                sensor, temporal, temporal, key_padding_mask=temporal_missing,
                need_weights=return_diagnostics, average_attn_weights=False,
            )
            temporal_delta, temporal_cross_weights = self.temporal_cross(
                temporal, sensor, sensor, key_padding_mask=sensor_missing,
                need_weights=return_diagnostics, average_attn_weights=False,
            )
            sensor = self.sensor_cross_norm(sensor + sensor_delta)
            temporal = self.temporal_cross_norm(temporal + temporal_delta)
        elif return_diagnostics:
            sensor_cross_weights = None
            temporal_cross_weights = None

        if self.sensor_fusion == "embrace":
            availability = (sensor_reliability > 1e-7).to(sensor.dtype)
            probabilities = availability / availability.sum(1, keepdim=True).clamp_min(1)
            if self.training:
                selected = torch.multinomial(
                    probabilities, sensor.shape[-1], replacement=True,
                )
                sensor_repr = torch.gather(
                    sensor, 1, selected.unsqueeze(1),
                ).squeeze(1)
            else:
                sensor_repr = torch.sum(sensor * probabilities.unsqueeze(-1), dim=1)
            sensor_repr = self.branch_norms[0](sensor_repr)
        else:
            sensor_repr = self.branch_norms[0](
                self._weighted_pool(sensor, sensor_reliability)
            )
        temporal_repr = self.branch_norms[1](self._weighted_pool(temporal, temporal_reliability))
        global_repr = self.branch_norms[2](global_token[:, 0])
        active = torch.tensor(
            (self.use_sensor_tokens, self.use_temporal_tokens, self.use_global_token),
            device=x.device,
            dtype=torch.bool,
        )
        sensor_repr = sensor_repr if self.use_sensor_tokens else torch.zeros_like(sensor_repr)
        temporal_repr = temporal_repr if self.use_temporal_tokens else torch.zeros_like(temporal_repr)
        global_repr = global_repr if self.use_global_token else torch.zeros_like(global_repr)
        branches = torch.stack((sensor_repr, temporal_repr, global_repr), dim=1)
        if self.static_router:
            route_logits = self.static_logits.unsqueeze(0).expand(len(x), -1)
        else:
            if self.use_quality:
                summary_sensor = sensor_reliability if self.use_sensor_tokens else torch.zeros_like(sensor_reliability)
                summary_temporal = temporal_reliability if self.use_temporal_tokens else torch.zeros_like(temporal_reliability)
                quality_summary = self._summary(summary_sensor, summary_temporal)
            else:
                quality_summary = torch.zeros(len(x), 10, device=x.device, dtype=x.dtype)
            route_logits = self.router(torch.cat((sensor_repr, temporal_repr, global_repr, quality_summary), dim=1))
        route_logits = route_logits.masked_fill(~active.unsqueeze(0), torch.finfo(route_logits.dtype).min)
        gates = torch.softmax(route_logits, dim=1)
        embedding = torch.sum(branches * gates.unsqueeze(-1), dim=1)
        output = self.head(embedding)
        if not return_aux and not return_diagnostics:
            return output
        auxiliary = {"gates": gates, "sensor_reliability": sensor_reliability,
                     "temporal_reliability": temporal_reliability, "embedding": embedding,
                     "sensor_statistics": sensor_stats, "temporal_statistics": temporal_stats}
        if return_diagnostics:
            auxiliary.update(diagnostics)
            auxiliary["sensor_to_temporal_attention"] = sensor_cross_weights
            auxiliary["temporal_to_sensor_attention"] = temporal_cross_weights
        return output, auxiliary


CORRUPTION_MODES = (
    "mixed",
    "dropout",
    "noise",
    "flatline",
    "dropout_flatline",
    "dropout_noise",
    "noise_flatline",
)


def corrupt_groups(x, quality_x, probability, groups, mode="mixed"):
    """Apply controlled group degradation and return its severity in [0, 1]."""
    if probability <= 0:
        return x, quality_x, torch.zeros(len(x), len(groups), device=x.device)
    x = x.clone(); quality_x = quality_x.clone()
    severity_target = torch.zeros(len(x), len(groups), device=x.device)
    selected = torch.rand(len(x), len(groups), device=x.device) < probability
    all_selected = selected.all(1)
    if all_selected.any():
        rows = all_selected.nonzero(as_tuple=False).squeeze(1)
        selected[rows, torch.randint(0, len(groups), (len(rows),), device=x.device)] = False
    for group, (start, end) in enumerate(groups):
        rows = selected[:, group].nonzero(as_tuple=False).squeeze(1)
        if not len(rows):
            continue
        kind_sets = {
            "mixed": (0, 1, 2),
            "dropout": (0,),
            "noise": (1,),
            "flatline": (2,),
            "dropout_flatline": (0, 2),
            "dropout_noise": (0, 1),
            "noise_flatline": (1, 2),
        }
        if mode not in kind_sets:
            raise ValueError(f"unknown group corruption mode: {mode}")
        kinds = torch.tensor(kind_sets[mode], dtype=torch.long, device=x.device)
        degradation = kinds[torch.randint(0, len(kinds), (len(rows),), device=x.device)]
        for kind in range(3):
            kind_rows = rows[degradation == kind]
            if not len(kind_rows):
                continue
            if kind == 0:
                x[kind_rows, :, start:end] = 0; quality_x[kind_rows, :, start:end] = 0
                severity_target[kind_rows, group] = 1
            elif kind == 1:
                severity = torch.empty(len(kind_rows), 1, 1, device=x.device).uniform_(.5, 1.5)
                for tensor in (x, quality_x):
                    segment = tensor[kind_rows, :, start:end]
                    scale = segment.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
                    tensor[kind_rows, :, start:end] = segment + torch.randn_like(segment) * scale * severity
                severity_target[kind_rows, group] = severity[:, 0, 0] / 1.5
            else:
                for tensor in (x, quality_x):
                    segment = tensor[kind_rows, :, start:end]
                    tensor[kind_rows, :, start:end] = segment.mean(dim=1, keepdim=True)
                severity_target[kind_rows, group] = .9
    return x, quality_x, severity_target


def monotonic_calibration_loss(clean_aux, corrupted_aux, severity):
    """Calibrate relative reliability without regressing an absolute noise level."""
    clean_sensor = clean_aux["sensor_reliability"]
    corrupted_sensor = corrupted_aux["sensor_reliability"]
    selected = severity > 0
    margin = .35 * severity
    ranking = torch.relu(margin - (clean_sensor - corrupted_sensor))
    selected_loss = ranking[selected].mean() if selected.any() else ranking.sum() * 0
    preserved = ~selected
    preservation_loss = ((clean_sensor - corrupted_sensor).square()[preserved].mean()
                         if preserved.any() else ranking.sum() * 0)

    clean_temporal = clean_aux["temporal_reliability"].mean(1)
    corrupted_temporal = corrupted_aux["temporal_reliability"].mean(1)
    sample_severity = severity.mean(1)
    affected = sample_severity > 0
    temporal_ranking = torch.relu(.2 * sample_severity - (clean_temporal - corrupted_temporal))
    temporal_loss = temporal_ranking[affected].mean() if affected.any() else temporal_ranking.sum() * 0
    return selected_loss + .25 * preservation_loss + .5 * temporal_loss
