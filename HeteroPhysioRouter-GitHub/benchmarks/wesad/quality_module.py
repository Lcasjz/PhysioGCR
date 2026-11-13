"""Heterogeneous WESAD adaptation of the established quality router.

The design follows :class:`heterophysiorouter.QualityAwareRouter`'s
``full_no_calibration`` variant: grouped quality proxies, sensor/temporal/global
branches, bidirectional cross-attention, dynamic routing, and group corruption.
WESAD's four original modality streams are the sensor groups here.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


MODALITY_NAMES = ("resp", "bvp", "ecg", "eda")
MODALITY_DIMS = (14, 4, 14, 4)
MODALITY_LENGTHS = (50, 16, 50, 1)
PATCH_SIZES = (5, 4, 5, 1)


def scale_invariant_quality(value):
    """Six scale-invariant quality proxies for [..., time, channels]."""
    availability = (value.abs().amax(dim=(-2, -1)) > 1e-7).float()
    centered = value - value.mean(dim=-2, keepdim=True)
    scale = centered.square().mean(dim=-2, keepdim=True).sqrt().clamp_min(1e-6)
    normalized = centered / scale
    if value.size(-2) > 1:
        difference = normalized[..., 1:, :] - normalized[..., :-1, :]
        roughness = difference.square().mean(dim=(-2, -1)).sqrt() / math.sqrt(2.0)
        flatness = (difference.abs() < 0.02).float().mean(dim=(-2, -1))
    else:
        roughness = normalized.new_zeros(normalized.shape[:-2])
        flatness = normalized.new_zeros(normalized.shape[:-2])
    outliers = (normalized.abs() > 4).float().mean(dim=(-2, -1))
    crest = normalized.abs().amax(dim=-2).mean(dim=-1).clamp(max=10) / 10
    spectrum = torch.fft.rfft(normalized, dim=-2).abs().square().clamp_min(1e-10)
    probability = spectrum / spectrum.sum(dim=-2, keepdim=True)
    entropy = -(probability * probability.log()).sum(dim=-2)
    spectral_entropy = entropy.mean(dim=-1) / math.log(max(spectrum.shape[-2], 2))
    return torch.stack((roughness, flatness, outliers, crest, spectral_entropy, availability), dim=-1)


class AttentionPool(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, values, dim):
        weights = torch.softmax(self.score(values).squeeze(-1), dim=dim)
        return torch.sum(weights.unsqueeze(-1) * values, dim=dim)


class QualityEstimator(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(6), nn.Linear(6, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1),
        )

    def forward(self, statistics):
        availability = statistics[..., -1]
        return torch.sigmoid(self.network(statistics).squeeze(-1)) * availability


class QualityAwareCrossGranularRouter(nn.Module):
    """Route modality, temporal-patch, and global representations by quality."""

    def __init__(self, hidden=128, dropout=0.2):
        super().__init__()
        self.patch_projects = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(size * dim), nn.Linear(size * dim, hidden),
                          nn.GELU(), nn.Dropout(dropout))
            for size, dim in zip(PATCH_SIZES, MODALITY_DIMS)
        ])
        self.modality_embedding = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(1, 25, hidden) * 0.02)
        self.temporal_pool = AttentionPool(hidden)
        self.global_pool = AttentionPool(hidden)
        self.token_types = nn.Parameter(torch.randn(3, 1, hidden) * 0.02)

        self.modality_quality = QualityEstimator(hidden)
        self.temporal_quality = QualityEstimator(hidden)
        sensor_layer = nn.TransformerEncoderLayer(
            hidden, 4, hidden * 4, dropout, batch_first=True, norm_first=True,
        )
        temporal_layer = nn.TransformerEncoderLayer(
            hidden, 4, hidden * 4, dropout, batch_first=True, norm_first=True,
        )
        self.sensor_encoder = nn.TransformerEncoder(sensor_layer, 1, norm=nn.LayerNorm(hidden))
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, 1, norm=nn.LayerNorm(hidden))
        self.sensor_cross = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
        self.temporal_cross = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
        self.sensor_cross_norm = nn.LayerNorm(hidden)
        self.temporal_cross_norm = nn.LayerNorm(hidden)
        self.branch_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])
        self.router = nn.Sequential(
            nn.LayerNorm(hidden * 3 + 10), nn.Linear(hidden * 3 + 10, hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 3),
        )
        self.output_norm = nn.LayerNorm(hidden)

    @staticmethod
    def summarize(sensor_quality, temporal_quality):
        def describe(value):
            return torch.stack((
                value.mean(1), value.std(1, unbiased=False), value.amin(1), value.amax(1),
            ), dim=1)
        sensor_missing = (sensor_quality <= 1e-7).float().mean(1, keepdim=True)
        temporal_missing = (temporal_quality <= 1e-7).float().mean(1, keepdim=True)
        return torch.cat((describe(sensor_quality), describe(temporal_quality),
                          sensor_missing, temporal_missing), dim=1)

    def forward(self, modalities, quality_modalities=None, return_aux=False):
        quality_modalities = modalities if quality_modalities is None else quality_modalities
        patches, patch_stats, modality_stats = [], [], []
        for value, quality_value, patch_size, project in zip(
                modalities, quality_modalities, PATCH_SIZES, self.patch_projects):
            batch, length, channels = value.shape
            raw_patches = value.unfold(1, patch_size, patch_size)
            raw_patches = raw_patches.permute(0, 1, 3, 2).contiguous()
            encoded = project(raw_patches.flatten(2))
            patches.append(encoded)
            modality_stats.append(scale_invariant_quality(quality_value))
            quality_patches = quality_value.unfold(1, patch_size, patch_size)
            quality_patches = quality_patches.permute(0, 1, 3, 2).contiguous()
            patch_stats.append(torch.stack([
                scale_invariant_quality(quality_patches[:, index])
                for index in range(quality_patches.size(1))
            ], dim=1))

        temporal = torch.cat(patches, dim=1) + self.time_embedding
        modality_tokens = torch.stack([
            self.temporal_pool(tokens, dim=1) for tokens in patches
        ], dim=1) + self.modality_embedding
        global_token = self.global_pool(temporal, dim=1).unsqueeze(1)
        temporal = temporal + self.token_types[0]
        modality_tokens = modality_tokens + self.token_types[1]
        global_token = global_token + self.token_types[2]

        modality_stats = torch.stack(modality_stats, dim=1)
        temporal_stats = torch.cat(patch_stats, dim=1)
        modality_reliability = self.modality_quality(modality_stats)
        temporal_reliability = self.temporal_quality(temporal_stats)
        modality_missing = modality_reliability <= 1e-7
        temporal_missing = temporal_reliability <= 1e-7

        modality_tokens = self.sensor_encoder(
            modality_tokens, src_key_padding_mask=modality_missing,
        )
        temporal = self.temporal_encoder(temporal, src_key_padding_mask=temporal_missing)
        modality_delta, _ = self.sensor_cross(
            modality_tokens, temporal, temporal,
            key_padding_mask=temporal_missing, need_weights=False,
        )
        temporal_delta, _ = self.temporal_cross(
            temporal, modality_tokens, modality_tokens,
            key_padding_mask=modality_missing, need_weights=False,
        )
        modality_tokens = self.sensor_cross_norm(modality_tokens + modality_delta)
        temporal = self.temporal_cross_norm(temporal + temporal_delta)

        modality_weights = modality_reliability / modality_reliability.sum(1, keepdim=True).clamp_min(1e-6)
        temporal_weights = temporal_reliability / temporal_reliability.sum(1, keepdim=True).clamp_min(1e-6)
        modality_repr = (modality_tokens * modality_weights.unsqueeze(-1)).sum(1)
        temporal_repr = (temporal * temporal_weights.unsqueeze(-1)).sum(1)
        branches = torch.stack((
            self.branch_norms[0](modality_repr),
            self.branch_norms[1](temporal_repr),
            self.branch_norms[2](global_token[:, 0]),
        ), dim=1)
        route_logits = self.router(torch.cat((
            branches.flatten(1), self.summarize(modality_reliability, temporal_reliability),
        ), dim=1))
        route_weights = torch.softmax(route_logits, dim=1)
        embedding = self.output_norm((branches * route_weights.unsqueeze(-1)).sum(1))
        if not return_aux:
            return embedding
        return embedding, {
            "route_weights": route_weights,
            "modality_reliability": modality_reliability,
            "temporal_reliability": temporal_reliability,
        }


class QualityFiLM(nn.Module):
    """Zero-start residual conditioning at a backbone's native fusion point."""

    def __init__(self, hidden, feature_dim):
        super().__init__()
        self.projection = nn.Linear(hidden, feature_dim * 2)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.feature_dim = feature_dim

    def forward(self, feature, quality_embedding):
        scale, shift = self.projection(quality_embedding).chunk(2, dim=-1)
        return feature * (1.0 + 0.5 * torch.tanh(scale)) + shift


def corrupt_modalities(modalities, probability=0.25, generator=None):
    """Apply per-sample modality dropout, relative noise, or flatline corruption."""
    corrupted = [value.clone() for value in modalities]
    batch, groups = len(modalities[0]), len(modalities)
    device = modalities[0].device
    selected = torch.rand((batch, groups), device=device, generator=generator) < probability
    all_selected = selected.all(1)
    if all_selected.any():
        rows = all_selected.nonzero(as_tuple=False).squeeze(1)
        choices = torch.randint(groups, (len(rows),), device=device, generator=generator)
        selected[rows, choices] = False
    kinds = torch.randint(3, (batch, groups), device=device, generator=generator)
    for group, value in enumerate(corrupted):
        rows = selected[:, group]
        dropout_rows = rows & (kinds[:, group] == 0)
        noise_rows = rows & (kinds[:, group] == 1)
        flatline_rows = rows & (kinds[:, group] == 2)
        if dropout_rows.any():
            value[dropout_rows] = 0
        if noise_rows.any():
            signal = value[noise_rows]
            scale = signal.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
            severity = torch.empty(
                (len(signal), 1, 1), device=device, dtype=value.dtype,
            ).uniform_(0.5, 1.5, generator=generator)
            noise = torch.randn(signal.shape, device=device, dtype=value.dtype, generator=generator)
            value[noise_rows] = signal + noise * scale * severity
        if flatline_rows.any():
            signal = value[flatline_rows]
            value[flatline_rows] = signal.mean(dim=(1, 2), keepdim=True)
    return tuple(corrupted), selected
