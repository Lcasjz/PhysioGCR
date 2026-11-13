"""Five literature-model adapters for the same Husformer four-modal input."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.wesad.literature_models import HusformerCrossLayer, PersonalizedAutoencoderHyperNet
from benchmarks.wesad.quality_module import QualityAwareCrossGranularRouter, QualityFiLM


MODALITY_NAMES = ("resp", "bvp", "ecg", "eda")
MODALITY_DIMS = (14, 4, 14, 4)
MODALITY_LENGTHS = (50, 16, 50, 1)


class QualityIntegratedModel(nn.Module):
    def __init__(self, feature_dim, quality_module=False, quality_hidden=128):
        super().__init__()
        self.quality_router = (
            QualityAwareCrossGranularRouter(hidden=quality_hidden) if quality_module else None
        )
        self.quality_film = (
            QualityFiLM(quality_hidden, feature_dim) if quality_module else None
        )
        self.last_quality_aux = None

    def condition_features(self, feature, modalities, quality_modalities=None, return_aux=False):
        if self.quality_router is None:
            self.last_quality_aux = None
            return (feature, None) if return_aux else feature
        routed, auxiliary = self.quality_router(
            modalities, quality_modalities=quality_modalities, return_aux=True,
        )
        feature = self.quality_film(feature, routed)
        self.last_quality_aux = auxiliary
        return (feature, auxiliary) if return_aux else feature


class TemporalModalityEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 48, 3, padding=1), nn.GroupNorm(4, 48), nn.GELU(),
            nn.Conv1d(48, 64, 3, padding=1), nn.GroupNorm(4, 64), nn.GELU(),
            nn.Conv1d(64, embedding_dim, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, value):
        return self.net(value.transpose(1, 2)).squeeze(-1)


class FourModalEncoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.encoders = nn.ModuleList(
            TemporalModalityEncoder(dim, embedding_dim) for dim in MODALITY_DIMS
        )

    def embeddings(self, modalities):
        return torch.stack([
            encoder(value) for encoder, value in zip(self.encoders, modalities)
        ])

    def forward(self, modalities):
        return self.embeddings(modalities).transpose(0, 1).flatten(1)


class COCOAHusformerInput(QualityIntegratedModel):
    def __init__(self, classes=3, embedding_dim=64, quality_module=False, quality_hidden=128):
        super().__init__(embedding_dim * 4, quality_module, quality_hidden)
        self.encoder = FourModalEncoder(embedding_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim * 4), nn.Linear(embedding_dim * 4, 128),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(128, classes),
        )

    def ssl_loss(self, modalities, temperature=0.1, negative_weight=3.9e-3):
        z = F.normalize(self.encoder.embeddings(modalities), dim=-1)
        modal_count, batch, _ = z.shape
        cross_modal = torch.einsum("mbd,nbd->bmn", z, z)
        modal_mask = ~torch.eye(modal_count, dtype=torch.bool, device=z.device)
        positive = torch.exp((1.0 - cross_modal[:, modal_mask]) / temperature).mean()
        if batch > 1:
            cross_sample = torch.einsum("mbd,mcd->mbc", z, z)
            sample_mask = ~torch.eye(batch, dtype=torch.bool, device=z.device)
            negative = torch.exp(cross_sample[:, sample_mask] / temperature).mean()
        else:
            negative = z.new_zeros(())
        return positive + negative_weight * negative

    def forward(self, modalities, subject_ids=None, quality_modalities=None, return_quality_aux=False):
        conditioned = self.condition_features(
            self.encoder(modalities), modalities, quality_modalities, return_aux=return_quality_aux,
        )
        feature, auxiliary = conditioned if return_quality_aux else (conditioned, None)
        logits = self.classifier(feature)
        return (logits, auxiliary) if return_quality_aux else logits


class SemiPFLHusformerInput(QualityIntegratedModel):
    def __init__(self, clients, classes=3, quality_module=False, quality_hidden=128):
        super().__init__(128, quality_module, quality_hidden)
        self.clients = clients
        self.encoder = FourModalEncoder(64)
        self.projector = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(),
        )
        self.hypernet = PersonalizedAutoencoderHyperNet(clients)
        self.client_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, classes))
            for _ in range(clients)
        ])

    def features(self, modalities, quality_modalities=None):
        feature = self.projector(self.encoder(modalities))
        return self.condition_features(feature, modalities, quality_modalities)

    def client_forward(self, feature, client_index):
        client = torch.tensor(client_index, device=feature.device, dtype=torch.long)
        latent, reconstruction = self.hypernet(feature, client)
        return self.client_heads[client_index](latent), reconstruction

    def reconstruction_loss(self, modalities, client_ids, quality_modalities=None):
        feature = self.features(modalities, quality_modalities)
        total, count = feature.new_zeros(()), 0
        for client in client_ids.unique(sorted=True):
            mask = client_ids == client
            _, reconstruction = self.client_forward(feature[mask], int(client.item()))
            total = total + F.mse_loss(reconstruction, feature[mask]) * int(mask.sum())
            count += int(mask.sum())
        return total / max(count, 1)

    def forward(self, modalities, subject_ids=None, quality_modalities=None, return_quality_aux=False):
        feature = self.features(modalities, quality_modalities)
        if self.training and subject_ids is not None:
            logits = feature.new_empty((len(feature), 3))
            for client in subject_ids.unique(sorted=True):
                mask = subject_ids == client
                logits[mask], _ = self.client_forward(feature[mask], int(client.item()))
            return (logits, self.last_quality_aux) if return_quality_aux else logits
        logits = torch.stack([
            self.client_forward(feature, client)[0] for client in range(self.clients)
        ]).mean(0)
        return (logits, self.last_quality_aux) if return_quality_aux else logits


class StressCalibrationHusformerInput(QualityIntegratedModel):
    def __init__(self, classes=3, quality_module=False, quality_hidden=128):
        super().__init__(256, quality_module, quality_hidden)
        self.encoder = FourModalEncoder(64)
        self.classifier = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(),
            nn.Dropout(0.25), nn.Linear(128, classes),
        )
        self.decoders = nn.ModuleList([
            nn.Linear(256, length * dim)
            for length, dim in zip(MODALITY_LENGTHS, MODALITY_DIMS)
        ])

    def masked_reconstruction_loss(self, modalities, mask_ratio=0.30):
        masked, masks = [], []
        for value in modalities:
            mask = torch.rand((len(value), value.size(1), 1), device=value.device) < mask_ratio
            masked.append(value.masked_fill(mask, 0.0))
            masks.append(mask.expand_as(value))
        feature = self.encoder(tuple(masked))
        total, count = feature.new_zeros(()), feature.new_zeros(())
        for decoder, value, mask in zip(self.decoders, modalities, masks):
            reconstruction = decoder(feature).view_as(value)
            total = total + ((reconstruction - value).square() * mask).sum()
            count = count + mask.sum()
        return total / count.clamp_min(1)

    def forward(self, modalities, subject_ids=None, quality_modalities=None, return_quality_aux=False):
        feature = self.encoder(modalities)
        conditioned = self.condition_features(
            feature, modalities, quality_modalities, return_aux=return_quality_aux,
        )
        feature, auxiliary = conditioned if return_quality_aux else (conditioned, None)
        logits = self.classifier(feature)
        return (logits, auxiliary) if return_quality_aux else logits


def modality_statistics(value):
    delta = value[:, 1:] - value[:, :-1]
    if value.size(1) == 1:
        delta_abs = torch.zeros_like(value[:, 0])
        delta_std = torch.zeros_like(value[:, 0])
    else:
        delta_abs = delta.abs().mean(1)
        delta_std = delta.std(1, unbiased=False)
    features = (
        value.mean(1), value.std(1, unbiased=False), value.amin(1), value.amax(1),
        value.square().mean(1).sqrt(), delta_abs, delta_std,
        value[:, -1] - value[:, 0], value[:, 0], value[:, -1],
    )
    return torch.cat(features, dim=1)


class FGDualNetHusformerInput(QualityIntegratedModel):
    def __init__(self, classes=3, quality_module=False, quality_hidden=128):
        input_dim = sum(MODALITY_DIMS) * 10
        super().__init__(input_dim, quality_module, quality_hidden)
        self.feature_gate = nn.Linear(input_dim, input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, classes),
        )

    def forward(self, modalities, subject_ids=None, quality_modalities=None,
                return_gate=False, return_quality_aux=False):
        features = torch.cat([modality_statistics(value) for value in modalities], dim=1)
        conditioned = self.condition_features(
            features, modalities, quality_modalities, return_aux=return_quality_aux,
        )
        features, auxiliary = conditioned if return_quality_aux else (conditioned, None)
        gate = torch.sigmoid(self.feature_gate(features))
        logits = self.classifier(features * gate)
        if return_quality_aux:
            return logits, auxiliary
        return (logits, gate) if return_gate else logits


def sinusoidal(length, width):
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
    output = torch.zeros(length, width)
    output[:, 0::2] = torch.sin(positions * div)
    output[:, 1::2] = torch.cos(positions * div)
    return output.unsqueeze(1)


class HusformerPublicInput(QualityIntegratedModel):
    def __init__(self, classes=3, width=30, heads=3, dropout=0.05, quality_module=False,
                 quality_hidden=128):
        super().__init__(width, quality_module, quality_hidden)
        self.width = width
        self.projections = nn.ModuleList([
            nn.Conv1d(dim, width, 1, bias=False) for dim in MODALITY_DIMS
        ])
        self.cross_streams = nn.ModuleList([
            nn.ModuleList([HusformerCrossLayer(width, heads, dropout) for _ in range(3)])
            for _ in MODALITY_DIMS
        ])
        self.fusion = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=width, nhead=heads, dim_feedforward=width * 4,
                dropout=dropout, batch_first=False, norm_first=True, activation="relu",
            )
            for _ in range(5)
        ])
        self.fusion_norm = nn.LayerNorm(width)
        self.final_pool = nn.Conv1d(sum(MODALITY_LENGTHS), 1, 1, bias=False)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(width, classes))
        for index, length in enumerate(MODALITY_LENGTHS):
            self.register_buffer(f"position_{index}", sinusoidal(length, width), persistent=False)

    def forward(self, modalities, subject_ids=None, quality_modalities=None, return_quality_aux=False):
        streams = []
        for index, (projection, value) in enumerate(zip(self.projections, modalities)):
            stream = projection(value.transpose(1, 2)).permute(2, 0, 1)
            stream = stream * math.sqrt(self.width) + getattr(self, f"position_{index}")
            streams.append(stream)
        context = torch.cat(streams, dim=0)
        encoded = []
        for stream, layers in zip(streams, self.cross_streams):
            for layer in layers:
                stream = layer(stream, context)
            encoded.append(stream)
        fused = torch.cat(encoded, dim=0)
        for layer in self.fusion:
            fused = layer(fused)
        pooled = self.final_pool(self.fusion_norm(fused).permute(1, 0, 2)).squeeze(1)
        conditioned = self.condition_features(
            pooled, modalities, quality_modalities, return_aux=return_quality_aux,
        )
        pooled, auxiliary = conditioned if return_quality_aux else (conditioned, None)
        logits = self.classifier(pooled)
        return (logits, auxiliary) if return_quality_aux else logits


HUSFORMER_INPUT_MODELS = {
    "cocoa": COCOAHusformerInput,
    "semipfl": SemiPFLHusformerInput,
    "stress_calibration_2026": StressCalibrationHusformerInput,
    "fg_dualnet": FGDualNetHusformerInput,
    "husformer": HusformerPublicInput,
}
