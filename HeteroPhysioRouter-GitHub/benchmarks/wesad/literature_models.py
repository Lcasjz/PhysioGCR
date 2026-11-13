"""Canonical-input adaptations of WESAD literature models.

The implementations in this file preserve each method's distinguishing
mechanism while accepting the benchmark tensor (batch, 240, 14). They are
adaptations, not byte-for-byte reproductions of the downloaded repositories.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Canonical channel groups: chest ACC, ECG, EDA, EMG, Resp, Temp, followed by
# wrist ACC, BVP, EDA, and Temp. These groups are also the graph sensor nodes.
SENSOR_GROUPS = (
    (0, 1, 2), (3,), (4,), (5,), (6,), (7,),
    (8, 9, 10), (11,), (12,), (13,),
)
SENSOR_NAMES = (
    "chest_acc", "chest_ecg", "chest_eda", "chest_emg", "chest_resp",
    "chest_temp", "wrist_acc", "wrist_bvp", "wrist_eda", "wrist_temp",
)


class ModalityEncoder(nn.Module):
    """PyTorch counterpart of COCOA's modality-specific temporal encoder."""

    def __init__(self, channels, embedding_dim=64, filters=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, filters * 2, 10, stride=4, padding=4),
            nn.GroupNorm(4, filters * 2), nn.PReLU(filters * 2),
            nn.Conv1d(filters * 2, filters, 8, stride=2, padding=3),
            nn.GroupNorm(4, filters), nn.PReLU(filters),
            nn.Conv1d(filters, embedding_dim, 4, padding=1),
            nn.PReLU(embedding_dim), nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class COCOAAdaptation(nn.Module):
    def __init__(self, classes=3, embedding_dim=64):
        super().__init__()
        self.encoders = nn.ModuleList(
            ModalityEncoder(len(group), embedding_dim) for group in SENSOR_GROUPS
        )
        fused_dim = embedding_dim * len(SENSOR_GROUPS)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim), nn.Linear(fused_dim, 128), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(128, classes),
        )

    def embeddings(self, x):
        # Output is (modalities, batch, embedding), matching the COCOA loss.
        return torch.stack([
            encoder(x[:, :, group].transpose(1, 2))
            for encoder, group in zip(self.encoders, SENSOR_GROUPS)
        ])

    def ssl_loss(self, x, temperature=0.1, negative_weight=3.9e-3):
        z = F.normalize(self.embeddings(x), dim=-1)
        modalities, batch, _ = z.shape

        # Same-window embeddings from different modalities are attracted.
        cross_modal = torch.einsum("mbd,nbd->bmn", z, z)
        modal_mask = ~torch.eye(modalities, dtype=torch.bool, device=x.device)
        positive = torch.exp((1.0 - cross_modal[:, modal_mask]) / temperature).mean()

        # Different windows within each modality are repelled, as in cocoa2.
        if batch > 1:
            cross_sample = torch.einsum("mbd,mcd->mbc", z, z)
            sample_mask = ~torch.eye(batch, dtype=torch.bool, device=x.device)
            negative = torch.exp(cross_sample[:, sample_mask] / temperature).mean()
        else:
            negative = z.new_zeros(())
        return positive + negative_weight * negative

    def forward(self, x, subject_ids=None):
        z = self.embeddings(x).transpose(0, 1).flatten(1)
        return self.classifier(z)


class TemporalProjector(nn.Module):
    def __init__(self, channels=14, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 64, 9, stride=4, padding=4), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 96, 7, stride=3, padding=3), nn.BatchNorm1d(96), nn.GELU(),
            nn.Conv1d(96, output_dim, 5, stride=2, padding=2), nn.BatchNorm1d(output_dim),
            nn.GELU(), nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).squeeze(-1)


class PersonalizedAutoencoderHyperNet(nn.Module):
    """Generates all autoencoder parameters from a train-client embedding."""

    SHAPES = ((96, 128), (96,), (64, 96), (64,),
              (96, 64), (96,), (128, 96), (128,))

    def __init__(self, clients, client_embedding=16, hidden=32):
        super().__init__()
        self.client_embedding = nn.Embedding(clients, client_embedding)
        self.generator = nn.Sequential(
            nn.Linear(client_embedding, hidden), nn.ReLU(),
            nn.Linear(hidden, sum(math.prod(shape) for shape in self.SHAPES)),
        )

    def parameters_for(self, client):
        raw = self.generator(self.client_embedding(client.reshape(1))).squeeze(0)
        output, offset = [], 0
        for shape in self.SHAPES:
            count = math.prod(shape)
            output.append(raw[offset:offset + count].view(shape))
            offset += count
        return output

    def forward(self, feature, client):
        w1, b1, w2, b2, w3, b3, w4, b4 = self.parameters_for(client)
        hidden = F.relu(F.linear(feature, w1, b1))
        latent = F.linear(hidden, w2, b2)
        recon = F.linear(F.relu(F.linear(latent, w3, b3)), w4, b4)
        return latent, recon


class SemiPFLAdaptation(nn.Module):
    """Train-client hypernetwork with leakage-safe unseen-client ensembling."""

    def __init__(self, clients, classes=3):
        super().__init__()
        self.clients = clients
        self.projector = TemporalProjector(output_dim=128)
        self.hypernet = PersonalizedAutoencoderHyperNet(clients)
        self.client_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, classes))
            for _ in range(clients)
        ])

    def client_forward(self, feature, client_index):
        client = torch.tensor(client_index, device=feature.device, dtype=torch.long)
        latent, recon = self.hypernet(feature, client)
        return self.client_heads[client_index](latent), recon

    def reconstruction_loss(self, x, client_ids):
        feature = self.projector(x)
        total = feature.new_zeros(())
        count = 0
        for client in client_ids.unique(sorted=True):
            mask = client_ids == client
            _, recon = self.client_forward(feature[mask], int(client.item()))
            total = total + F.mse_loss(recon, feature[mask]) * int(mask.sum())
            count += int(mask.sum())
        return total / max(count, 1)

    def forward(self, x, subject_ids=None):
        feature = self.projector(x)
        if self.training and subject_ids is not None:
            logits = feature.new_empty((len(feature), 3))
            for client in subject_ids.unique(sorted=True):
                mask = subject_ids == client
                logits[mask], _ = self.client_forward(feature[mask], int(client.item()))
            return logits

        # Validation and test subjects are unseen clients. No labels or
        # subject-specific updates are used; train-client logits are averaged.
        return torch.stack([
            self.client_forward(feature, client)[0] for client in range(self.clients)
        ]).mean(0)


def sensor_node_signals(x):
    signals = []
    for group in SENSOR_GROUPS:
        values = x[:, :, group]
        if len(group) > 1:
            values = torch.sqrt(values.square().mean(-1) + 1e-6)
        else:
            values = values.squeeze(-1)
        signals.append(values)
    return torch.stack(signals, dim=1)


def node_statistics(x):
    signals = sensor_node_signals(x)
    delta = signals[:, :, 1:] - signals[:, :, :-1]
    return torch.stack((
        signals.mean(-1), signals.std(-1), signals.amin(-1), signals.amax(-1),
        signals.square().mean(-1).sqrt(), delta.abs().mean(-1),
        delta.std(-1), signals[:, :, -1] - signals[:, :, 0],
    ), dim=-1)


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.register_buffer("weight_mask", torch.ones_like(self.linear.weight))

    def forward(self, x, adjacency):
        weight = self.linear.weight * self.weight_mask
        return F.linear(torch.matmul(adjacency, x), weight, self.linear.bias)

    @torch.no_grad()
    def prune(self, amount):
        live = self.weight_mask.bool()
        count = max(1, int(live.sum().item() * amount))
        values = self.linear.weight.abs().masked_fill(~live, float("inf")).flatten()
        indices = values.topk(count, largest=False).indices
        self.weight_mask.flatten()[indices] = 0.0


def physiological_adjacency():
    adjacency = torch.eye(len(SENSOR_GROUPS))
    # Dense within-device connectivity, plus corresponding cross-device links.
    adjacency[:6, :6] = 1.0
    adjacency[6:, 6:] = 1.0
    for left, right in ((0, 6), (1, 7), (2, 8), (5, 9)):
        adjacency[left, right] = adjacency[right, left] = 1.0
    degree = adjacency.sum(1).clamp_min(1.0).pow(-0.5)
    return degree[:, None] * adjacency * degree[None, :]


class ExplainableGCNAdaptation(nn.Module):
    def __init__(self, classes=3):
        super().__init__()
        self.gcn1 = GraphConvolution(8, 64)
        self.gcn2 = GraphConvolution(64, 64)
        self.attention = nn.Linear(64, 1)
        self.classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(64, classes))
        self.register_buffer("adjacency", physiological_adjacency())

    def forward(self, x, subject_ids=None, return_attention=False):
        nodes = node_statistics(x)
        hidden = F.relu(self.gcn1(nodes, self.adjacency))
        hidden = F.dropout(hidden, 0.25, self.training)
        hidden = F.relu(self.gcn2(hidden, self.adjacency))
        attention = self.attention(hidden).softmax(1)
        logits = self.classifier((hidden * attention).sum(1))
        return (logits, attention.squeeze(-1)) if return_attention else logits

    def apply_pruning(self, amount=0.3):
        self.gcn1.prune(amount)
        self.gcn2.prune(amount)


class CalibrationEncoder(nn.Module):
    def __init__(self, channels=14, width=64, embedding_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, width, 7, stride=2, padding=3), nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width * 2, 5, stride=2, padding=2), nn.BatchNorm1d(width * 2), nn.GELU(),
            nn.Conv1d(width * 2, width * 2, 5, stride=2, padding=2), nn.BatchNorm1d(width * 2), nn.GELU(),
            nn.Conv1d(width * 2, embedding_dim, 3, stride=2, padding=1),
            nn.BatchNorm1d(embedding_dim), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2))


class StressCalibrationAdaptation(nn.Module):
    def __init__(self, classes=3):
        super().__init__()
        self.encoder = CalibrationEncoder()
        self.decoder = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1), nn.GELU(), nn.Conv1d(128, 14, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.25), nn.Linear(64, classes),
        )

    def masked_reconstruction_loss(self, x, mask_ratio=0.30):
        mask = torch.rand((len(x), x.size(1), 1), device=x.device) < mask_ratio
        masked = x.masked_fill(mask, 0.0)
        reconstruction = self.decoder(self.encoder(masked))
        reconstruction = F.interpolate(reconstruction, size=x.size(1), mode="linear", align_corners=False)
        target = x.transpose(1, 2)
        expanded_mask = mask.transpose(1, 2).expand_as(target)
        return (reconstruction.sub(target).square() * expanded_mask).sum() / expanded_mask.sum().clamp_min(1)

    def forward(self, x, subject_ids=None):
        return self.classifier(self.encoder(x).mean(-1))


def statistical_features(x):
    values = x.transpose(1, 2)
    delta = values[:, :, 1:] - values[:, :, :-1]
    features = (
        values.mean(-1), values.std(-1), values.amin(-1), values.amax(-1),
        values.square().mean(-1).sqrt(), delta.abs().mean(-1), delta.std(-1),
        values[:, :, -1] - values[:, :, 0], values[:, :, 0], values[:, :, -1],
    )
    return torch.cat(features, dim=1)


class FGDualNetAdaptation(nn.Module):
    """Multiclass canonical adapter for the author's feature-gated MLP."""

    def __init__(self, classes=3, channels=14):
        super().__init__()
        input_dim = channels * 10
        self.feature_gate = nn.Linear(input_dim, input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, classes),
        )

    def forward(self, x, subject_ids=None, return_gate=False):
        features = statistical_features(x)
        gate = torch.sigmoid(self.feature_gate(features))
        logits = self.classifier(features * gate)
        return (logits, gate) if return_gate else logits


class HusformerCrossLayer(nn.Module):
    """Pre-norm cross-modal Transformer block used by the Husformer adapter."""

    def __init__(self, width=30, heads=3, dropout=0.1):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_out_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width * 4, width), nn.Dropout(dropout),
        )

    def forward(self, query, context):
        update, _ = self.attention(
            self.query_norm(query), self.context_norm(context),
            self.context_norm(context), need_weights=False,
        )
        query = self.attention_out_norm(query + self.attention_dropout(update))
        return query + self.ffn(self.ffn_norm(query))


class HusformerAdaptation(nn.Module):
    """Canonical-input adaptation of the public Husformer architecture.

    The source implementation accepts a variable number of modality streams.
    We use four streams that cover every canonical channel, patch each 60-second
    window into 16-sample tokens, then retain the source cross-modal and final
    self-attention stages. This keeps the benchmark input identical across
    models while avoiding a high-rate/source-specific preprocessing pipeline.
    """

    STREAM_GROUPS = (
        (3, 4, 6),       # chest ECG, EDA, respiration
        (0, 1, 2, 5, 7),  # chest acceleration, EMG, temperature
        (8, 9, 10),      # wrist acceleration
        (11, 12, 13),    # wrist BVP, EDA, temperature
    )

    def __init__(self, channels=14, length=240, classes=3, width=30, heads=3, patch=16):
        super().__init__()
        if channels != 14 or length <= 0:
            raise ValueError(f"Canonical Husformer expects 14 channels and a positive length, got {(length, channels)}")
        if length % patch:
            raise ValueError("Husformer patch size must divide the canonical window length")
        flattened = [channel for group in self.STREAM_GROUPS for channel in group]
        if sorted(flattened) != list(range(channels)):
            raise ValueError("Husformer stream mapping must cover each canonical channel exactly once")
        self.width = width
        self.length = length
        self.patch = patch
        self.tokens = length // patch
        self.projections = nn.ModuleList([
            nn.Conv1d(len(group), width, kernel_size=patch, stride=patch, bias=False)
            for group in self.STREAM_GROUPS
        ])
        self.cross_streams = nn.ModuleList([
            nn.ModuleList([HusformerCrossLayer(width, heads) for _ in range(3)])
            for _ in self.STREAM_GROUPS
        ])
        self.fusion = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=width, nhead=heads, dim_feedforward=width * 4,
                dropout=0.1, batch_first=False, norm_first=True, activation="relu",
            )
            for _ in range(5)
        ])
        self.fusion_norm = nn.LayerNorm(width)
        self.final_conv = nn.Conv1d(len(self.STREAM_GROUPS) * self.tokens, 1, 1, bias=False)
        self.classifier = nn.Linear(width, classes)
        positions = torch.arange(self.tokens, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
        positional = torch.zeros(self.tokens, width)
        positional[:, 0::2] = torch.sin(positions * div_term)
        positional[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("positional_embedding", positional.unsqueeze(1), persistent=False)

    def forward(self, x, subject_ids=None):
        if x.ndim != 3 or x.size(1) != self.length or x.size(2) != 14:
            raise ValueError(f"Husformer expected (batch, {self.length}, 14), got {tuple(x.shape)}")
        streams = [
            projection(x[:, :, group].transpose(1, 2)).permute(2, 0, 1)
            for projection, group in zip(self.projections, self.STREAM_GROUPS)
        ]
        streams = [stream * math.sqrt(self.width) + self.positional_embedding for stream in streams]
        all_modalities = torch.cat(streams, dim=0)
        encoded = []
        for stream, layers in zip(streams, self.cross_streams):
            for layer in layers:
                stream = layer(stream, all_modalities)
            encoded.append(stream)
        fused = torch.cat(encoded, dim=0)
        for layer in self.fusion:
            fused = layer(fused)
        fused = self.fusion_norm(fused).permute(1, 0, 2)
        pooled = self.final_conv(fused).squeeze(1)
        return self.classifier(pooled)


LITERATURE_MODELS = {
    "cocoa": COCOAAdaptation,
    "semipfl": SemiPFLAdaptation,
    "explainable_gcn": ExplainableGCNAdaptation,
    "stress_calibration_2026": StressCalibrationAdaptation,
    "fg_dualnet": FGDualNetAdaptation,
    "husformer": HusformerAdaptation,
}
