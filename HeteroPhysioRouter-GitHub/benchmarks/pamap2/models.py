import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception1D(nn.Module):
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        self.p1 = nn.Sequential(nn.Conv1d(c1[0], c1[1], 1), nn.ReLU())
        self.p2 = nn.Sequential(nn.Conv1d(c1[0], c2[0], 1), nn.ReLU(), nn.Conv1d(c2[0], c2[1], 3, padding=1), nn.ReLU())
        self.p3 = nn.Sequential(nn.Conv1d(c1[0], c3[0], 1), nn.ReLU(), nn.Conv1d(c3[0], c3[1], 3, padding=1), nn.ReLU())
        self.p4_conv = nn.Sequential(nn.Conv1d(c1[0], c4, 1), nn.ReLU())

    def forward(self, x):
        p4 = self.p4_conv(F.max_pool1d(x, 3, stride=1, padding=1))
        return torch.cat((self.p1(x), self.p2(x), self.p3(x), p4), dim=1)


class InnoHAR(nn.Module):
    """InnoHAR Inception-GRU architecture."""
    def __init__(self, d=52, classes=12):
        super().__init__()
        self.features = nn.Sequential(
            Inception1D((d, 64), (32, 64), (32, 64), 64),
            Inception1D((256, 56), (32, 64), (32, 64), 64),
            Inception1D((248, 32), (32, 56), (32, 56), 32),
            nn.MaxPool1d(2, ceil_mode=True),
            Inception1D((176, 32), (16, 32), (16, 32), 24),
            nn.MaxPool1d(2, ceil_mode=True),
        )
        self.gru1 = nn.GRU(120, 120, batch_first=True)
        self.gru2 = nn.GRU(120, 40, batch_first=True)
        self.fc = nn.Linear(40, classes)

    def forward(self, x):
        z = self.features(x.transpose(1, 2)).transpose(1, 2)
        z, _ = self.gru1(z)
        z, _ = self.gru2(z)
        return self.fc(z[:, -1])


class SameConvBlock(nn.Module):
    def __init__(self, cin, cout, batch_norm=False):
        super().__init__()
        layers = [nn.Conv2d(cin, cout, (5, 1), padding=(2, 0)), nn.ReLU()]
        if batch_norm:
            layers.append(nn.BatchNorm2d(cout))
        layers += [nn.Conv2d(cout, cout, (5, 1), stride=(2, 1), padding=(2, 0)), nn.ReLU()]
        if batch_norm:
            layers.append(nn.BatchNorm2d(cout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MCNN(nn.Module):
    """TinyHAR repository MCNN, adapted to the canonical 24-sample window."""
    def __init__(self, d=52, classes=12):
        super().__init__()
        self.blocks = nn.Sequential(SameConvBlock(1, 64), SameConvBlock(64, 64))
        self.dropout = nn.Dropout(0.25)
        self.sensor_fusion = nn.Linear(64 * d, 128)
        self.temporal_fusion = nn.Linear(6 * 128, 128)
        self.fc = nn.Linear(128, classes)

    def encode_tokens(self, x):
        z = self.blocks(x[:, None]).permute(0, 2, 1, 3).flatten(2)
        return F.relu(self.sensor_fusion(self.dropout(z)))

    def encode(self, x):
        z = self.encode_tokens(x).flatten(1)
        return F.relu(self.temporal_fusion(z)).unsqueeze(1)

    def forward(self, x):
        return self.fc(self.encode(x).squeeze(1))


class ChannelInteraction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        beta = torch.softmax(torch.bmm(self.q(x), self.k(x).transpose(1, 2)), dim=1)
        return x + self.gamma * torch.bmm(beta.transpose(1, 2), self.v(x))


class FilterAggregation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.value = nn.Linear(dim, dim)
        self.weight = nn.Linear(dim, dim)

    def forward(self, x):
        weights = torch.softmax(torch.tanh(self.weight(x)), dim=1)
        return (F.relu(self.value(x)) * weights).sum(1)


class TinyHAR(nn.Module):
    """TinyHAR default attention/filter/GRU/FC configuration."""
    def __init__(self, d=52, classes=12):
        super().__init__()
        layers = []
        for i in range(4):
            layers.append(nn.Sequential(
                nn.Conv2d(1 if i == 0 else 20, 20, (5, 1), stride=(2 if i % 2 else 1, 1), padding=(2, 0)),
                nn.ReLU(), nn.BatchNorm2d(20)))
        self.layers = nn.ModuleList(layers)
        self.channel_interaction = ChannelInteraction(20)
        self.channel_fusion = FilterAggregation(20)
        self.dropout = nn.Dropout(0.1)
        self.gru = nn.GRU(20, 20, batch_first=True)
        self.temporal_fusion = nn.Linear(6 * 20, 40)
        self.fc = nn.Linear(40, classes)

    def encode_tokens(self, x):
        z = x[:, None]
        for layer in self.layers:
            z = layer(z)
        z = z.permute(0, 3, 2, 1)
        z = torch.stack([self.channel_interaction(z[:, :, t]) for t in range(z.size(2))], dim=1)
        z = self.dropout(z)
        z = torch.stack([self.channel_fusion(z[:, t]) for t in range(z.size(1))], dim=1)
        z, _ = self.gru(F.relu(z))
        return z

    def encode(self, x):
        z = self.encode_tokens(x)
        return F.relu(self.temporal_fusion(z.flatten(1))).unsqueeze(1)

    def forward(self, x):
        return self.fc(self.encode(x).squeeze(1))


class SAConvBlock(nn.Module):
    """SA-HAR ConvBlock, matching the authors' implementation."""
    def __init__(self, width=5, filters=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, filters, (width, 1), padding='same')
        self.conv2 = nn.Conv2d(filters, 1, (width, 1), padding='same')
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(filters)
        self.norm2 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.norm1(self.relu(self.conv1(x)))
        return self.norm2(self.relu(self.conv2(x)))


class SAAttentionLayer(nn.Module):
    """The custom multi-head attention used by the SA-HAR authors."""
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=True)
        self.value_projection = nn.Linear(d_model, d_model, bias=True)
        self.out_projection = nn.Linear(d_model, d_model, bias=True)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        batch, length, _ = queries.shape
        heads = self.n_heads
        queries = self.query_projection(queries).view(batch, length, heads, -1)
        keys = self.key_projection(keys).view(batch, length, heads, -1)
        values = self.value_projection(values).view(batch, length, heads, -1)
        scores = torch.einsum('blhe,bshe->bhls', queries, keys)
        attention = torch.softmax(scores / (queries.shape[-1] ** 0.5), dim=-1)
        output = torch.einsum('bhls,bshd->blhd', attention, values).contiguous()
        return self.out_projection(output.view(batch, length, -1)), attention


class SAEncoderLayer(nn.Module):
    """Post-norm SA-HAR encoder layer, including the authors' dropout sites."""
    def __init__(self, d_model=128, n_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.attention = SAAttentionLayer(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn1 = nn.Linear(d_model, d_ff)
        self.ffn2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        attention, _ = self.attention(x, x, x)
        x = self.layernorm1(x + self.dropout1(attention))
        feed_forward = self.ffn2(F.relu(self.ffn1(x)))
        return self.layernorm2(x + self.dropout2(feed_forward))


class SensorAttentionHAR(nn.Module):
    """SA-HAR author architecture adapted only to the canonical input shape."""
    def __init__(self, d=52, classes=12, filters=128):
        super().__init__()
        self.first_conv = SAConvBlock(filters=filters)
        self.sensor_ln = nn.LayerNorm(d)
        self.sensor_conv1 = nn.Conv2d(1, filters, 3, dilation=2, padding='same')
        self.sensor_conv2 = nn.Conv2d(filters, 1, 1, padding='same')
        self.project = nn.Conv1d(d, filters, 1)
        self.encoder1 = SAEncoderLayer(filters, 4, filters * 4, dropout=0.1)
        self.encoder2 = SAEncoderLayer(filters, 4, filters * 4, dropout=0.1)
        self.context_w = nn.Linear(filters, filters)
        self.context_u = nn.Linear(filters, 1, bias=False)
        self.fc1 = nn.Linear(filters, 4 * classes)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(4 * classes, classes)

    def encode_tokens(self, x):
        x = self.first_conv(x[:, None]).squeeze(1)
        x = self.sensor_ln(x)
        weights = torch.softmax(self.sensor_conv2(F.relu(self.sensor_conv1(x[:, None]))), dim=3).squeeze(1)
        x = x * weights
        x = F.relu(self.project(x.transpose(1, 2)).transpose(1, 2))
        x = self.encoder1(x)
        x = self.encoder2(x)
        return x

    def encode(self, x):
        x = self.encode_tokens(x)
        weights = torch.softmax(self.context_u(torch.tanh(self.context_w(x))), dim=1)
        return (weights * x).sum(1, keepdim=True)

    def forward(self, x):
        x = self.encode(x).squeeze(1)
        return self.fc(self.drop(F.relu(self.fc1(x))))
