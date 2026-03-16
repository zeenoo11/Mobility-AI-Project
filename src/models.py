"""
Three models for EV battery energy consumption prediction:
  1. LSTMBaseline   - BMS features only, 2-layer LSTM
  2. CNNLSTMConcat  - Simple concatenation fusion
  3. CrossAttentionNet - Proposed cross-attention multimodal fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model 1: LSTM Baseline (battery features only)
# ---------------------------------------------------------------------------

class LSTMBaseline(nn.Module):
    """
    2-layer LSTM on BMS (battery) features only.
    Input: (batch, seq_len, 4)
    Output: (batch,) scalar energy Wh/km
    """

    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, drv, bms):
        # drv is ignored; only bms used
        _, (h_n, _) = self.lstm(bms)  # h_n: (num_layers, batch, hidden)
        feat = h_n[-1]                # last layer hidden: (batch, hidden)
        out = self.fc(feat).squeeze(-1)
        return out


# ---------------------------------------------------------------------------
# Model 2: CNN-LSTM Concatenation Fusion
# ---------------------------------------------------------------------------

class CNNLSTMConcat(nn.Module):
    """
    Driving branch: 1D-CNN → pooled feature
    Battery branch: LSTM → last hidden
    Concat → FC → output
    """

    def __init__(self, drv_dim=6, bms_dim=4, hidden_dim=64, cnn_channels=32, dropout=0.2):
        super().__init__()
        # Driving CNN branch
        self.drv_cnn = nn.Sequential(
            nn.Conv1d(drv_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )
        # Battery LSTM branch
        self.bms_lstm = nn.LSTM(
            input_size=bms_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        # Fusion FC
        self.fc = nn.Sequential(
            nn.Linear(cnn_channels + hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, drv, bms):
        # drv: (batch, seq, drv_dim) → (batch, drv_dim, seq) for Conv1d
        drv_t = drv.permute(0, 2, 1)
        cnn_out = self.drv_cnn(drv_t)           # (batch, channels, seq)
        drv_feat = cnn_out.mean(dim=-1)          # global avg pooling → (batch, channels)

        _, (h_n, _) = self.bms_lstm(bms)
        bms_feat = h_n[-1]                       # (batch, hidden)

        fused = torch.cat([drv_feat, bms_feat], dim=-1)
        out = self.fc(fused).squeeze(-1)
        return out


# ---------------------------------------------------------------------------
# Model 3: Cross-Attention Network (proposed model)
# ---------------------------------------------------------------------------

class MultiScaleCNN(nn.Module):
    """Multi-scale 1D-CNN with kernel sizes 3, 5, 7 in parallel."""

    def __init__(self, in_channels, out_channels=32):
        super().__init__()
        assert out_channels % 3 == 0 or out_channels >= 3
        ch = out_channels // 3
        remainder = out_channels - 3 * ch

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, ch + remainder, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch + remainder),
            nn.ReLU(),
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
        )
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_channels, ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
        )
        self.out_channels = ch + remainder + ch + ch  # = out_channels

    def forward(self, x):
        # x: (batch, in_channels, seq)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        return torch.cat([b3, b5, b7], dim=1)   # (batch, out_channels, seq)


class CrossAttentionNet(nn.Module):
    """
    Proposed multimodal fusion model with bidirectional cross-attention.

    Improvements over MVP v1:
    - Pre-LayerNorm (normalize before attention for training stability)
    - Residual connections (attn_output + projected_input)
    - 8 attention heads by default (matching CLAUDE.md design)

    Driving encoder: Multi-scale CNN → seq features
    Battery encoder: Bi-LSTM → seq features
    Cross-attention (bidirectional):
      - BMS queries driving (Driving→Battery attention)
      - Driving queries BMS (Battery→Driving attention)
    Residual + LayerNorm → FC → output
    """

    def __init__(self, drv_dim=6, bms_dim=4, cnn_out=48, lstm_hidden=64,
                 n_heads=8, dropout=0.2, attn_dim=64):
        super().__init__()
        # Driving encoder
        self.drv_encoder = MultiScaleCNN(in_channels=drv_dim, out_channels=cnn_out)
        drv_feat_dim = self.drv_encoder.out_channels

        # Battery encoder: Bi-LSTM (bidirectional → output_dim = 2 * hidden)
        self.bms_encoder = nn.LSTM(
            input_size=bms_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        bms_feat_dim = lstm_hidden * 2

        # Project both to same dimension for attention
        self.attn_dim = attn_dim
        self.drv_proj = nn.Linear(drv_feat_dim, attn_dim)
        self.bms_proj = nn.Linear(bms_feat_dim, attn_dim)

        # Pre-LayerNorm (applied before attention)
        self.pre_ln_drv = nn.LayerNorm(attn_dim)
        self.pre_ln_bms = nn.LayerNorm(attn_dim)

        # Cross-attention: BMS queries Driving  (Q=bms, K=V=drv)
        self.attn_bms_q_drv_kv = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        # Cross-attention: Driving queries BMS  (Q=drv, K=V=bms)
        self.attn_drv_q_bms_kv = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        # Post-residual LayerNorm
        self.post_ln_bms = nn.LayerNorm(attn_dim)
        self.post_ln_drv = nn.LayerNorm(attn_dim)

        self.dropout_layer = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(attn_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Store last attention weights for interpretability
        self._last_attn_bms_q = None
        self._last_attn_drv_q = None

    def forward(self, drv, bms, return_attn=False):
        """
        drv: (batch, seq, drv_dim)
        bms: (batch, seq, bms_dim)
        """
        # Driving: CNN → (batch, cnn_out, seq) → (batch, seq, cnn_out)
        drv_enc = self.drv_encoder(drv.permute(0, 2, 1)).permute(0, 2, 1)

        # Battery: Bi-LSTM → (batch, seq, 2*hidden)
        bms_enc, _ = self.bms_encoder(bms)

        # Project to common attention dimension
        drv_proj = self.drv_proj(drv_enc)   # (batch, seq, attn_dim)
        bms_proj = self.bms_proj(bms_enc)   # (batch, seq, attn_dim)

        # Pre-LayerNorm
        drv_normed = self.pre_ln_drv(drv_proj)
        bms_normed = self.pre_ln_bms(bms_proj)

        # Cross-attention 1: BMS queries Driving (with Pre-LN inputs)
        attn_bms_out, attn_w_bms = self.attn_bms_q_drv_kv(
            query=bms_normed, key=drv_normed, value=drv_normed
        )
        # Residual connection: attention output + original projected BMS
        bms_fused = self.post_ln_bms(bms_proj + self.dropout_layer(attn_bms_out))

        # Cross-attention 2: Driving queries BMS (with Pre-LN inputs)
        attn_drv_out, attn_w_drv = self.attn_drv_q_bms_kv(
            query=drv_normed, key=bms_normed, value=bms_normed
        )
        # Residual connection: attention output + original projected Driving
        drv_fused = self.post_ln_drv(drv_proj + self.dropout_layer(attn_drv_out))

        # Store attention weights
        self._last_attn_bms_q = attn_w_bms.detach()
        self._last_attn_drv_q = attn_w_drv.detach()

        # Global average pooling over sequence
        bms_pooled = bms_fused.mean(dim=1)   # (batch, attn_dim)
        drv_pooled = drv_fused.mean(dim=1)   # (batch, attn_dim)

        # Concat → FC
        fused = torch.cat([bms_pooled, drv_pooled], dim=-1)
        out = self.fc(fused).squeeze(-1)

        if return_attn:
            return out, attn_w_bms, attn_w_drv
        return out

    def get_attention_weights(self):
        """Return last stored attention weights."""
        return self._last_attn_bms_q, self._last_attn_drv_q


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(name, drv_dim=6, bms_dim=4, device='cpu', **kwargs):
    """Build a model by name and move to device.

    Extra kwargs are forwarded to CrossAttentionNet (e.g. cnn_out, lstm_hidden,
    n_heads, dropout, attn_dim).
    """
    if name == 'LSTMBaseline':
        model = LSTMBaseline(input_dim=bms_dim, hidden_dim=64)
    elif name == 'CNNLSTMConcat':
        model = CNNLSTMConcat(drv_dim=drv_dim, bms_dim=bms_dim, hidden_dim=64, cnn_channels=32)
    elif name == 'CrossAttentionNet':
        ca_defaults = dict(cnn_out=48, lstm_hidden=64, n_heads=8, dropout=0.2, attn_dim=64)
        ca_defaults.update(kwargs)
        model = CrossAttentionNet(drv_dim=drv_dim, bms_dim=bms_dim, **ca_defaults)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model.to(device)
