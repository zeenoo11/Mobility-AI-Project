"""
Graph WaveNet for Spatial-Temporal EV Charging Demand Prediction

Architecture:
  - Adaptive adjacency matrix (learned) + predefined adjacency
  - Stacked Spatial-Temporal Conv blocks:
      - Dilated Causal Conv (temporal axis)
      - Graph Conv with mixed adjacency (spatial axis)
      - Gated activation (tanh ⊙ sigmoid)
      - Residual + skip connections
  - Per-node prediction head

Reference: Wu et al., "Graph WaveNet for Deep Spatial-Temporal Graph Modeling"
           (IJCAI 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """Graph convolution with support for multiple adjacency matrices."""

    def __init__(self, in_dim: int, out_dim: int, order: int = 2):
        super().__init__()
        self.order = order
        self.linear = nn.Linear(in_dim * (order + 1), out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) node features
            adj: (N, N) adjacency matrix
        Returns:
            (B, N, D_out)
        """
        out = [x]  # order 0 = identity
        h = x
        for _ in range(self.order):
            h = torch.einsum("bnd,nm->bmd", h, adj)
            out.append(h)
        out = torch.cat(out, dim=-1)  # (B, N, D*(order+1))
        return self.linear(out)


class STConvBlock(nn.Module):
    """
    Spatial-Temporal Convolution Block.

    Temporal: Dilated causal 1D conv (per-node)
    Spatial: Graph conv with mixed adjacency
    Gated activation: tanh(filter) ⊙ sigmoid(gate)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 2, dilation: int = 1,
                 graph_order: int = 2, dropout: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Temporal: dilated causal conv (filter + gate)
        self.padding = (kernel_size - 1) * dilation
        self.temporal_filter = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation,
        )
        self.temporal_gate = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation,
        )

        # Spatial: graph conv on predefined + adaptive adj
        self.graph_conv = GraphConv(out_channels, out_channels, graph_order)

        # Skip connection
        self.skip_conv = nn.Conv1d(out_channels, out_channels, 1)

        # Residual projection (if dims differ)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, T, D) input
            adj: (N, N) mixed adjacency
        Returns:
            residual: (B, N, T', D) for next block
            skip: (B, N, T', D) for skip aggregation
        """
        B, N, T, D = x.shape
        residual = x

        # Reshape for temporal conv: (B*N, D, T)
        h = x.reshape(B * N, T, D).permute(0, 2, 1)

        # Causal padding
        h_padded = F.pad(h, (self.padding, 0))

        # Gated activation
        h_filter = torch.tanh(self.temporal_filter(h_padded))
        h_gate = torch.sigmoid(self.temporal_gate(h_padded))
        h = h_filter * h_gate  # (B*N, out_channels, T)

        # Skip connection output
        skip = self.skip_conv(h)  # (B*N, out_channels, T)
        skip = skip.permute(0, 2, 1).reshape(B, N, T, -1)

        # Reshape for graph conv: (B, N, T, D)
        h = h.permute(0, 2, 1).reshape(B, N, T, -1)

        # Graph conv at each time step
        out_list = []
        for t in range(T):
            h_t = h[:, :, t, :]  # (B, N, D)
            g_t = self.graph_conv(h_t, adj)  # (B, N, D)
            out_list.append(g_t)
        h = torch.stack(out_list, dim=2)  # (B, N, T, D)

        h = self.dropout(h)

        # Residual connection
        if self.residual_conv is not None:
            res = residual.reshape(B * N, T, self.in_channels).permute(0, 2, 1)
            res = self.residual_conv(res).permute(0, 2, 1).reshape(B, N, T, -1)
        else:
            res = residual
        h = self.norm(h + res)

        return h, skip


class GraphWaveNet(nn.Module):
    """
    Graph WaveNet for charging demand prediction.

    Args:
        n_nodes: number of graph nodes (charging stations)
        in_features: input feature dimension per node per time step
        out_features: output dimension (prediction targets)
        hidden_dim: hidden channel dimension
        n_blocks: number of ST-Conv blocks
        kernel_size: temporal conv kernel size
        n_layers_per_block: layers per block (dilation doubles each layer)
        adaptive_adj: whether to learn adaptive adjacency
        node_emb_dim: embedding dimension for adaptive adjacency
        graph_order: diffusion order for graph conv
        dropout: dropout rate
        pred_horizon: number of future time steps to predict
    """

    def __init__(self, n_nodes: int = 53, in_features: int = 6,
                 out_features: int = 1, hidden_dim: int = 32,
                 n_blocks: int = 2, kernel_size: int = 2,
                 n_layers_per_block: int = 3,
                 adaptive_adj: bool = True, node_emb_dim: int = 16,
                 graph_order: int = 2, dropout: float = 0.2,
                 pred_horizon: int = 1):
        super().__init__()
        self.n_nodes = n_nodes
        self.pred_horizon = pred_horizon

        # Input projection
        self.input_proj = nn.Linear(in_features, hidden_dim)

        # Adaptive adjacency learnable embeddings
        self.adaptive_adj = adaptive_adj
        if adaptive_adj:
            self.node_emb_src = nn.Parameter(
                torch.randn(n_nodes, node_emb_dim)
            )
            self.node_emb_tgt = nn.Parameter(
                torch.randn(n_nodes, node_emb_dim)
            )

        # ST-Conv blocks with increasing dilation
        self.blocks = nn.ModuleList()
        for b in range(n_blocks):
            for l in range(n_layers_per_block):
                dilation = 2 ** l
                self.blocks.append(
                    STConvBlock(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        graph_order=graph_order,
                        dropout=dropout,
                    )
                )

        # Output: aggregate skips → prediction
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features * pred_horizon),
        )

        # Store last adjacency for interpretability
        self._last_adj = None

    def _compute_adaptive_adj(self) -> torch.Tensor:
        """Compute adaptive adjacency from learned embeddings."""
        # Softmax over source-target inner products
        logits = F.relu(
            torch.mm(self.node_emb_src, self.node_emb_tgt.T)
        )
        adj = F.softmax(logits, dim=-1)
        return adj

    def forward(self, x: torch.Tensor,
                adj_predefined: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, T, F) input features
            adj_predefined: (N, N) predefined adjacency matrix

        Returns:
            pred: (B, N, pred_horizon, out_features) predictions
        """
        B, N, T, _F = x.shape

        # Input projection
        h = self.input_proj(x)  # (B, N, T, hidden)

        # Build mixed adjacency
        adj = adj_predefined
        if self.adaptive_adj:
            adj_adaptive = self._compute_adaptive_adj()
            adj = adj + adj_adaptive
            self._last_adj = adj.detach()
        else:
            self._last_adj = adj

        # ST-Conv blocks
        skip_sum = torch.zeros_like(h)
        for block in self.blocks:
            h, skip = block(h, adj)
            # Align temporal dimensions (skip may differ in T)
            T_skip = skip.shape[2]
            T_sum = skip_sum.shape[2]
            if T_skip < T_sum:
                skip_sum = skip_sum[:, :, -T_skip:, :]
            skip_sum = skip_sum + skip

        # Aggregate: use last time step of skip sum
        out = F.gelu(skip_sum[:, :, -1, :])  # (B, N, hidden)

        # Per-node prediction
        pred = self.output_mlp(out)  # (B, N, out * horizon)
        pred = pred.view(B, N, self.pred_horizon, -1)

        return pred

    def get_learned_adjacency(self) -> torch.Tensor | None:
        """Return the last computed mixed adjacency for analysis."""
        return self._last_adj


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = GraphWaveNet(
        n_nodes=53, in_features=6, out_features=1,
        hidden_dim=32, n_blocks=2, n_layers_per_block=3,
        pred_horizon=1,
    )
    print(f"GraphWaveNet parameters: {count_parameters(model):,}")

    # Dummy forward pass
    B, N, T, n_feat = 4, 53, 24, 6
    x = torch.randn(B, N, T, n_feat)
    adj = torch.randn(N, N).abs()
    adj = adj / adj.sum(dim=-1, keepdim=True)

    pred = model(x, adj)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {pred.shape}")
    print(f"Expected:     (4, 53, 1, 1)")

    adj_learned = model.get_learned_adjacency()
    if adj_learned is not None:
        print(f"Learned adj:  {adj_learned.shape}")
