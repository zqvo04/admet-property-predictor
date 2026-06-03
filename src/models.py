"""모델 — ECFPNet(MLP) / GNNNet(GCN|GIN). 공유 trunk + task별 출력 head.

회귀/분류 공용: 분류 출력은 logit(평가 시 sigmoid), 회귀 출력은 정규화 공간 값.
멀티태스크: 마지막에 task 수(T) 차원 출력.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .featurizers import ATOM_FEATURE_DIM


class MultiHead(nn.Module):
    """공유 표현 -> task별 1차원 출력 head (총 T개)."""

    def __init__(self, in_dim: int, num_tasks: int, hidden: int = 0):
        super().__init__()
        if hidden > 0:
            self.heads = nn.ModuleList([
                nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                              nn.Linear(hidden, 1))
                for _ in range(num_tasks)
            ])
        else:
            self.heads = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(num_tasks)])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(h) for head in self.heads], dim=1)  # (B, T)


# ---------------------------------------------------------------------------
# ECFP MLP
# ---------------------------------------------------------------------------
class ECFPNet(nn.Module):
    """2048 -> 512 -> 128 -> multi-head. 회귀/분류 공용."""

    def __init__(self, num_tasks: int, in_dim: int = 2048,
                 dims=(512, 128), dropout: float = 0.25, head_hidden: int = 0):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for d in dims:
            layers += [nn.Linear(prev, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(dropout)]
            prev = d
        self.trunk = nn.Sequential(*layers)
        self.head = MultiHead(prev, num_tasks, hidden=head_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


# ---------------------------------------------------------------------------
# GNN (GCN / GIN)
# ---------------------------------------------------------------------------
class GNNNet(nn.Module):
    """GCNConv 또는 GINConv ×n -> global_mean_pool -> multi-head."""

    def __init__(self, num_tasks: int, conv_type: str = "gin",
                 in_dim: int = ATOM_FEATURE_DIM, hidden: int = 128,
                 num_layers: int = 3, dropout: float = 0.2, head_hidden: int = 64):
        super().__init__()
        from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
        self.global_mean_pool = global_mean_pool
        self.conv_type = conv_type.lower()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        prev = in_dim
        for _ in range(num_layers):
            if self.conv_type == "gin":
                mlp = nn.Sequential(nn.Linear(prev, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden))
                self.convs.append(GINConv(mlp))
            elif self.conv_type == "gcn":
                self.convs.append(GCNConv(prev, hidden))
            else:
                raise ValueError(f"unknown conv_type {conv_type}")
            self.bns.append(nn.BatchNorm1d(hidden))
            prev = hidden

        self.head = MultiHead(hidden, num_tasks, hidden=head_hidden)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.global_mean_pool(x, batch)   # (B, hidden)
        return self.head(h)


def build_model(kind: str, num_tasks: int, **kwargs) -> nn.Module:
    kind = kind.lower()
    if kind in ("ecfp", "mlp", "ecfpnet"):
        return ECFPNet(num_tasks, **kwargs)
    if kind in ("gin", "gcn"):
        return GNNNet(num_tasks, conv_type=kind, **kwargs)
    raise ValueError(f"unknown model kind {kind}")
