"""혼합 멀티태스크 손실 — masked MSE(회귀) + masked BCE(분류).

task-type 벡터(0=reg, 1=cls)로 task별 손실을 자동 분기하고,
관측 마스크(결측 라벨)와 task 가중치로 가중합한다. Tox21 마스킹 손실의 확장.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMultiTaskLoss(nn.Module):
    """회귀/분류 혼합 멀티태스크 손실.

    Args:
        task_types: (T,) long tensor. 0=reg(MSE), 1=cls(BCEWithLogits).
        task_weights: (T,) float tensor 또는 None(=1).
        reg_weight / cls_weight: 회귀/분류 그룹 전체 스케일(혼합 균형용).
    """

    def __init__(
        self,
        task_types,
        task_weights: Optional[torch.Tensor] = None,
        reg_weight: float = 1.0,
        cls_weight: float = 1.0,
    ):
        super().__init__()
        tt = torch.as_tensor(task_types, dtype=torch.long)
        self.register_buffer("task_types", tt)
        if task_weights is None:
            task_weights = torch.ones(len(tt), dtype=torch.float)
        self.register_buffer("task_weights", torch.as_tensor(task_weights, dtype=torch.float))
        self.reg_weight = reg_weight
        self.cls_weight = cls_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """preds/targets/mask: (B, T). mask=1 관측, 0 결측."""
        is_cls = (self.task_types == 1).view(1, -1).to(preds.device)
        mask = mask.float()

        # 회귀: MSE (정규화된 타깃 공간에서)
        mse = F.mse_loss(preds, torch.nan_to_num(targets), reduction="none")
        # 분류: BCEWithLogits
        bce = F.binary_cross_entropy_with_logits(
            preds, torch.nan_to_num(targets), reduction="none"
        )

        per_elem = torch.where(is_cls, self.cls_weight * bce, self.reg_weight * mse)
        per_elem = per_elem * mask * self.task_weights.view(1, -1).to(preds.device)

        denom = (mask * self.task_weights.view(1, -1).to(preds.device)).sum().clamp_min(1.0)
        return per_elem.sum() / denom


def masked_mse(preds, targets, mask):
    mask = mask.float()
    se = F.mse_loss(preds, torch.nan_to_num(targets), reduction="none") * mask
    return se.sum() / mask.sum().clamp_min(1.0)


def masked_bce(preds, targets, mask):
    mask = mask.float()
    bce = F.binary_cross_entropy_with_logits(
        preds, torch.nan_to_num(targets), reduction="none"
    ) * mask
    return bce.sum() / mask.sum().clamp_min(1.0)
