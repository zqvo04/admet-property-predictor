"""공통 Trainer — Tox21 Trainer 확장. 회귀/분류 혼합 멀티태스크 공용.

기능:
  - ECFP(텐서 배치) / GNN(PyG 배치) 공통 forward 어댑터.
  - masked 혼합 손실(losses.MaskedMultiTaskLoss).
  - early stopping(valid 손실 기준) + best 체크포인트 저장.
  - 학습곡선(train/valid loss) 기록 + 저장.
체크포인트 기본 경로: /content/drive/MyDrive/admet/ (Colab). 없으면 ./checkpoints/.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .losses import MaskedMultiTaskLoss


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class ECFPDataset(Dataset):
    """X(N,nbits), Y(N,T), mask(N,T) -> 텐서 배치."""

    def __init__(self, X, Y, mask):
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float)
        self.Y = torch.as_tensor(np.asarray(Y), dtype=torch.float)
        self.mask = torch.as_tensor(np.asarray(mask), dtype=torch.float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.mask[i]


def graph_dataset(graphs, Y, mask):
    """PyG Data 리스트에 (T,) 타깃/마스크를 부착해 반환 (PyG DataLoader용)."""
    Y = np.asarray(Y, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    out = []
    for i, g in enumerate(graphs):
        g = g.clone() if hasattr(g, "clone") else g
        g.y = torch.as_tensor(Y[i], dtype=torch.float).view(1, -1)
        g.mask = torch.as_tensor(mask[i], dtype=torch.float).view(1, -1)
        out.append(g)
    return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 15
    ckpt_dir: str = "/content/drive/MyDrive/admet/"
    run_name: str = "model"
    device: Optional[str] = None
    grad_clip: float = 5.0
    verbose: bool = True


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(self, model, task_types, cfg: TrainConfig,
                 reg_weight: float = 1.0, cls_weight: float = 1.0,
                 task_weights=None):
        self.cfg = cfg
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.is_graph = model.__class__.__name__ == "GNNNet"
        self.criterion = MaskedMultiTaskLoss(
            task_types, task_weights=task_weights,
            reg_weight=reg_weight, cls_weight=cls_weight
        ).to(self.device)
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                    weight_decay=cfg.weight_decay)
        self.history: Dict[str, List[float]] = {"train_loss": [], "valid_loss": []}
        self.ckpt_path = self._resolve_ckpt()

    def _resolve_ckpt(self) -> str:
        d = self.cfg.ckpt_dir
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            d = "./checkpoints/"
            os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{self.cfg.run_name}.pt")

    # --- forward 어댑터: ECFP 텐서배치 vs PyG 배치 공용 ---
    def _forward_batch(self, batch):
        if self.is_graph:
            batch = batch.to(self.device)
            preds = self.model(batch)
            targets = batch.y.view(preds.shape)
            mask = batch.mask.view(preds.shape)
        else:
            x, targets, mask = batch
            x, targets, mask = x.to(self.device), targets.to(self.device), mask.to(self.device)
            preds = self.model(x)
        return preds, targets, mask

    def _run_epoch(self, loader, train: bool) -> float:
        self.model.train(train)
        total, nb = 0.0, 0
        torch.set_grad_enabled(train)
        for batch in loader:
            preds, targets, mask = self._forward_batch(batch)
            loss = self.criterion(preds, targets, mask)
            if train:
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.opt.step()
            total += float(loss.item())
            nb += 1
        torch.set_grad_enabled(True)
        return total / max(nb, 1)

    def fit(self, train_loader, valid_loader=None):
        best = float("inf")
        bad = 0
        for ep in range(1, self.cfg.epochs + 1):
            tr = self._run_epoch(train_loader, train=True)
            va = self._run_epoch(valid_loader, train=False) if valid_loader else tr
            self.history["train_loss"].append(tr)
            self.history["valid_loss"].append(va)
            improved = va < best - 1e-5
            if improved:
                best, bad = va, 0
                torch.save({"model": self.model.state_dict(),
                            "history": self.history, "epoch": ep}, self.ckpt_path)
            else:
                bad += 1
            if self.cfg.verbose and (ep == 1 or ep % 5 == 0 or improved):
                print(f"  epoch {ep:3d} | train {tr:.4f} | valid {va:.4f}"
                      f"{' *' if improved else ''}")
            if bad >= self.cfg.patience:
                if self.cfg.verbose:
                    print(f"  early stop @ epoch {ep} (best valid {best:.4f})")
                break
        self.load_best()
        return self.history

    def load_best(self):
        if os.path.exists(self.ckpt_path):
            ck = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(ck["model"])

    @torch.no_grad()
    def predict(self, loader) -> np.ndarray:
        self.model.eval()
        outs = []
        for batch in loader:
            preds, _, _ = self._forward_batch(batch)
            outs.append(preds.cpu().numpy())
        return np.concatenate(outs, axis=0)

    def save_curve(self, path: Optional[str] = None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        path = path or os.path.join(os.path.dirname(self.ckpt_path),
                                    f"{self.cfg.run_name}_curve.png")
        plt.figure(figsize=(6, 4))
        plt.plot(self.history["train_loss"], label="train")
        plt.plot(self.history["valid_loss"], label="valid")
        plt.xlabel("epoch"); plt.ylabel("masked loss"); plt.legend()
        plt.title(self.cfg.run_name); plt.tight_layout()
        plt.savefig(path, dpi=120); plt.close()
        return path
