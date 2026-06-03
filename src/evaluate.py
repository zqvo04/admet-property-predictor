"""평가 — 회귀(RMSE,MAE,Pearson,Spearman) + 분류(ROC-AUC,PR-AUC,F1)
+ TDC 공식 지표 래퍼(리더보드 동일 기준).

회귀 예측은 정규화 공간 -> 스케일러로 원단위 역변환 후 지표 계산.
분류 예측은 logit -> sigmoid 확률로 변환 후 지표 계산.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (average_precision_score, f1_score,
                             mean_absolute_error, mean_squared_error,
                             roc_auc_score)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# ---------------------------------------------------------------------------
# 단일 task 지표
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    pear = float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan")
    spear = float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan")
    return {"rmse": rmse, "mae": mae, "pearson": pear, "spearman": spear}


def classification_metrics(y_true, y_score, thr: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_score = np.asarray(y_score, float)
    m = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[m], y_score[m]
    out: Dict[str, float] = {}
    if len(np.unique(y_true)) < 2:
        return {"roc-auc": float("nan"), "pr-auc": float("nan"), "f1": float("nan")}
    out["roc-auc"] = float(roc_auc_score(y_true, y_score))
    out["pr-auc"] = float(average_precision_score(y_true, y_score))
    out["f1"] = float(f1_score(y_true, (y_score >= thr).astype(int)))
    return out


# ---------------------------------------------------------------------------
# 멀티태스크 예측 -> task별 지표 (원단위 역변환 포함)
# ---------------------------------------------------------------------------
def evaluate_multitask(
    preds: np.ndarray,            # (N, T) 회귀=정규화공간값, 분류=logit
    Y: np.ndarray,                # (N, T) 정규화공간/라벨, NaN=결측
    mask: np.ndarray,             # (N, T)
    task_names: List[str],
    task_types: List[str],        # 'reg'|'cls'
    scalers: Optional[Dict] = None,
) -> Dict[str, Dict[str, float]]:
    scalers = scalers or {}
    results: Dict[str, Dict[str, float]] = {}
    for ti, (name, ttype) in enumerate(zip(task_names, task_types)):
        sel = mask[:, ti].astype(bool)
        if sel.sum() == 0:
            results[name] = {}
            continue
        yt, yp = Y[sel, ti], preds[sel, ti]
        if ttype == "reg":
            sc = scalers.get(name)
            if sc is not None:                       # 원단위 역변환 후 지표
                yt = sc.inverse_transform(yt)
                yp = sc.inverse_transform(yp)
            results[name] = regression_metrics(yt, yp)
        else:
            results[name] = classification_metrics(yt, _sigmoid(yp))
    return results


# ---------------------------------------------------------------------------
# TDC 공식 지표 래퍼 (리더보드와 동일 기준)
# ---------------------------------------------------------------------------
# 엔드포인트명 -> 공식 지표 (data.ENDPOINTS와 일치)
def official_metric_value(metric: str, y_true, y_pred) -> float:
    """endpoint의 TDC 공식 지표 1개 값만 반환."""
    metric = metric.lower()
    if metric == "mae":
        return regression_metrics(y_true, y_pred)["mae"]
    if metric == "rmse":
        return regression_metrics(y_true, y_pred)["rmse"]
    if metric == "spearman":
        return regression_metrics(y_true, y_pred)["spearman"]
    if metric == "pearson":
        return regression_metrics(y_true, y_pred)["pearson"]
    if metric in ("roc-auc", "auroc"):
        return classification_metrics(y_true, y_pred)["roc-auc"]
    if metric in ("pr-auc", "auprc"):
        return classification_metrics(y_true, y_pred)["pr-auc"]
    if metric == "f1":
        return classification_metrics(y_true, y_pred)["f1"]
    raise ValueError(f"unknown metric {metric}")


def tdc_evaluate(group, predictions_list):
    """TDC benchmark group 공식 평가 래퍼.

    predictions_list: [{tdc_name: y_pred_test}, ...]  (seed별)
    반환: {tdc_name: [mean, std]} — 리더보드 동일 기준.
    """
    return group.evaluate_many(predictions_list)


def summarize(results: Dict[str, Dict[str, float]],
              endpoints_meta: Optional[Dict] = None) -> "object":
    """task별 지표 dict -> pandas DataFrame. endpoints_meta 있으면 공식지표 컬럼 강조."""
    import pandas as pd
    rows = []
    for name, m in results.items():
        row = {"endpoint": name}
        row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
        if endpoints_meta and name in endpoints_meta:
            off = endpoints_meta[name].get("metric")
            row["official_metric"] = off
            row["official_value"] = row.get(off)
        rows.append(row)
    return pd.DataFrame(rows)
