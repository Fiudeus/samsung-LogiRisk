"""
Metrics for driver-month ranking / screening.

Модуль намеренно не зависит от модели: на вход подаются бинарные метки (0/1)
и непрерывные risk-score (чем больше — тем рискованнее).

Это ранжирующие метрики (top-K), а не калибровка вероятностей.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _to_1d_numpy(x) -> np.ndarray:
    if isinstance(x, (pd.Series, pd.Index)):
        x = x.to_numpy()
    x = np.asarray(x)
    if x.ndim != 1:
        x = np.ravel(x)
    return x


def topk_indices(scores, k: int) -> np.ndarray:
    """Индексы top-K по убыванию score."""
    scores = _to_1d_numpy(scores).astype(float)
    k = int(min(max(k, 0), len(scores)))
    if k == 0:
        return np.array([], dtype=int)
    return np.argsort(scores)[::-1][:k]


def recall_at_k(y_true, scores, k: int) -> float:
    """Recall@K: доля всех позитивов, попавших в top-K."""
    y_true = _to_1d_numpy(y_true).astype(int)
    scores = _to_1d_numpy(scores).astype(float)

    k = int(min(max(k, 0), len(y_true)))
    if k == 0:
        return 0.0

    pos = int(y_true.sum())
    if pos == 0:
        return 0.0

    idx = topk_indices(scores, k)
    return float(y_true[idx].sum() / pos)


def precision_at_k(y_true, scores, k: int) -> float:
    """Precision@K: доля позитивов внутри top-K."""
    y_true = _to_1d_numpy(y_true).astype(int)
    scores = _to_1d_numpy(scores).astype(float)

    k = int(min(max(k, 0), len(y_true)))
    if k == 0:
        return 0.0

    idx = topk_indices(scores, k)
    return float(y_true[idx].mean())


def lift_at_k(y_true, scores, k: int) -> float:
    """Lift@K = Precision@K / base_rate."""
    y_true = _to_1d_numpy(y_true).astype(int)
    base = float(np.mean(y_true)) if len(y_true) else float("nan")
    if base == 0:
        return float("nan")
    return float(precision_at_k(y_true, scores, k) / base)


def tp_at_k(y_true, scores, k: int) -> int:
    """TP@K: сколько позитивов попало в top-K."""
    y_true = _to_1d_numpy(y_true).astype(int)
    scores = _to_1d_numpy(scores).astype(float)

    k = int(min(max(k, 0), len(y_true)))
    if k == 0:
        return 0

    idx = topk_indices(scores, k)
    return int(y_true[idx].sum())


def fn_at_k(y_true, scores, k: int) -> int:
    """FN@K: сколько позитивов НЕ попало в top-K (т.е. мы их 'признали безопасными')."""
    y_true = _to_1d_numpy(y_true).astype(int)
    pos = int(y_true.sum())
    return int(pos - tp_at_k(y_true, scores, k))


def miss_rate_at_k(y_true, scores, k: int) -> float:
    """MissRate@K = FN / Positives = 1 - Recall@K."""
    y_true = _to_1d_numpy(y_true).astype(int)
    pos = int(y_true.sum())
    if pos == 0:
        return 0.0
    return float(fn_at_k(y_true, scores, k) / pos)


@dataclass(frozen=True)
class DriverMonthMetricsConfig:
    ks: Sequence[int] = (10, 20, 30, 50, 100, 200)


def evaluate_driver_month(
    y_true,
    scores,
    tag: str | None = None,
    ks: Iterable[int] = (10, 20, 30, 50, 100, 200),
) -> pd.Series:
    """Единый отчёт метрик для driver-month предсказаний."""
    y_true_np = _to_1d_numpy(y_true).astype(int)
    scores_np = _to_1d_numpy(scores).astype(float)

    out: dict[str, float | int] = {}
    out["n"] = int(len(y_true_np))
    out["positives"] = int(y_true_np.sum())
    out["base_rate"] = float(np.mean(y_true_np)) if len(y_true_np) else float("nan")

    # Threshold-free метрики (корректны, только если есть оба класса)
    if len(np.unique(y_true_np)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true_np, scores_np))
        out["ap"] = float(average_precision_score(y_true_np, scores_np))
    else:
        out["roc_auc"] = float("nan")
        out["ap"] = float("nan")

    for k in ks:
        k = int(k)
        out[f"tp@{k}"] = int(tp_at_k(y_true_np, scores_np, k))
        out[f"fn@{k}"] = int(fn_at_k(y_true_np, scores_np, k))
        out[f"miss_rate@{k}"] = float(miss_rate_at_k(y_true_np, scores_np, k))
        out[f"recall@{k}"] = float(recall_at_k(y_true_np, scores_np, k))
        out[f"precision@{k}"] = float(precision_at_k(y_true_np, scores_np, k))
        out[f"lift@{k}"] = float(lift_at_k(y_true_np, scores_np, k))

    return pd.Series(out, name=tag or "driver_month")


def evaluate_driver_month_from_frame(
    df: pd.DataFrame,
    y_col: str,
    score_col: str,
    tag: str | None = None,
    ks: Iterable[int] = (10, 20, 30, 50, 100, 200),
) -> pd.Series:
    """То же самое, но когда y и score лежат в DataFrame."""
    return evaluate_driver_month(df[y_col], df[score_col], tag=tag, ks=ks)