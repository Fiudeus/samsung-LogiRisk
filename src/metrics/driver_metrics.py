"""
Driver-level metrics.

Сценарий:
- у нас есть предсказания на уровне driver-month (driver_id, month, y_true, score)
- мы агрегируем score в один скор на водителя на тестовом окне
- и считаем ранжирующие метрики уже по списку водителей.

Важно:
y_true на driver-level получается как "был ли позитив хоть раз в тестовом окне" (any).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .driver_month_metrics import (
    lift_at_k,
    precision_at_k,
    recall_at_k,
    tp_at_k,
    fn_at_k,
    miss_rate_at_k
)

AggMethod = Literal["max", "mean", "last", "topq_mean"]


@dataclass(frozen=True)
class DriverAggregationConfig:
    driver_col: str = "driver_id"
    month_col: str = "month"
    y_col: str = "y_true"
    score_col: str = "score"
    agg: AggMethod = "max"
    # используется только для topq_mean
    q: float = 0.2


def _ensure_datetime_month(s: pd.Series) -> pd.Series:
    # допускаем, что month может быть уже datetime/period/строкой
    dt = pd.to_datetime(s, errors="coerce")
    return dt


def aggregate_driver_scores(
    df: pd.DataFrame,
    cfg: DriverAggregationConfig = DriverAggregationConfig(),
) -> pd.DataFrame:
    """
    Преобразует driver-month предсказания в driver-level.

    Возвращает df с колонками:
    - driver_id
    - y_true (0/1): был ли позитив в тестовом окне хотя бы раз
    - score: агрегированный риск
    - n_months: сколько месяцев было у водителя в df
    """
    need = {cfg.driver_col, cfg.y_col, cfg.score_col}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    work = df.copy()

    if cfg.month_col in work.columns:
        work[cfg.month_col] = _ensure_datetime_month(work[cfg.month_col])

    # y_true по водителю: любой позитив
    y_drv = (
        work.groupby(cfg.driver_col, as_index=False)[cfg.y_col]
        .max()
        .rename(columns={cfg.y_col: "y_true"})
    )

    # сколько месяцев
    if cfg.month_col in work.columns:
        n_months = (
            work.groupby(cfg.driver_col, as_index=False)[cfg.month_col]
            .nunique()
            .rename(columns={cfg.month_col: "n_months"})
        )
    else:
        n_months = (
            work.groupby(cfg.driver_col, as_index=False)[cfg.score_col]
            .size()
            .rename(columns={"size": "n_months"})
        )

    # score агрегация
    if cfg.agg == "max":
        s_drv = (
            work.groupby(cfg.driver_col, as_index=False)[cfg.score_col]
            .max()
            .rename(columns={cfg.score_col: "score"})
        )

    elif cfg.agg == "mean":
        s_drv = (
            work.groupby(cfg.driver_col, as_index=False)[cfg.score_col]
            .mean()
            .rename(columns={cfg.score_col: "score"})
        )

    elif cfg.agg == "last":
        if cfg.month_col not in work.columns:
            raise KeyError("agg='last' requires month_col to exist in df")
        # берём последний месяц в рамках df (тестового окна)
        work2 = work.sort_values([cfg.driver_col, cfg.month_col])
        last_rows = work2.groupby(cfg.driver_col, as_index=False).tail(1)
        s_drv = last_rows[[cfg.driver_col, cfg.score_col]].rename(columns={cfg.score_col: "score"})

    elif cfg.agg == "topq_mean":
        if not (0.0 < cfg.q <= 1.0):
            raise ValueError("q must be in (0, 1]")
        # среднее по верхним q-доле месяцев (по score)
        def _topq_mean(g: pd.DataFrame) -> float:
            g = g.sort_values(cfg.score_col, ascending=False)
            k = max(1, int(np.ceil(len(g) * cfg.q)))
            return float(g[cfg.score_col].head(k).mean())

        s_drv = (
            work.groupby(cfg.driver_col, as_index=False)
            .apply(lambda g: _topq_mean(g))
            .rename(columns={None: "score"})
        )
        # после groupby.apply driver_id становится колонкой только в новых pandas, подстрахуемся:
        if cfg.driver_col not in s_drv.columns:
            s_drv = s_drv.rename(columns={"index": cfg.driver_col})

    else:
        raise ValueError(f"Unknown agg: {cfg.agg}")

    out = y_drv.merge(s_drv, on=cfg.driver_col, how="inner").merge(n_months, on=cfg.driver_col, how="left")
    return out


def evaluate_driver(
    y_true,
    scores,
    tag: str | None = None,
    ks: Iterable[int] = (5, 10, 20, 30),
) -> pd.Series:
    """Метрики на уровне водителей."""
    y_true_np = np.asarray(y_true).astype(int).ravel()
    scores_np = np.asarray(scores).astype(float).ravel()

    out: dict[str, float | int] = {}
    out["n_drivers"] = int(len(y_true_np))
    out["positives"] = int(y_true_np.sum())
    out["base_rate"] = float(np.mean(y_true_np)) if len(y_true_np) else float("nan")

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

    return pd.Series(out, name=tag or "driver")


def evaluate_driver_from_frame(
    df_driver: pd.DataFrame,
    y_col: str = "y_true",
    score_col: str = "score",
    tag: str | None = None,
    ks: Iterable[int] = (5, 10, 20, 30),
) -> pd.Series:
    return evaluate_driver(df_driver[y_col], df_driver[score_col], tag=tag, ks=ks)


def evaluate_driver_from_driver_month_frame(
    df_driver_month: pd.DataFrame,
    cfg: DriverAggregationConfig = DriverAggregationConfig(),
    tag: str | None = None,
    ks: Iterable[int] = (5, 10, 20, 30),
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Удобный хелпер: подал driver-month df → получил (report, driver_df).
    """
    driver_df = aggregate_driver_scores(df_driver_month, cfg=cfg)
    report = evaluate_driver_from_frame(driver_df, tag=tag, ks=ks)
    return report, driver_df

def evaluate_driver_many_aggs_from_driver_month_frame(
    df_driver_month: pd.DataFrame,
    *,
    driver_col: str = "driver_id",
    month_col: str = "month",
    y_col: str = "y_true",
    score_col: str = "score",
    ks: Iterable[int] = (5, 10, 20, 30),
    q: float = 0.2,
    tag_prefix: str | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Считает driver-level метрики сразу для 4 агрегаций:
    - max
    - mean
    - last
    - topq_mean (q по умолчанию 0.2)

    Возвращает:
    - metrics_df: DataFrame где строки = агрегации, колонки = метрики
    - ranked_drivers: dict[agg_name] -> driver_df (driver_id, y_true, score, n_months), отсортировать можно снаружи

    Никаких переименований колонок не делает: ожидает стандартный df_driver_month
    с колонками driver_id/month/y_true/score (или передай свои имена параметрами).
    """
    aggs: list[tuple[str, DriverAggregationConfig]] = [
        ("max", DriverAggregationConfig(driver_col=driver_col, month_col=month_col, y_col=y_col, score_col=score_col, agg="max")),
        ("mean", DriverAggregationConfig(driver_col=driver_col, month_col=month_col, y_col=y_col, score_col=score_col, agg="mean")),
        ("last", DriverAggregationConfig(driver_col=driver_col, month_col=month_col, y_col=y_col, score_col=score_col, agg="last")),
        ("topq_mean", DriverAggregationConfig(driver_col=driver_col, month_col=month_col, y_col=y_col, score_col=score_col, agg="topq_mean", q=q)),
    ]

    rows = []
    ranked: dict[str, pd.DataFrame] = {}

    for agg_name, cfg in aggs:
        tag = agg_name if tag_prefix is None else f"{tag_prefix}_{agg_name}"
        rep, driver_df = evaluate_driver_from_driver_month_frame(
            df_driver_month,
            cfg=cfg,
            tag=tag,
            ks=ks,
        )
        ranked[agg_name] = driver_df.sort_values("score", ascending=False)
        r = rep.to_dict()
        r["agg"] = agg_name
        rows.append(r)

    metrics_df = pd.DataFrame(rows).set_index("agg").sort_index()
    return metrics_df, ranked


def print_driver_agg_report(metrics_df: pd.DataFrame) -> None:
    """
    Удобная печать таблицы метрик по агрегациям.
    """
    # Чтобы вывод был стабильный и читаемый
    with pd.option_context("display.max_columns", 200, "display.width", 200):
        print(metrics_df.to_string())