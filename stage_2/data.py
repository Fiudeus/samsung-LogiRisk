from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# -------------------------
# Paths / config
# -------------------------
ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"

# ВАЖНО: зафиксируй реальную колонку месяца в driver_monthly_metrics.csv
MONTH_COL = "month"  # <-- поменяй на правильную, если у вас иначе


@dataclass(frozen=True)
class DatasetConfig:
    test_months: int = 6
    horizon_months: int = 3
    # окна для исторических фич
    history_windows: Sequence[int] = (3, 6, 12)


# -------------------------
# Utils
# -------------------------
def safe_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def to_month_start(dt: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp()


def train_test_split_by_time(df: pd.DataFrame, test_months: int = 6):
    months = np.sort(df["month"].unique())
    if len(months) <= test_months + 3:
        raise ValueError("Not enough months to create a stable train/test split.")
    test_start = pd.to_datetime(months[-test_months])
    train = df[df["month"] < test_start].copy()
    test = df[df["month"] >= test_start].copy()
    return train, test, test_start


# -------------------------
# Panel building
# -------------------------
def build_driver_month_incidents(trips: pd.DataFrame, safety_incidents: pd.DataFrame) -> pd.DataFrame:
    trips = trips.copy()
    trips["dispatch_date"] = pd.to_datetime(trips["dispatch_date"], errors="coerce")
    trips = trips.dropna(subset=["dispatch_date", "driver_id", "trip_id"])
    trips["month"] = to_month_start(trips["dispatch_date"])

    trip_has_inc = trips[["trip_id"]].copy()
    trip_has_inc["has_incident"] = trip_has_inc["trip_id"].isin(safety_incidents["trip_id"]).astype(int)

    t = trips.merge(trip_has_inc, on="trip_id", how="left")
    t["has_incident"] = t["has_incident"].fillna(0).astype(int)

    agg = t.groupby(["driver_id", "month"], as_index=False).agg(
        trips_cnt=("trip_id", "count"),
        incidents_cnt=("has_incident", "sum"),
    )
    return agg


def add_rolling_history(panel: pd.DataFrame, windows: Sequence[int] = (3, 6, 12)) -> pd.DataFrame:
    df = panel.sort_values(["driver_id", "month"]).copy()

    for w in windows:
        df[f"incidents_prev_{w}m"] = (
            df.groupby("driver_id")["incidents_cnt"]
            .shift(1)
            .rolling(window=w, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        df[f"trips_prev_{w}m"] = (
            df.groupby("driver_id")["trips_cnt"]
            .shift(1)
            .rolling(window=w, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        df[f"incident_rate_prev_{w}m"] = df[f"incidents_prev_{w}m"] / df[f"trips_prev_{w}m"].replace(0, np.nan)

    def months_since_last_incident(g: pd.DataFrame) -> pd.Series:
        has = (g["incidents_cnt"] > 0).astype(int).values
        last = np.where(has == 1, np.arange(len(has)), -1)
        last = np.maximum.accumulate(last)
        last = np.roll(last, 1)
        last[0] = -1
        dist = np.arange(len(has)) - last
        dist[last < 0] = 999
        return pd.Series(dist, index=g.index)

    df["months_since_last_incident"] = df.groupby("driver_id", group_keys=False).apply(months_since_last_incident)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def add_target_next_3m(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["driver_id", "month"]).copy()

    def future_any(g: pd.DataFrame) -> pd.Series:
        fut = g["incidents_cnt"].shift(-1).rolling(window=3, min_periods=1).sum()
        return (fut > 0).astype(int)

    df["has_incident_next_3m"] = df.groupby("driver_id", group_keys=False).apply(future_any)
    return df


def build_panel(cfg: DatasetConfig = DatasetConfig()) -> tuple[pd.DataFrame, list[str], str]:
    """
    Возвращает:
    - panel: dataframe с колонками driver_id, month, фичами и target
    - feature_cols: список фичей (как в 06_...: perf + history + months_since_last_incident)
    - target: имя таргета (строка)
    """
    trips_path = DATASETS / "trips.csv"
    inc_path = DATASETS / "safety_incidents.csv"
    metrics_path = DATASETS / "driver_monthly_metrics.csv"

    trips = pd.read_csv(trips_path)
    safety_incidents = pd.read_csv(inc_path)
    driver_monthly_metrics = pd.read_csv(metrics_path)

    if MONTH_COL not in driver_monthly_metrics.columns:
        raise KeyError(
            f"MONTH_COL='{MONTH_COL}' not found in driver_monthly_metrics.csv. "
            f"Available columns: {list(driver_monthly_metrics.columns)[:30]}..."
        )

    driver_monthly_metrics = driver_monthly_metrics.copy()
    driver_monthly_metrics["month"] = to_month_start(driver_monthly_metrics[MONTH_COL])

    id_cols = {"driver_id", "month"}
    perf_cols = [
        c
        for c in driver_monthly_metrics.columns
        if c not in id_cols and pd.api.types.is_numeric_dtype(driver_monthly_metrics[c])
    ]

    perf = driver_monthly_metrics[["driver_id", "month"] + perf_cols].copy()

    inc_panel = build_driver_month_incidents(trips, safety_incidents)

    panel = perf.merge(inc_panel, on=["driver_id", "month"], how="left")
    panel["trips_cnt"] = panel["trips_cnt"].fillna(0).astype(int)
    panel["incidents_cnt"] = panel["incidents_cnt"].fillna(0).astype(int)

    panel = panel.sort_values(["driver_id", "month"]).reset_index(drop=True)
    panel = add_rolling_history(panel, windows=cfg.history_windows)
    panel = add_target_next_3m(panel)

    # Drop last horizon months (чтобы таргет "следующие 3 месяца" был определён)
    max_month = panel["month"].max()
    panel = panel[panel["month"] <= (max_month - pd.offsets.MonthBegin(cfg.horizon_months))].copy()

    target = "has_incident_next_3m"
    feature_cols = (
        perf_cols
        + [c for c in panel.columns if c.startswith("incidents_prev_")]
        + [c for c in panel.columns if c.startswith("trips_prev_")]
        + [c for c in panel.columns if c.startswith("incident_rate_prev_")]
        + ["months_since_last_incident"]
    )
    feature_cols = [c for c in feature_cols if c in panel.columns]

    return panel, feature_cols, target