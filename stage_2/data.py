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

MONTH_COL = "month"


@dataclass(frozen=True)
class DatasetConfig:
    test_months: int = 6
    horizon_months: int = 3
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
# Feature Engineering Modules
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
        unique_trucks=("truck_id", "nunique"),
    )

    agg["truck_churn_rate"] = agg["unique_trucks"] / agg["trips_cnt"].replace(0, np.nan)
    agg["truck_churn_rate"] = agg["truck_churn_rate"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return agg


def build_detention_features(trips: pd.DataFrame, delivery_events: pd.DataFrame) -> pd.DataFrame:
    ev = delivery_events[["trip_id", "detention_minutes"]].dropna()
    t = trips[["trip_id", "driver_id", "dispatch_date"]].dropna()
    t["month"] = to_month_start(t["dispatch_date"])

    merged = t.merge(ev, on="trip_id", how="inner")
    agg = merged.groupby(["driver_id", "month"], as_index=False).agg(
        detention_minutes_sum=("detention_minutes", "sum"),
        detention_minutes_avg=("detention_minutes", "mean")
    )
    return agg


def add_rolling_and_lifetime_history(panel: pd.DataFrame, windows: Sequence[int] = (3, 6, 12)) -> pd.DataFrame:
    df = panel.sort_values(["driver_id", "month"]).copy()

    for w in windows:
        df[f"incidents_prev_{w}m"] = (
            df.groupby("driver_id")["incidents_cnt"]
            .shift(1).rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)
        )
        df[f"trips_prev_{w}m"] = (
            df.groupby("driver_id")["trips_cnt"]
            .shift(1).rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)
        )
        df[f"incident_rate_prev_{w}m"] = df[f"incidents_prev_{w}m"] / df[f"trips_prev_{w}m"].replace(0, np.nan)
        df[f"incident_rate_prev_{w}m"] = df[f"incident_rate_prev_{w}m"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["lifetime_trips_cnt"] = df.groupby("driver_id")["trips_cnt"].shift(1).expanding().sum().reset_index(level=0,
                                                                                                           drop=True)
    df["lifetime_incidents_cnt"] = df.groupby("driver_id")["incidents_cnt"].shift(1).expanding().sum().reset_index(
        level=0, drop=True)

    df["lifetime_incident_rate"] = df["lifetime_incidents_cnt"] / df["lifetime_trips_cnt"].replace(0, np.nan)
    df["lifetime_incident_rate"] = df["lifetime_incident_rate"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df[["lifetime_trips_cnt", "lifetime_incidents_cnt"]] = df[["lifetime_trips_cnt", "lifetime_incidents_cnt"]].fillna(
        0)

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
    return df


def add_perf_dynamics(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["driver_id", "month"]).copy()

    df["mpg_roll_3m"] = df.groupby("driver_id")["average_mpg"].shift(1).rolling(3, min_periods=1).mean()
    df["idle_roll_3m"] = df.groupby("driver_id")["average_idle_hours"].shift(1).rolling(3, min_periods=1).mean()
    df["ontime_roll_3m"] = df.groupby("driver_id")["on_time_delivery_rate"].shift(1).rolling(3, min_periods=1).mean()

    df["mpg_drop_ratio"] = df["average_mpg"] / df["mpg_roll_3m"].replace(0, np.nan)
    df["idle_spike_ratio"] = df["average_idle_hours"] / df["idle_roll_3m"].replace(0, np.nan)
    df["ontime_drop_ratio"] = df["on_time_delivery_rate"] / df["ontime_roll_3m"].replace(0, np.nan)

    df = df.drop(columns=["mpg_roll_3m", "idle_roll_3m", "ontime_roll_3m"])

    ratio_cols = ["mpg_drop_ratio", "idle_spike_ratio", "ontime_drop_ratio"]
    df[ratio_cols] = df[ratio_cols].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    return df


def add_seasonality_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет признаки сезонности: циклическое кодирование месяца и бизнес-кварталы.
    """
    df = panel.copy()

    # Извлекаем номер месяца (1-12)
    month_num = df["month"].dt.month

    # 1. Циклическое кодирование (переводим месяц в координаты на окружности)
    # Это позволяет алгоритму понять, что декабрь (12) и январь (1) находятся рядом
    df["month_sin"] = np.sin(2 * np.pi * month_num / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month_num / 12.0)

    # 2. Квартал (1-4) - важно для бизнес-циклов в логистике
    df["quarter"] = df["month"].dt.quarter

    # 3. Флаг высокого сезона (4-й квартал: праздники, сложные погодные условия)
    df["is_high_season"] = (df["quarter"] == 4).astype(int)

    return df


def add_target_next_3m(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["driver_id", "month"]).copy()

    def future_any(g: pd.DataFrame) -> pd.Series:
        fut = g["incidents_cnt"].shift(-1).rolling(window=3, min_periods=1).sum()
        return (fut > 0).astype(int)

    df["has_incident_next_3m"] = df.groupby("driver_id", group_keys=False).apply(future_any)
    return df


# -------------------------
# Main Pipeline
# -------------------------
def build_panel(cfg: DatasetConfig = DatasetConfig()) -> tuple[pd.DataFrame, list[str], str]:
    trips = pd.read_csv(DATASETS / "trips.csv")
    safety_incidents = pd.read_csv(DATASETS / "safety_incidents.csv")
    driver_monthly_metrics = pd.read_csv(DATASETS / "driver_monthly_metrics.csv")
    delivery_events = pd.read_csv(DATASETS / "delivery_events.csv")
    drivers = pd.read_csv(DATASETS / "drivers.csv")

    if MONTH_COL not in driver_monthly_metrics.columns:
        raise KeyError(f"MONTH_COL='{MONTH_COL}' not found in metrics.")

    driver_monthly_metrics = driver_monthly_metrics.copy()
    driver_monthly_metrics["month"] = to_month_start(driver_monthly_metrics[MONTH_COL])

    id_cols = {"driver_id", "month"}
    perf_cols = [
        c for c in driver_monthly_metrics.columns
        if c not in id_cols and pd.api.types.is_numeric_dtype(driver_monthly_metrics[c])
    ]
    perf = driver_monthly_metrics[["driver_id", "month"] + perf_cols].copy()

    inc_panel = build_driver_month_incidents(trips, safety_incidents)
    detention_panel = build_detention_features(trips, delivery_events)

    panel = perf.merge(inc_panel, on=["driver_id", "month"], how="left")
    panel = panel.merge(detention_panel, on=["driver_id", "month"], how="left")

    panel["trips_cnt"] = panel["trips_cnt"].fillna(0).astype(int)
    panel["incidents_cnt"] = panel["incidents_cnt"].fillna(0).astype(int)
    panel["unique_trucks"] = panel["unique_trucks"].fillna(0).astype(int)
    panel["truck_churn_rate"] = panel["truck_churn_rate"].fillna(0.0)
    panel["detention_minutes_sum"] = panel["detention_minutes_sum"].fillna(0.0)
    panel["detention_minutes_avg"] = panel["detention_minutes_avg"].fillna(0.0)

    panel = panel.sort_values(["driver_id", "month"]).reset_index(drop=True)

    panel = add_rolling_and_lifetime_history(panel, windows=cfg.history_windows)
    panel = add_perf_dynamics(panel)

    # Внедряем сезонность
    panel = add_seasonality_features(panel)

    drivers["hire_date"] = pd.to_datetime(drivers["hire_date"], errors="coerce")
    panel = panel.merge(drivers[["driver_id", "hire_date"]], on="driver_id", how="left")

    panel["months_in_company"] = (
            (panel["month"].dt.year - panel["hire_date"].dt.year) * 12 +
            (panel["month"].dt.month - panel["hire_date"].dt.month)
    ).fillna(999).astype(int)
    panel = panel.drop(columns=["hire_date"])

    panel = add_target_next_3m(panel)
    max_month = panel["month"].max()
    panel = panel[panel["month"] <= (max_month - pd.offsets.MonthBegin(cfg.horizon_months))].copy()

    target = "has_incident_next_3m"

    feature_cols = (
            perf_cols
            + [c for c in panel.columns if c.startswith("incidents_prev_")]
            + [c for c in panel.columns if c.startswith("trips_prev_")]
            + [c for c in panel.columns if c.startswith("incident_rate_prev_")]
            + [
                "lifetime_trips_cnt", "lifetime_incidents_cnt", "lifetime_incident_rate",
                "months_since_last_incident", "months_in_company",
                "unique_trucks", "truck_churn_rate",
                "detention_minutes_sum", "detention_minutes_avg",
                "mpg_drop_ratio", "idle_spike_ratio", "ontime_drop_ratio",
                "month_sin", "month_cos", "quarter", "is_high_season"
            ]
    )
    feature_cols = [c for c in feature_cols if c in panel.columns]

    panel[feature_cols] = panel[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return panel, feature_cols, target