from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import clickhouse_connect
import os


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"


@dataclass(frozen=True)
class DatasetConfig:
    test_months: int = 6
    horizon_months: int = 3
    history_windows: Sequence[int] = (3, 6, 12)


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



def build_panel(cfg: DatasetConfig = DatasetConfig()) -> tuple[pd.DataFrame, list[str], str]:
    print("Запрашиваем готовую витрину из ClickHouse...")

    ch_host = os.getenv('CLICKHOUSE_HOST', '127.0.0.1')
    client = clickhouse_connect.get_client(host=ch_host, port=8123, username='admin', password='admin')

    panel = client.query_df("SELECT * FROM default.driver_features_view")

    # 2. Приводим типы
    panel['month'] = pd.to_datetime(panel['month'])

    # 3. Обрезаем горизонт предсказания (убираем последние 3 месяца, где мы еще не знаем будущее)
    max_month = panel["month"].max()
    panel = panel[panel["month"] <= (max_month - pd.offsets.MonthBegin(cfg.horizon_months))].copy()

    target = "has_incident_next_3m"

    exclude = {"driver_id", "month", target, "trips_cnt", "incidents_cnt"}
    feature_cols = [c for c in panel.columns if c not in exclude]

    panel[feature_cols] = panel[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"Матрица получена: {panel.shape[0]} строк, {len(feature_cols)} фичей.")
    return panel, feature_cols, target