from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetConfig:
    test_months: int = 6
    horizon_months: int = 3
    history_windows: Sequence[int] = (3, 6, 12)


def safe_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().replace([np.inf, -np.inf], np.nan)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def train_test_split_by_time(df: pd.DataFrame, test_months: int = 6):
    months = np.sort(df["month"].unique())
    if len(months) <= test_months + 3:
        raise ValueError("Not enough months to create a stable train/test split.")
    test_start = pd.to_datetime(months[-test_months])
    train = df[df["month"] < test_start].copy()
    test = df[df["month"] >= test_start].copy()
    return train, test, test_start


def build_panel(cfg: DatasetConfig = DatasetConfig()) -> tuple[pd.DataFrame, list[str], str]:
    print("Загружаем готовую Spark-матрицу из Parquet...")
    parquet_path = ROOT / "src" / "output" / "spark_features.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Spark-матрица фичей не обнаружена по пути {parquet_path}. Сначала запустите src.spark_features.")

    panel = pd.read_parquet(parquet_path)
    panel['month'] = pd.to_datetime(panel['month'])

    target = "has_incident_next_3m"
    # Исключаем служебные поля, идентификаторы и прямые лики
    exclude = {"driver_id", "month", target, "trips_cnt", "incidents_cnt", "month_idx", "incident_marker",
               "last_incident_month_idx"}
    feature_cols = [c for c in panel.columns if c not in exclude]

    # Строгая типизация
    for c in feature_cols:
        if any(tp in str(panel[c].dtype).lower() for tp in ['uint', 'int']):
            panel[c] = panel[c].astype('float64')

    print(f"Матрица успешно собрана: {panel.shape[0]} строк, {len(feature_cols)} фичей.")
    return panel, feature_cols, target