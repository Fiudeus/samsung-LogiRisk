from __future__ import annotations

import warnings
import pandas as pd

from src.data import ROOT
from src.metrics.driver_month_metrics import evaluate_driver_month
from src.metrics.driver_metrics import (
    evaluate_driver_many_aggs_from_driver_month_frame,
    print_driver_agg_report,
)

warnings.filterwarnings("ignore")

OUT_DIR = ROOT / "src" / "output"


def main():
    ensemble_file = OUT_DIR / "hybrid_ensemble_scored.csv"
    if not ensemble_file.exists():
        print(f"[ОШИБКА] Файл {ensemble_file} не найден. Сначала запустите src.ensemble")
        return

    # Загружаем результаты работы гибридного ансамбля
    df = pd.read_csv(ensemble_file)

    y_test = df["y_true"].values
    scores = df["ensemble_mean"].values

    # 1. Считаем метрики Driver-Month
    rep_dm = evaluate_driver_month(
        y_test,
        scores,
        tag="ensemble_driver_month",
        ks=(10, 20, 30, 50, 100, 200),
    )
    print("\n=== METRICS: ENSEMBLE DRIVER-MONTH ===")
    print(rep_dm.to_string())

    # 2. Подготавливаем формат под функции расчета агрегаций по водителям
    scored_dm = df[["driver_id", "month", "y_true", "ensemble_mean"]].copy()
    scored_dm.rename(columns={"ensemble_mean": "score"}, inplace=True)

    metrics_aggs, ranked_by_agg = evaluate_driver_many_aggs_from_driver_month_frame(
        scored_dm,
        ks=(20, 30, 50),
        q=0.2,
        tag_prefix="ensemble_driver",
    )

    print("\n=== METRICS: ENSEMBLE DRIVER (all aggs) ===")
    print_driver_agg_report(metrics_aggs)

    # 3. Сохраняем файлы в стандартизированном виде, чтобы summarize_metrics их нашел
    scored_dm_out = OUT_DIR / "ensemble_driver_month_scored.csv"
    scored_dm.to_csv(scored_dm_out, index=False)
    print(f"\nSaved: {scored_dm_out}")

    for agg_name, df_drv in ranked_by_agg.items():
        out = OUT_DIR / f"ensemble_drivers_ranked_{agg_name}.csv"
        df_drv.sort_values("score", ascending=False).to_csv(out, index=False)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()