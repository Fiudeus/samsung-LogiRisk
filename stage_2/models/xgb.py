from __future__ import annotations

import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from stage_2.data import (
    DATASETS,
    ROOT,
    DatasetConfig,
    build_panel,
    safe_numeric_frame,
    train_test_split_by_time,
)
from stage_2.metrics.driver_month_metrics import evaluate_driver_month
from stage_2.metrics.driver_metrics import (
    DriverAggregationConfig,
    evaluate_driver_from_driver_month_frame,
    evaluate_driver_many_aggs_from_driver_month_frame,
    print_driver_agg_report
)

warnings.filterwarnings("ignore")


OUT_DIR = ROOT / "stage_2" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # Конфиг оставляем как было в 06_... (6 месяцев тест, горизонт 3 месяца)
    cfg = DatasetConfig(test_months=6, horizon_months=3)

    panel, feature_cols, target = build_panel(cfg)

    # split
    train, test, test_start = train_test_split_by_time(panel, test_months=cfg.test_months)

    print("=== DATA ===")
    print("Panel rows:", len(panel))
    print("Months range:", panel["month"].min().date(), "->", panel["month"].max().date())
    print("Test starts at:", pd.to_datetime(test_start).date())
    print("Train rows:", len(train), "Test rows:", len(test))
    print("Target rate (train):", f"{train[target].mean():.2%}")
    print("Target rate (test) :", f"{test[target].mean():.2%}")
    print("n_features:", len(feature_cols))

    # matrices
    X_train = safe_numeric_frame(train[feature_cols])
    y_train = train[target].astype(int).to_numpy()
    X_test = safe_numeric_frame(test[feature_cols])
    y_test = test[target].astype(int).to_numpy()

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    spw = neg / max(1, pos)

    model = XGBClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=5,
        gamma=0.0,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    print("\n=== TRAIN ===")
    print("XGB params:", {**model.get_params(), "scale_pos_weight": spw})
    model.fit(X_train, y_train)

    scores = model.predict_proba(X_test)[:, 1]

    # --- driver-month metrics
    rep_dm = evaluate_driver_month(
        y_test,
        scores,
        tag="xgb_driver_month",
        ks=(10, 20, 30, 50, 100, 200),
    )

    print("\n=== METRICS: DRIVER-MONTH ===")
    print(rep_dm.to_string())

    # Prepare scored frame for saving + driver-level aggregation
    scored_dm = test[["driver_id", "month", target]].copy()
    scored_dm["score"] = scores  # score = proba класса 1
    scored_dm.rename(columns={target: "y_true"}, inplace=True)


    metrics_aggs, ranked_by_agg = evaluate_driver_many_aggs_from_driver_month_frame(
        scored_dm,
        ks=(5, 10, 20, 30),
        q=0.2,
        tag_prefix="xgb_driver",  # или logreg_driver
    )

    print("\n=== METRICS: DRIVER (all aggs) ===")
    print_driver_agg_report(metrics_aggs)

    # --- driver-level metrics (agg=max by default)
    rep_drv, drivers_ranked = evaluate_driver_from_driver_month_frame(
        scored_dm,
        cfg=DriverAggregationConfig(
            driver_col="driver_id",
            month_col="month",
            y_col="y_true",
            score_col="score",
            agg="max",
        ),
        tag="xgb_driver_max",
        ks=(5, 10, 20, 30),
    )
    drivers_ranked = drivers_ranked.sort_values("score", ascending=False)

    print("\n=== METRICS: DRIVER (agg=max over test window) ===")
    print(rep_drv.to_string())

    # Save artifacts
    scored_dm_out = OUT_DIR / "xgb_driver_month_scored.csv"
    drivers_out = OUT_DIR / "xgb_drivers_ranked.csv"
    scored_dm.to_csv(scored_dm_out, index=False)
    drivers_ranked.to_csv(drivers_out, index=False)
    print("\nSaved:", scored_dm_out)
    print("Saved:", drivers_out)

    # Feature importance (gain)
    try:
        booster = model.get_booster()
        score_gain = booster.get_score(importance_type="gain")
        fi = pd.DataFrame([{"feature": k, "gain": v} for k, v in score_gain.items()]).sort_values(
            "gain", ascending=False
        )

        # xgboost иногда возвращает f0,f1,... поэтому маппим на названия фичей
        if not fi.empty and fi["feature"].str.fullmatch(r"f\d+").all():
            fi["feature_idx"] = fi["feature"].str.replace("f", "", regex=False).astype(int)
            fi["feature_name"] = fi["feature_idx"].map(
                lambda i: feature_cols[i] if i < len(feature_cols) else f"f{i}"
            )
        else:
            fi["feature_name"] = fi["feature"]

        fi_out = OUT_DIR / "xgb_feature_importance_gain.csv"
        fi[["feature_name", "gain"]].to_csv(fi_out, index=False)
        print("Saved:", fi_out)
    except Exception as e:
        print("Could not extract XGB feature importance:", e)


if __name__ == "__main__":
    main()