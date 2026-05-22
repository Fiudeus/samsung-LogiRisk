from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier

from stage_2.data import (
    ROOT,
    DatasetConfig,
    build_panel,
    safe_numeric_frame,
    train_test_split_by_time,
)
from stage_2.metrics.driver_month_metrics import evaluate_driver_month
from stage_2.metrics.driver_metrics import (
    evaluate_driver_many_aggs_from_driver_month_frame,
    print_driver_agg_report,
)

warnings.filterwarnings("ignore")

OUT_DIR = ROOT / "stage_2" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cfg = DatasetConfig(test_months=6, horizon_months=3)

    panel, feature_cols, target = build_panel(cfg)
    train, test, test_start = train_test_split_by_time(panel, test_months=cfg.test_months)

    print("=== DATA ===")
    print("Panel rows:", len(panel))
    print("Months range:", panel["month"].min().date(), "->", panel["month"].max().date())
    print("Test starts at:", pd.to_datetime(test_start).date())
    print("Train rows:", len(train), "Test rows:", len(test))
    print("Target rate (train):", f"{train[target].mean():.2%}")
    print("Target rate (test) :", f"{test[target].mean():.2%}")
    print("n_features:", len(feature_cols))

    X_train = safe_numeric_frame(train[feature_cols])
    y_train = train[target].astype(int).to_numpy()
    X_test = safe_numeric_frame(test[feature_cols])
    y_test = test[target].astype(int).to_numpy()

    model = ExtraTreesClassifier(
        n_estimators=1500,
        max_depth=None,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features="sqrt",
        bootstrap=False,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("\n=== TRAIN ===")
    print(
        "ExtraTrees params:",
        {
            "n_estimators": 1500,
            "min_samples_leaf": 5,
            "min_samples_split": 10,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },
    )

    model.fit(X_train, y_train)

    scores = model.predict_proba(X_test)[:, 1]

    rep_dm = evaluate_driver_month(
        y_test,
        scores,
        tag="extratrees_driver_month",
        ks=(10, 20, 30, 50, 100, 200),
    )

    print("\n=== METRICS: DRIVER-MONTH ===")
    print(rep_dm.to_string())

    scored_dm = test[["driver_id", "month", target]].copy()
    scored_dm["score"] = scores
    scored_dm.rename(columns={target: "y_true"}, inplace=True)

    metrics_aggs, ranked_by_agg = evaluate_driver_many_aggs_from_driver_month_frame(
        scored_dm,
        ks=(5, 10, 20, 30),
        q=0.2,
        tag_prefix="extratrees_driver",
    )

    print("\n=== METRICS: DRIVER (all aggs) ===")
    print_driver_agg_report(metrics_aggs)

    # save artifacts
    scored_dm_out = OUT_DIR / "extratrees_driver_month_scored.csv"
    scored_dm.to_csv(scored_dm_out, index=False)
    print("\nSaved:", scored_dm_out)

    for agg_name, df_drv in ranked_by_agg.items():
        out = OUT_DIR / f"extratrees_drivers_ranked_{agg_name}.csv"
        df_drv.sort_values("score", ascending=False).to_csv(out, index=False)
        print("Saved:", out)

    # feature importance
    try:
        fi = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values(
            "importance", ascending=False
        )
        fi_out = OUT_DIR / "extratrees_feature_importance.csv"
        fi.to_csv(fi_out, index=False)
        print("Saved:", fi_out)
    except Exception as e:
        print("Could not extract ExtraTrees feature importance:", e)


if __name__ == "__main__":
    main()