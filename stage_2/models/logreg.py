from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from stage_2.data import (
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

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=None,  # lbfgs ignores n_jobs
    )

    print("\n=== TRAIN ===")
    print("LogReg params:", model.get_params())
    model.fit(X_train, y_train)

    scores = model.predict_proba(X_test)[:, 1]

    rep_dm = evaluate_driver_month(
        y_test,
        scores,
        tag="logreg_driver_month",
        ks=(10, 20, 30, 50, 100, 200),
    )

    print("\n=== METRICS: DRIVER-MONTH ===")
    print(rep_dm.to_string())

    scored_dm = test[["driver_id", "month", target]].copy()
    scored_dm["score"] = scores
    scored_dm.rename(columns={target: "y_true"}, inplace=True)

    rep_drv, drivers_ranked = evaluate_driver_from_driver_month_frame(
        scored_dm,
        cfg=DriverAggregationConfig(
            driver_col="driver_id",
            month_col="month",
            y_col="y_true",
            score_col="score",
            agg="max",
        ),
        tag="logreg_driver_max",
        ks=(5, 10, 20, 30),
    )
    drivers_ranked = drivers_ranked.sort_values("score", ascending=False)

    print("\n=== METRICS: DRIVER (agg=max over test window) ===")
    print(rep_drv.to_string())

    scored_dm_out = OUT_DIR / "logreg_driver_month_scored.csv"
    drivers_out = OUT_DIR / "logreg_drivers_ranked.csv"
    scored_dm.to_csv(scored_dm_out, index=False)
    drivers_ranked.to_csv(drivers_out, index=False)

    print("\nSaved:", scored_dm_out)
    print("Saved:", drivers_out)

    # "Feature importance" for logreg: coefficients
    try:
        coefs = pd.DataFrame(
            {
                "feature": feature_cols,
                "coef": model.coef_.ravel().astype(float),
                "abs_coef": np.abs(model.coef_.ravel().astype(float)),
            }
        ).sort_values("abs_coef", ascending=False)

        coef_out = OUT_DIR / "logreg_coefficients.csv"
        coefs[["feature", "coef", "abs_coef"]].to_csv(coef_out, index=False)
        print("Saved:", coef_out)
    except Exception as e:
        print("Could not save logreg coefficients:", e)


if __name__ == "__main__":
    main()