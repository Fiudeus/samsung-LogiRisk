from __future__ import annotations

import warnings

import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.data import (
    ROOT,
    DatasetConfig,
    build_panel,
    safe_numeric_frame,
    train_test_split_by_time,
)
from src.metrics.driver_month_metrics import evaluate_driver_month
from src.metrics.driver_metrics import (
    DriverAggregationConfig,
    evaluate_driver_from_driver_month_frame,
    evaluate_driver_many_aggs_from_driver_month_frame,
    print_driver_agg_report
)

warnings.filterwarnings("ignore")


OUT_DIR = ROOT / "src" / "output"
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

    # === ФИКС 1: Магия XGBoost с отсутствующими данными ===
    # Значение 999 сводит бустинги с ума, они тратят сплиты на отделение 999 от 10.
    # XGBoost из коробки гениально работает с NaN, сам вычисляя, куда их отправить
    # (в безопасную ветку). Мы возвращаем NaN на место.
    if "months_since_last_incident" in X_train.columns:
        X_train["months_since_last_incident"] = X_train["months_since_last_incident"].replace(999.0, np.nan)
        X_test["months_since_last_incident"] = X_test["months_since_last_incident"].replace(999.0, np.nan)

    if "months_in_company" in X_train.columns:
        X_train["months_in_company"] = X_train["months_in_company"].replace(999.0, np.nan)
        X_test["months_in_company"] = X_test["months_in_company"].replace(999.0, np.nan)

    model = XGBClassifier(
        n_estimators=500,  # Больше деревьев...
        learning_rate=0.005,  # ...но учимся очень медленно
        max_depth=2,
        subsample=0.5,
        colsample_bytree=0.5,
        reg_lambda=20.0,
        scale_pos_weight=1.0,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    print("\n=== TRAIN ===")
    print("XGB params:", {**model.get_params()})

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