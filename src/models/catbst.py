from __future__ import annotations

import warnings

import pandas as pd

from catboost import CatBoostClassifier

from src.data import (
    ROOT,
    DatasetConfig,
    build_panel,
    safe_numeric_frame,
    train_test_split_by_time,
)
from src.metrics.driver_month_metrics import evaluate_driver_month
from src.metrics.driver_metrics import (
    evaluate_driver_many_aggs_from_driver_month_frame,
    print_driver_agg_report,
)

warnings.filterwarnings("ignore")

OUT_DIR = ROOT / "src" / "output"
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

    # Балансировка как в XGB: scale_pos_weight = neg/pos
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    spw = neg / max(1, pos)

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=200,
        scale_pos_weight=spw,
        allow_writing_files=False,
    )

    print("\n=== TRAIN ===")
    print(
        "CatBoost params:",
        {
            "iterations": 2000,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "scale_pos_weight": spw,
        },
    )

    model.fit(X_train, y_train)

    scores = model.predict_proba(X_test)[:, 1]

    # --- driver-month metrics (как в xgb.py)
    rep_dm = evaluate_driver_month(
        y_test,
        scores,
        tag="catboost_driver_month",
        ks=(10, 20, 30, 50, 100, 200),
    )

    print("\n=== METRICS: DRIVER-MONTH ===")
    print(rep_dm.to_string())

    # --- driver metrics (all aggs) (как в xgb.py после рефактора)
    scored_dm = test[["driver_id", "month", target]].copy()
    scored_dm["score"] = scores
    scored_dm.rename(columns={target: "y_true"}, inplace=True)

    metrics_aggs, ranked_by_agg = evaluate_driver_many_aggs_from_driver_month_frame(
        scored_dm,
        ks=(5, 10, 20, 30),  # ваш “бюджет” K=30 тут уже включён
        q=0.2,
        tag_prefix="catboost_driver",
    )

    print("\n=== METRICS: DRIVER (all aggs) ===")
    print_driver_agg_report(metrics_aggs)

    # --- save artifacts
    scored_dm_out = OUT_DIR / "catboost_driver_month_scored.csv"
    scored_dm.to_csv(scored_dm_out, index=False)
    print("\nSaved:", scored_dm_out)

    # Сохраним рейтинги по водителям для каждой агрегации (удобно смотреть топ-30)
    for agg_name, df_drv in ranked_by_agg.items():
        out = OUT_DIR / f"catboost_drivers_ranked_{agg_name}.csv"
        df_drv.sort_values("score", ascending=False).to_csv(out, index=False)
        print("Saved:", out)

    # feature importance
    try:
        fi = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model.get_feature_importance(type="FeatureImportance"),
            }
        ).sort_values("importance", ascending=False)

        fi_out = OUT_DIR / "catboost_feature_importance.csv"
        fi.to_csv(fi_out, index=False)
        print("Saved:", fi_out)
    except Exception as e:
        print("Could not extract CatBoost feature importance:", e)


if __name__ == "__main__":
    main()