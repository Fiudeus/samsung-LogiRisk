from __future__ import annotations

import warnings

import pandas as pd

from imblearn.ensemble import EasyEnsembleClassifier

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

    # EasyEnsemble: ансамбль бустингов на сбалансированных (undersampled) подвыборках.
    # Полезен, когда хотим не пропускать редкий класс (минимизировать FN при фиксированном K).
    model = EasyEnsembleClassifier(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
        # По умолчанию base_estimator = AdaBoostClassifier, что ок для старта.
    )

    print("\n=== TRAIN ===")
    print(
        "EasyEnsemble params:",
        {
            "n_estimators": 250,
        },
    )

    model.fit(X_train, y_train)

    # В imblearn predict_proba обычно есть
    scores = model.predict_proba(X_test)[:, 1]

    rep_dm = evaluate_driver_month(
        y_test,
        scores,
        tag="easyensemble_driver_month",
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
        tag_prefix="easyensemble_driver",
    )

    print("\n=== METRICS: DRIVER (all aggs) ===")
    print_driver_agg_report(metrics_aggs)

    scored_dm_out = OUT_DIR / "easyensemble_driver_month_scored.csv"
    scored_dm.to_csv(scored_dm_out, index=False)
    print("\nSaved:", scored_dm_out)

    for agg_name, df_drv in ranked_by_agg.items():
        out = OUT_DIR / f"easyensemble_drivers_ranked_{agg_name}.csv"
        df_drv.sort_values("score", ascending=False).to_csv(out, index=False)
        print("Saved:", out)

    # Feature importance: у EasyEnsemble её “одной” нет (это ансамбль ансамблей),
    # поэтому здесь ничего не сохраняем.


if __name__ == "__main__":
    main()