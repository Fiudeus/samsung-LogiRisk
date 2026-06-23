from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier

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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def _get_scores(model, X: pd.DataFrame) -> np.ndarray:
    # Prefer probabilities if available
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
    # Fallback: decision_function -> sigmoid
    if hasattr(model, "decision_function"):
        return _sigmoid(model.decision_function(X))
    # Last resort: predict labels (not great, but avoids crash)
    return model.predict(X).astype(float)


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

    # Class weights: приблизительный аналог scale_pos_weight
    # sklearn HGB принимает class_weight начиная с некоторых версий;
    # если не поддерживается — просто обучится без него.
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    spw = neg / max(1, pos)

    class_weight = {0: 1.0, 1: float(spw)}

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_depth=6,
        max_iter=600,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42,
        # early_stopping=True по умолчанию, но пусть будет явно:
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
    )

    print("\n=== TRAIN ===")
    print(
        "HGB params:",
        {
            "learning_rate": 0.05,
            "max_depth": 6,
            "max_iter": 600,
            "min_samples_leaf": 20,
            "early_stopping": True,
        },
    )

    # Поддержка class_weight зависит от версии sklearn
    try:
        model.set_params(class_weight=class_weight)
        print("Using class_weight:", class_weight)
    except Exception:
        print("class_weight is not supported in this sklearn version; training without it.")

    model.fit(X_train, y_train)

    scores = _get_scores(model, X_test)

    rep_dm = evaluate_driver_month(
        y_test,
        scores,
        tag="hgb_driver_month",
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
        tag_prefix="hgb_driver",
    )

    print("\n=== METRICS: DRIVER (all aggs) ===")
    print_driver_agg_report(metrics_aggs)

    scored_dm_out = OUT_DIR / "hgb_driver_month_scored.csv"
    scored_dm.to_csv(scored_dm_out, index=False)
    print("\nSaved:", scored_dm_out)

    for agg_name, df_drv in ranked_by_agg.items():
        out = OUT_DIR / f"hgb_drivers_ranked_{agg_name}.csv"
        df_drv.sort_values("score", ascending=False).to_csv(out, index=False)
        print("Saved:", out)

    # Feature importance: у HGB есть feature_importances_ не всегда;
    # попробуем аккуратно.
    try:
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values(
                "importance", ascending=False
            )
            fi_out = OUT_DIR / "hgb_feature_importance.csv"
            fi.to_csv(fi_out, index=False)
            print("Saved:", fi_out)
    except Exception as e:
        print("Could not extract HGB feature importance:", e)


if __name__ == "__main__":
    main()