from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score

from stage_2.data import (
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
    print_driver_agg_report,
)

warnings.filterwarnings("ignore")


# -------------------------
# Models used for checks
# -------------------------
def train_xgb_scores(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> np.ndarray:
    """Train the same-ish XGB as in stage_2/models/xgb.py and return test scores."""
    from xgboost import XGBClassifier

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
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


# -------------------------
# Sanity checks
# -------------------------
def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_train_test_key_leak(train: pd.DataFrame, test: pd.DataFrame) -> None:
    _print_header("CHECK 1: Train/Test key intersection (driver_id, month)")

    train_keys = set(map(tuple, train[["driver_id", "month"]].itertuples(index=False, name=None)))
    test_keys = set(map(tuple, test[["driver_id", "month"]].itertuples(index=False, name=None)))
    inter = train_keys & test_keys

    print("Train unique keys:", len(train_keys))
    print("Test unique keys :", len(test_keys))
    print("Intersection     :", len(inter))
    if inter:
        sample = list(inter)[:10]
        print("Sample intersecting keys (first 10):", sample)
        print(">>> This should be 0. If not, split is broken / data duplicated.")
    else:
        print("OK: no intersection.")


def check_target_alignment(panel: pd.DataFrame, n_drivers: int = 2) -> None:
    _print_header("CHECK 2: Target alignment sample (manual inspection)")

    # Pick drivers with at least 8 months
    counts = panel.groupby("driver_id")["month"].nunique().sort_values(ascending=False)
    driver_ids = list(counts.head(max(n_drivers, 1)).index)

    cols = ["driver_id", "month", "incidents_cnt", "has_incident_next_3m"]
    for d in driver_ids[:n_drivers]:
        print(f"\nDriver {d}:")
        tmp = panel.loc[panel["driver_id"] == d, cols].sort_values("month").copy()
        # Show last 18 rows for easier "future" checking
        print(tmp.tail(18).to_string(index=False))
        print("Tip: verify that has_incident_next_3m at month t reflects incidents in t+1..t+3, not t.")


def driver_month_metrics_on_scores(
    df_test: pd.DataFrame,
    scores: np.ndarray,
    ks: Iterable[int] = (10, 20, 30, 50, 100, 200),
) -> pd.Series:
    y = df_test["has_incident_next_3m"].astype(int).to_numpy()
    return evaluate_driver_month(y, scores, tag="dm", ks=ks)


def driver_metrics_on_scores(
    df_test: pd.DataFrame,
    scores: np.ndarray,
    ks: Iterable[int] = (5, 10, 20, 30),
) -> pd.DataFrame:
    scored_dm = df_test[["driver_id", "month", "has_incident_next_3m"]].copy()
    scored_dm.rename(columns={"has_incident_next_3m": "y_true"}, inplace=True)
    scored_dm["score"] = scores

    metrics_df, _ = evaluate_driver_many_aggs_from_driver_month_frame(
        scored_dm,
        ks=ks,
        q=0.2,
        tag_prefix="xgb_driver",
    )
    return metrics_df


def check_perm_y_within_month(df_test: pd.DataFrame, scores: np.ndarray) -> None:
    _print_header("CHECK 3: Permutation test (shuffle y within each month)")

    work = df_test[["driver_id", "month", "has_incident_next_3m"]].copy()
    work["score"] = scores
    work["month"] = pd.to_datetime(work["month"])

    # ВАЖНО: делаем позиционный индекс 0..n-1, чтобы groups можно было безопасно использовать с iloc
    work = work.reset_index(drop=True)

    # True metrics
    y_true = work["has_incident_next_3m"].astype(int).to_numpy()
    s = work["score"].to_numpy()

    if len(np.unique(y_true)) > 1:
        ap_true = float(average_precision_score(y_true, s))
        auc_true = float(roc_auc_score(y_true, s))
    else:
        ap_true, auc_true = float("nan"), float("nan")

    # Shuffle y within each month (preserve class balance per month)
    rng = np.random.default_rng(42)
    y_perm = work["has_incident_next_3m"].astype(int).to_numpy().copy()

    for _, idx in work.groupby("month").groups.items():
        idx = np.asarray(list(idx), dtype=int)  # now safe: idx are positions 0..n-1
        y_perm[idx] = rng.permutation(y_perm[idx])

    ap_perm = float(average_precision_score(y_perm, s)) if len(np.unique(y_perm)) > 1 else float("nan")
    auc_perm = float(roc_auc_score(y_perm, s)) if len(np.unique(y_perm)) > 1 else float("nan")

    print("True AP / ROC-AUC :", ap_true, "/", auc_true)
    print("Perm AP / ROC-AUC :", ap_perm, "/", auc_perm)
    print("Base rate         :", float(np.mean(y_true)))
    print("Expectation: permuted metrics should drop close to base-rate (AP) and ~0.5 (ROC-AUC).")


def check_single_month_driver_eval(df_test: pd.DataFrame, scores: np.ndarray) -> None:
    _print_header("CHECK 4: Driver-level metrics on SINGLE month only (no 6-month aggregation)")

    work = df_test[["driver_id", "month", "has_incident_next_3m"]].copy()
    work["score"] = scores
    work["month"] = pd.to_datetime(work["month"])

    last_month = work["month"].max()
    one = work[work["month"] == last_month].copy()
    print("Using last test month:", last_month.date(), "rows:", len(one))

    # Now "driver-level" is same as driver-month for that month (1 row per driver expected)
    one.rename(columns={"has_incident_next_3m": "y_true"}, inplace=True)

    rep_drv, drivers_df = evaluate_driver_from_driver_month_frame(
        one,
        cfg=DriverAggregationConfig(
            driver_col="driver_id",
            month_col="month",
            y_col="y_true",
            score_col="score",
            agg="last",  # same thing here
        ),
        tag="single_month",
        ks=(5, 10, 20, 30),
    )
    print(rep_drv.to_string())

    # also report driver-month AP/AUC for this month
    y = one["y_true"].astype(int).to_numpy()
    s = one["score"].to_numpy()
    if len(np.unique(y)) > 1:
        print("Single-month AP:", float(average_precision_score(y, s)))
        print("Single-month ROC-AUC:", float(roc_auc_score(y, s)))
    else:
        print("Single-month AP/ROC-AUC: n/a (only one class present)")


def check_ablation_feature_sets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    _print_header("CHECK 5: Ablation - perf-only vs history-only (XGB)")

    perf_cols = [c for c in feature_cols if not (c.startswith("incidents_prev_") or c.startswith("trips_prev_") or c.startswith("incident_rate_prev_") or c == "months_since_last_incident")]
    hist_cols = [c for c in feature_cols if c not in perf_cols]

    print("perf_cols:", len(perf_cols), perf_cols)
    print("hist_cols:", len(hist_cols), hist_cols)

    y_train = train["has_incident_next_3m"].astype(int).to_numpy()
    y_test = test["has_incident_next_3m"].astype(int).to_numpy()

    def run(cols: list[str], label: str) -> None:
        X_train = safe_numeric_frame(train[cols])
        X_test = safe_numeric_frame(test[cols])
        scores = train_xgb_scores(X_train, y_train, X_test)
        rep = evaluate_driver_month(y_test, scores, tag=label, ks=(10, 20, 30, 50, 100, 200))
        print(f"\n--- {label} ---")
        print(rep[["roc_auc", "ap", "recall@20", "precision@20", "lift@20", "recall@100", "precision@100"]].to_string())

    run(feature_cols, "all_features")
    if perf_cols:
        run(perf_cols, "perf_only")
    if hist_cols:
        run(hist_cols, "history_only")


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

    _print_header("BASELINE: XGB scores on test")
    scores = train_xgb_scores(X_train, y_train, X_test)

    rep_dm = driver_month_metrics_on_scores(test, scores)
    print("\nDriver-month metrics:")
    print(rep_dm.to_string())

    rep_drv = driver_metrics_on_scores(test, scores)
    print("\nDriver metrics (all aggs):")
    print_driver_agg_report(rep_drv)

    check_train_test_key_leak(train, test)
    check_target_alignment(panel, n_drivers=2)
    check_perm_y_within_month(test, scores)
    check_single_month_driver_eval(test, scores)
    check_ablation_feature_sets(train, test, feature_cols)


if __name__ == "__main__":
    main()