import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


# -------------------------
# Config
# -------------------------
ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
OUT_DIR = ROOT / "stage_1" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MONTH_COL = "month"


@dataclass
class SplitConfig:
    test_months: int = 3
    horizon_months: int = 6


# -------------------------
# Utils / metrics
# -------------------------
def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    k = min(k, len(y_true))
    if y_true.sum() == 0:
        return 0.0
    idx = np.argsort(scores)[::-1][:k]
    return float(y_true[idx].sum() / y_true.sum())

def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    k = min(k, len(y_true))
    idx = np.argsort(scores)[::-1][:k]
    return float(y_true[idx].mean())

def lift_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    base = float(np.mean(y_true))
    if base == 0:
        return np.nan
    return precision_at_k(y_true, scores, k) / base

def safe_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def to_month_start(dt: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp()

def evaluate(y_true: np.ndarray, scores: np.ndarray, tag: str, ks=(10, 20, 30, 50, 100, 200)):
    out = {}
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, scores)
        out["ap"] = average_precision_score(y_true, scores)
    else:
        out["roc_auc"] = np.nan
        out["ap"] = np.nan
    out["positives"] = int(y_true.sum())
    out["n"] = int(len(y_true))
    out["base_rate"] = float(np.mean(y_true))
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(y_true, scores, k)
        out[f"precision@{k}"] = precision_at_k(y_true, scores, k)
        out[f"lift@{k}"] = lift_at_k(y_true, scores, k)
    return pd.Series(out, name=tag)

def train_test_split_by_time(df: pd.DataFrame, test_months: int = 6):
    months = np.sort(df["month"].unique())
    if len(months) <= test_months + 3:
        raise ValueError("Not enough months to create a stable train/test split.")
    test_start = pd.to_datetime(months[-test_months])
    train = df[df["month"] < test_start].copy()
    test = df[df["month"] >= test_start].copy()
    return train, test, test_start


# -------------------------
# Data prep
# -------------------------
def build_driver_month_incidents(trips: pd.DataFrame, safety_incidents: pd.DataFrame) -> pd.DataFrame:
    trips = trips.copy()
    trips["dispatch_date"] = pd.to_datetime(trips["dispatch_date"], errors="coerce")
    trips = trips.dropna(subset=["dispatch_date", "driver_id", "trip_id"])
    trips["month"] = to_month_start(trips["dispatch_date"])

    trip_has_inc = trips[["trip_id"]].copy()
    trip_has_inc["has_incident"] = trip_has_inc["trip_id"].isin(safety_incidents["trip_id"]).astype(int)

    t = trips.merge(trip_has_inc, on="trip_id", how="left")
    t["has_incident"] = t["has_incident"].fillna(0).astype(int)

    agg = t.groupby(["driver_id", "month"], as_index=False).agg(
        trips_cnt=("trip_id", "count"),
        incidents_cnt=("has_incident", "sum"),
    )
    return agg

def add_rolling_history(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["driver_id", "month"]).copy()

    for w in [3, 6, 12]:
        df[f"incidents_prev_{w}m"] = (
            df.groupby("driver_id")["incidents_cnt"]
            .shift(1)
            .rolling(window=w, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        df[f"trips_prev_{w}m"] = (
            df.groupby("driver_id")["trips_cnt"]
            .shift(1)
            .rolling(window=w, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        df[f"incident_rate_prev_{w}m"] = df[f"incidents_prev_{w}m"] / df[f"trips_prev_{w}m"].replace(0, np.nan)

    def months_since_last_incident(g: pd.DataFrame) -> pd.Series:
        has = (g["incidents_cnt"] > 0).astype(int).values
        last = np.where(has == 1, np.arange(len(has)), -1)
        last = np.maximum.accumulate(last)
        last = np.roll(last, 1)
        last[0] = -1
        dist = np.arange(len(has)) - last
        dist[last < 0] = 999
        return pd.Series(dist, index=g.index)

    df["months_since_last_incident"] = df.groupby("driver_id", group_keys=False).apply(months_since_last_incident)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

def add_target_next_3m(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["driver_id", "month"]).copy()

    def future_any(g: pd.DataFrame) -> pd.Series:
        fut = g["incidents_cnt"].shift(-1).rolling(window=3, min_periods=1).sum()
        return (fut > 0).astype(int)

    df["has_incident_next_3m"] = df.groupby("driver_id", group_keys=False).apply(future_any)
    return df


# -------------------------
# Main
# -------------------------
def main() -> None:
    cfg = SplitConfig(test_months=6, horizon_months=3)

    trips_path = DATASETS / "trips.csv"
    inc_path = DATASETS / "safety_incidents.csv"
    metrics_path = DATASETS / "driver_monthly_metrics.csv"

    trips = pd.read_csv(trips_path)
    safety_incidents = pd.read_csv(inc_path)
    driver_monthly_metrics = pd.read_csv(metrics_path)

    if MONTH_COL not in driver_monthly_metrics.columns:
        raise KeyError(
            f"MONTH_COL='{MONTH_COL}' not found in driver_monthly_metrics.csv. "
            f"Available columns: {list(driver_monthly_metrics.columns)[:30]}..."
        )

    driver_monthly_metrics = driver_monthly_metrics.copy()
    driver_monthly_metrics["month"] = to_month_start(driver_monthly_metrics[MONTH_COL])

    id_cols = {"driver_id", "month"}
    perf_cols = [c for c in driver_monthly_metrics.columns
                 if c not in id_cols and pd.api.types.is_numeric_dtype(driver_monthly_metrics[c])]

    perf = driver_monthly_metrics[["driver_id", "month"] + perf_cols].copy()

    inc_panel = build_driver_month_incidents(trips, safety_incidents)

    panel = perf.merge(inc_panel, on=["driver_id", "month"], how="left")
    panel["trips_cnt"] = panel["trips_cnt"].fillna(0).astype(int)
    panel["incidents_cnt"] = panel["incidents_cnt"].fillna(0).astype(int)

    panel = panel.sort_values(["driver_id", "month"]).reset_index(drop=True)
    panel = add_rolling_history(panel)
    panel = add_target_next_3m(panel)

    # Drop last horizon months
    max_month = panel["month"].max()
    panel = panel[panel["month"] <= (max_month - pd.offsets.MonthBegin(cfg.horizon_months))].copy()

    train, test, test_start = train_test_split_by_time(panel, test_months=cfg.test_months)

    print("Panel rows:", len(panel))
    print("Months range:", panel["month"].min().date(), "->", panel["month"].max().date())
    print("Test starts at:", pd.to_datetime(test_start).date())
    print("Train rows:", len(train), "Test rows:", len(test))
    print("Target rate (train):", f"{train['has_incident_next_3m'].mean():.2%}")
    print("Target rate (test) :", f"{test['has_incident_next_3m'].mean():.2%}")

    target = "has_incident_next_3m"
    feature_cols = perf_cols + \
        [c for c in panel.columns if c.startswith("incidents_prev_")] + \
        [c for c in panel.columns if c.startswith("trips_prev_")] + \
        [c for c in panel.columns if c.startswith("incident_rate_prev_")] + \
        ["months_since_last_incident"]

    feature_cols = [c for c in feature_cols if c in panel.columns]

    X_train = safe_numeric_frame(train[feature_cols])
    y_train = train[target].astype(int).to_numpy()
    X_test = safe_numeric_frame(test[feature_cols])
    y_test = test[target].astype(int).to_numpy()

    results = {}
    scored_test = test[["driver_id", "month", target]].copy()

    dm = DummyClassifier(strategy="most_frequent", random_state=42)
    dm.fit(X_train, y_train)
    results["dummy_most_frequent"] = evaluate(y_test, dm.predict_proba(X_test)[:, 1], "dummy_most_frequent")

    ds = DummyClassifier(strategy="stratified", random_state=42)
    ds.fit(X_train, y_train)
    results["dummy_stratified"] = evaluate(y_test, ds.predict_proba(X_test)[:, 1], "dummy_stratified")

    lr = LogisticRegression(max_iter=5000, class_weight="balanced")
    lr.fit(X_train, y_train)
    p_lr = lr.predict_proba(X_test)[:, 1]
    results["logreg_balanced"] = evaluate(y_test, p_lr, "logreg_balanced")

    p_xgb = None
    xgb_model = None
    try:
        from xgboost import XGBClassifier

        pos = y_train.sum()
        neg = len(y_train) - pos
        spw = neg / max(1, pos)

        xgb_model = XGBClassifier(
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
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        p_xgb = xgb_model.predict_proba(X_test)[:, 1]
        results["xgb"] = evaluate(y_test, p_xgb, "xgb")

        try:
            booster = xgb_model.get_booster()
            score_gain = booster.get_score(importance_type="gain")
            fi = pd.DataFrame(
                [{"feature": k, "gain": v} for k, v in score_gain.items()]
            ).sort_values("gain", ascending=False)
            if fi["feature"].str.fullmatch(r"f\d+").all():
                fi["feature_idx"] = fi["feature"].str.replace("f", "", regex=False).astype(int)
                fi["feature_name"] = fi["feature_idx"].map(lambda i: feature_cols[i] if i < len(feature_cols) else f"f{i}")
            else:
                fi["feature_name"] = fi["feature"]
            fi[["feature_name", "gain"]].to_csv(OUT_DIR / "xgb_feature_importance_gain.csv", index=False)
            print("\nSaved: xgb_feature_importance_gain.csv")
        except Exception as e:
            print("Could not extract XGB feature importance:", e)

    except Exception as e:
        print("XGBoost not available or failed:", e)

    if p_xgb is not None:
        scored_test["risk_score"] = p_xgb
        best_name = "xgb"
    else:
        scored_test["risk_score"] = p_lr
        best_name = "logreg_balanced"

    months_in_test = np.sort(scored_test["month"].unique())

    if len(months_in_test) < 2:
        display_m = months_in_test[-1]
        print("Only one month in test; displaying:", pd.to_datetime(display_m).date())
    else:
        display_m = months_in_test[-2]
        print("Displaying penultimate test month:", pd.to_datetime(display_m).date())

    top_month = (
        scored_test[scored_test["month"] == display_m]
        .sort_values("risk_score", ascending=False)
        .head(30)
    )
    print(f"\nTop-30 risky driver-months (display month = {pd.to_datetime(display_m).date()}) [{best_name}]:")
    print(top_month.to_string(index=False))

    per_driver = scored_test.groupby("driver_id", as_index=False).agg(
        max_risk=("risk_score", "max"),
        any_positive_in_test=(target, "max"),
        n_months=("month", "nunique"),
    ).sort_values("max_risk", ascending=False)

    print(f"\nTop-30 drivers by MAX risk across test window [{best_name}]:")
    print(per_driver.head(30).to_string(index=False))

    per_driver.to_csv(OUT_DIR / "test_drivers_ranked_by_risk.csv", index=False)
    scored_test.to_csv(OUT_DIR / "test_driver_month_scored.csv", index=False)
    print("\nSaved: test_drivers_ranked_by_risk.csv, test_driver_month_scored.csv")

    metrics_df = pd.DataFrame(results).T
    print("\n=== DRIVER-LEVEL FORECAST (next 3 months) ===")
    print(metrics_df.sort_values(["ap", "roc_auc"], ascending=False).to_string())

    metrics_df.to_csv(OUT_DIR / "driver_risk_forecast_metrics_v2.csv", index=True)
    print("\nSaved: driver_risk_forecast_metrics_v2.csv")


if __name__ == "__main__":
    main()

# посмотреть другие модели, сделать модель с вероятностью, сделать сайт, объяснимый ии?