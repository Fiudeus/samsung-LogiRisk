from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "stage_1" / "output"
OUT_DIR = ROOT / "stage_1" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODE = "pre_trip"

# -------------------------
# Features
# -------------------------
BASE_PRETRIP_FEATURES = [
    "weight_lbs",
    "pieces",
    "typical_distance_miles",
    "typical_transit_days",
    "base_rate_per_mile",
    "years_experience",
    "truck_age",
    "dispatch_month",
    "dispatch_dow",
    "is_weekend",
]

# old/simple history
HIST_COUNTS = [
    "driver_incidents_before_trip",
    "truck_incidents_before_trip",
]
HIST_EXPOSURE = [
    "driver_trips_before_trip",
    "truck_trips_before_trip",
]
HIST_RATES = [
    "driver_incident_rate_before_trip",
    "truck_incident_rate_before_trip",
]
HIST_RATES_SMOOTHED = [
    "driver_incident_rate_smoothed_before_trip",
    "truck_incident_rate_smoothed_before_trip",
]

# new/stronger history
HIST_RECENCY = [
    "driver_days_since_last_incident",
    "truck_days_since_last_incident",
]

HIST_WINDOWS_CORE = [
    "driver_incidents_last_30d",
    "driver_incidents_last_90d",
    "driver_incidents_last_180d",
    "driver_incidents_last_365d",
    "truck_incidents_last_30d",
    "truck_incidents_last_90d",
    "truck_incidents_last_180d",
    "truck_incidents_last_365d",
]

HIST_WINDOWS_SAFETY_FLAGS = [
    "driver_at_fault_incidents_last_365d",
    "driver_preventable_incidents_last_365d",
    "driver_injury_incidents_last_365d",
    "truck_at_fault_incidents_last_365d",
    "truck_preventable_incidents_last_365d",
    "truck_injury_incidents_last_365d",
]

HIST_TYPE_DIVERSITY = [
    "driver_incident_type_nunique_before_trip",
    "truck_incident_type_nunique_before_trip",
]

HIST_SEVERITY_COSTS = [
    "driver_claim_amount_sum_before_trip",
    "truck_claim_amount_sum_before_trip",
    "driver_damage_cost_sum_before_trip",
    "truck_damage_cost_sum_before_trip",
]


# -------------------------
# Utils
# -------------------------
def recall_at_k(y_true, y_score, k: int) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    k = min(int(k), len(y_true))
    if y_true.sum() == 0:
        return 0.0
    idx = np.argsort(-y_score)[:k]
    return float(y_true[idx].sum() / y_true.sum())


def time_split_idx(df: pd.DataFrame, frac_train: float = 0.8):
    df_sorted = df.sort_values("dispatch_date").reset_index(drop=True)
    cutoff = int(len(df_sorted) * float(frac_train))
    train_idx = np.arange(0, cutoff)
    test_idx = np.arange(cutoff, len(df_sorted))
    return df_sorted, train_idx, test_idx


def _existing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def fit_eval(
    df_sorted: pd.DataFrame,
    train_idx,
    test_idx,
    *,
    variant: str,
    features: list[str],
) -> list[dict]:
    feats = _existing(df_sorted, features)
    missing = [c for c in features if c not in df_sorted.columns]
    if missing:
        print(f"{variant}: missing columns (dropped): {missing}")

    if not feats:
        raise ValueError(f"{variant}: no usable features found")

    X_train = df_sorted.loc[train_idx, feats].fillna(0)
    y_train = df_sorted.loc[train_idx, "has_incident"].astype(int)
    X_test = df_sorted.loc[test_idx, feats].fillna(0)
    y_test = df_sorted.loc[test_idx, "has_incident"].astype(int)

    models = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        solver="liblinear",
                    ),
                ),
            ]
        ),
    }

    rows: list[dict] = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        rows.append(
            {
                "mode": MODE,
                "variant": variant,
                "model": model_name,
                "n_features": int(len(feats)),
                "roc_auc": roc_auc_score(y_test, proba) if y_test.nunique() > 1 else np.nan,
                "ap": average_precision_score(y_test, proba),
                "recall@100": recall_at_k(y_test, proba, 100),
                "recall@500": recall_at_k(y_test, proba, 500),
                "recall@1000": recall_at_k(y_test, proba, 1000),
                "test_incident_rate": float(y_test.mean()),
                "n_test": int(len(y_test)),
                "n_test_incidents": int(y_test.sum()),
            }
        )

        pred = df_sorted.loc[test_idx, ["trip_id", "dispatch_date", "has_incident"]].copy()
        pred["pred_proba"] = proba
        pred_path = OUT_DIR / f"pred_time_{variant}_{model_name}.parquet"
        pred.to_parquet(pred_path, index=False)

        top_path = OUT_DIR / f"top_predictions_time_{variant}_{model_name}.csv"
        pred.sort_values("pred_proba", ascending=False).head(200).to_csv(top_path, index=False)

        if model_name == "logreg":
            clf = model.named_steps["clf"]
            coefs = pd.DataFrame({"feature": feats, "coef": clf.coef_.ravel()})
            coefs["abs_coef"] = coefs["coef"].abs()
            coefs = coefs.sort_values("abs_coef", ascending=False)
            coefs_path = OUT_DIR / f"logreg_coefs_time_{variant}.csv"
            coefs.to_csv(coefs_path, index=False)

    return rows


def main() -> None:
    base = pd.read_parquet(IN_DIR / "trips_features_base.parquet")
    hist = pd.read_parquet(IN_DIR / "trips_features_history.parquet")

    for name, df in [("base", base), ("history", hist)]:
        for col in ["dispatch_date", "has_incident", "trip_id"]:
            if col not in df.columns:
                raise KeyError(f"{name}: missing {col}")

    base["dispatch_date"] = pd.to_datetime(base["dispatch_date"], errors="coerce")
    hist["dispatch_date"] = pd.to_datetime(hist["dispatch_date"], errors="coerce")

    base = base.dropna(subset=["dispatch_date", "has_incident"]).reset_index(drop=True)
    hist = hist.dropna(subset=["dispatch_date", "has_incident"]).reset_index(drop=True)

    base_sorted, base_train_idx, base_test_idx = time_split_idx(base, frac_train=0.8)
    hist_sorted, hist_train_idx, hist_test_idx = time_split_idx(hist, frac_train=0.8)

    rows: list[dict] = []

    # base only
    rows += fit_eval(
        base_sorted,
        base_train_idx,
        base_test_idx,
        variant="base",
        features=BASE_PRETRIP_FEATURES,
    )

    # old ablations (for comparability)
    rows += fit_eval(
        hist_sorted,
        hist_train_idx,
        hist_test_idx,
        variant="base+hist_counts",
        features=BASE_PRETRIP_FEATURES + HIST_COUNTS,
    )
    rows += fit_eval(
        hist_sorted,
        hist_train_idx,
        hist_test_idx,
        variant="base+hist_counts+exposure",
        features=BASE_PRETRIP_FEATURES + HIST_COUNTS + HIST_EXPOSURE,
    )
    rows += fit_eval(
        hist_sorted,
        hist_train_idx,
        hist_test_idx,
        variant="base+hist_rates",
        features=BASE_PRETRIP_FEATURES + HIST_RATES,
    )
    rows += fit_eval(
        hist_sorted,
        hist_train_idx,
        hist_test_idx,
        variant="base+hist_rates_smoothed",
        features=BASE_PRETRIP_FEATURES + HIST_RATES_SMOOTHED,
    )

    # new ablations
    rows += fit_eval(
        hist_sorted,
        hist_train_idx,
        hist_test_idx,
        variant="base+hist_recency",
        features=BASE_PRETRIP_FEATURES + HIST_RECENCY,
    )
    rows += fit_eval(
        hist_sorted,
        hist_train_idx,
        hist_test_idx,
        variant="base+hist_windows",
        features=BASE_PRETRIP_FEATURES + HIST_WINDOWS_CORE,
    )
    rows += fit_eval(
        hist_sorted,
        hist_train_idx,
        hist_test_idx,
        variant="base+hist_windows+flags",
        features=BASE_PRETRIP_FEATURES + HIST_WINDOWS_CORE + HIST_WINDOWS_SAFETY_FLAGS,
    )
    rows += fit_eval(
        hist_sorted,
        hist_train_idx,
        hist_test_idx,
        variant="base+hist_type_diversity",
        features=BASE_PRETRIP_FEATURES + HIST_TYPE_DIVERSITY,
    )
    rows += fit_eval(
        hist_sorted,
        hist_train_idx,
        hist_test_idx,
        variant="base+hist_severity_costs",
        features=BASE_PRETRIP_FEATURES + HIST_SEVERITY_COSTS,
    )

    # all history together
    ALL_HISTORY = (
        HIST_COUNTS
        + HIST_EXPOSURE
        + HIST_RATES_SMOOTHED
        + HIST_RECENCY
        + HIST_WINDOWS_CORE
        + HIST_WINDOWS_SAFETY_FLAGS
        + HIST_TYPE_DIVERSITY
        + HIST_SEVERITY_COSTS
    )
    rows += fit_eval(
        hist_sorted,
        hist_train_idx,
        hist_test_idx,
        variant="base+hist_all",
        features=BASE_PRETRIP_FEATURES + ALL_HISTORY,
    )

    res = pd.DataFrame(rows)
    out_path = OUT_DIR / "results_history_eval.csv"
    res.to_csv(out_path, index=False)

    print(res.sort_values(["ap", "roc_auc"], ascending=False).to_string(index=False))
    print("saved:", out_path)


if __name__ == "__main__":
    main()