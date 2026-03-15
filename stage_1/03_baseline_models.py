from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "stage_1" / "output"
OUT_DIR = ROOT / "stage_1" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# MODE:
# - "post_trip": можно использовать actual_* и т.п. (если они известны)
# - "pre_trip": убираем явный leakage (если прогноз до выезда)
MODE = "pre_trip"

FEATURES_ALL = [
    "actual_distance_miles",
    "actual_duration_hours",
    "fuel_gallons_used",
    "average_mpg",
    "idle_time_hours",
    "weight_lbs",
    "pieces",
    "revenue",
    "fuel_surcharge",
    "accessorial_charges",
    "typical_distance_miles",
    "typical_transit_days",
    "base_rate_per_mile",
    "fuel_surcharge_rate",
    "years_experience",
    "truck_age",
    "dispatch_month",
    "dispatch_dow",
    "is_weekend",
    "revenue_per_mile",
]

LEAKY_PRETRIP = {
    "actual_distance_miles",
    "actual_duration_hours",
    "fuel_gallons_used",
    "average_mpg",
    "idle_time_hours",
    "revenue",
    "fuel_surcharge",
    "accessorial_charges",
    "fuel_surcharge_rate",
    "revenue_per_mile",
}


def recall_at_k(y_true, y_score, k: int) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    k = min(int(k), len(y_true))
    if y_true.sum() == 0:
        return 0.0
    idx = np.argsort(-y_score)[:k]
    return float(y_true[idx].sum() / y_true.sum())


def select_features(df: pd.DataFrame) -> list[str]:
    feats = FEATURES_ALL.copy()
    if MODE == "pre_trip":
        feats = [f for f in feats if f not in LEAKY_PRETRIP]

    missing = [f for f in feats if f not in df.columns]
    if missing:
        print("Missing features (will be dropped):", missing)
    feats = [f for f in feats if f in df.columns]
    if not feats:
        raise ValueError("No usable features found in dataframe.")
    return feats


def fit_eval(df: pd.DataFrame, split_name: str, train_idx, test_idx) -> list[dict]:
    feats = select_features(df)

    X_train = df.loc[train_idx, feats].copy().fillna(0)
    y_train = df.loc[train_idx, "has_incident"].astype(int)
    X_test = df.loc[test_idx, feats].copy().fillna(0)
    y_test = df.loc[test_idx, "has_incident"].astype(int)

    models = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
        ),
    }

    rows: list[dict] = []
    for name, model in models.items():
        model.fit(X_train, y_train)

        # DummyClassifier тоже имеет predict_proba
        proba = model.predict_proba(X_test)[:, 1]

        rows.append(
            {
                "mode": MODE,
                "split": split_name,
                "model": name,
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

        # сохраняем предикты (удобно для анализа)
        pred = df.loc[test_idx, ["trip_id", "dispatch_date", "has_incident"]].copy()
        pred["pred_proba"] = proba
        pred_path = OUT_DIR / f"pred_{split_name}_{name}_{MODE}.parquet"
        pred.to_parquet(pred_path, index=False)

        # top-200 для просмотра глазами
        top_path = OUT_DIR / f"top_predictions_{split_name}_{name}_{MODE}.csv"
        pred.sort_values("pred_proba", ascending=False).head(200).to_csv(top_path, index=False)

        # коэффициенты логрега (в CSV)
        if name == "logreg":
            coefs = pd.DataFrame({"feature": feats, "coef": model.coef_.ravel()})
            coefs["abs_coef"] = coefs["coef"].abs()
            coefs = coefs.sort_values("abs_coef", ascending=False)
            coefs_path = OUT_DIR / f"logreg_coefs_{split_name}_{MODE}.csv"
            coefs.to_csv(coefs_path, index=False)

    return rows


def main() -> None:
    df = pd.read_parquet(IN_DIR / "trips_features_base.parquet")
    df["dispatch_date"] = pd.to_datetime(df["dispatch_date"], errors="coerce")

    # keep only rows with target and date
    df = df.dropna(subset=["dispatch_date", "has_incident"]).reset_index(drop=True)

    results: list[dict] = []

    # random split (демо)
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=42,
        stratify=df["has_incident"],
    )
    results += fit_eval(df, "random_split", train_idx, test_idx)

    # time split (основной)
    df_sorted = df.sort_values("dispatch_date").reset_index(drop=True)
    cutoff = int(len(df_sorted) * 0.8)
    train_idx = np.arange(0, cutoff)
    test_idx = np.arange(cutoff, len(df_sorted))
    results += fit_eval(df_sorted, "time_split", train_idx, test_idx)

    res_df = pd.DataFrame(results)
    out_path = OUT_DIR / "results_base.csv"
    res_df.to_csv(out_path, index=False)
    print(res_df.to_string(index=False))
    print("saved:", out_path)


if __name__ == "__main__":
    main()