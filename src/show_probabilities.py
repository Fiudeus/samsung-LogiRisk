from __future__ import annotations

import argparse

import pandas as pd

from src.data import ROOT

OUT_DIR = ROOT / "src" / "output"


def load_scored(model: str) -> pd.DataFrame:
    path = OUT_DIR / f"{model}_driver_month_scored.csv"
    if not path.exists():
        raise SystemExit(
            f"File not found: {path}\n"
            f"Run the model first, e.g.: python -m src.models.{model}"
        )

    df = pd.read_csv(path)
    required = {"driver_id", "month", "y_true", "score"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} missing columns: {sorted(missing)}")

    df["month"] = pd.to_datetime(df["month"])
    df["score"] = df["score"].astype(float)
    df["y_true"] = df["y_true"].astype(int)
    return df


def driver_prob_last(scored_dm: pd.DataFrame) -> pd.DataFrame:
    """
    One probability per driver: take the latest (last) month score.
    Also keep y_true for that (driver, month).
    """
    scored_dm = scored_dm.sort_values(["driver_id", "month"]).copy()
    last = scored_dm.groupby("driver_id", as_index=False).tail(1)

    last = last.rename(
        columns={
            "score": "prob_3m",
            "month": "as_of_month",
            "y_true": "y_true_3m",
        }
    )
    last = last[["driver_id", "as_of_month", "prob_3m", "y_true_3m"]].sort_values(
        "prob_3m", ascending=False
    )
    return last


def driver_prob_at_month(scored_dm: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """
    One probability per driver at a specific month.
    """
    month = pd.to_datetime(month).normalize()
    block = scored_dm[scored_dm["month"].dt.normalize() == month].copy()

    if block.empty:
        available = scored_dm["month"].dt.normalize().sort_values().unique()
        avail_str = ", ".join(pd.Series(available).astype(str).tolist()[:12])
        raise SystemExit(
            f"No rows found for month={month.date()}.\n"
            f"Example available months (first 12): {avail_str}"
        )

    block = block.rename(
        columns={
            "score": "prob_3m",
            "month": "as_of_month",
            "y_true": "y_true_3m",
        }
    )
    block = block[["driver_id", "as_of_month", "prob_3m", "y_true_3m"]].sort_values(
        "prob_3m", ascending=False
    )
    return block


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show driver accident probabilities from saved model scores."
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model name prefix used in src/output/<model>_driver_month_scored.csv "
            "(e.g. extratrees, easyensemble, xgb, hgb, catboost, logreg)."
        ),
    )
    parser.add_argument("--top", type=int, default=20, help="How many top drivers to print.")
    parser.add_argument("--driver-id", type=str, default=None, help="Optional: show probability for a specific driver_id.")
    parser.add_argument(
        "--month",
        type=str,
        default=None,
        help="Optional: choose a specific month (YYYY-MM-01). If omitted, uses latest month per driver (agg=last).",
    )
    args = parser.parse_args()

    scored = load_scored(args.model)

    if args.month is None:
        drv = driver_prob_last(scored)
        suffix = "last"
        mode_str = "agg=last (latest month per driver)"
    else:
        m = pd.to_datetime(args.month)
        drv = driver_prob_at_month(scored, m)
        suffix = f"month_{m.date()}"
        mode_str = f"fixed month = {m.date()}"

    # Nicer console view (round only for printing; keep full precision in CSV)
    pretty = drv.copy()
    pretty["prob_3m"] = pretty["prob_3m"].round(6)

    out_path = OUT_DIR / f"{args.model}_driver_probabilities_{suffix}.csv"
    drv.to_csv(out_path, index=False)

    print("\n=== PROBABILITIES (driver-level view) ===")
    print(f"Mode: {mode_str}")
    print("prob_3m: model score interpreted as probability of an accident in the next 3 months.")
    print("y_true_3m: whether an accident actually happened in that 3-month horizon (1=yes, 0=no).")
    print(f"Model: {args.model}")
    print(f"Saved: {out_path}")

    print(f"\n--- TOP {args.top} DRIVERS BY PROBABILITY ---")
    print(pretty.head(args.top).to_string(index=False))

    if args.driver_id is not None:
        q = pretty[pretty["driver_id"].astype(str) == str(args.driver_id)]
        print(f"\n--- QUERY driver_id={args.driver_id} ---")
        if q.empty:
            print("Not found.")
        else:
            print(q.to_string(index=False))


if __name__ == "__main__":
    main()