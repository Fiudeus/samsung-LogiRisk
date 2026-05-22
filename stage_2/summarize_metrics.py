from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stage_2.data import ROOT
from stage_2.metrics.driver_month_metrics import evaluate_driver_month
from stage_2.metrics.driver_metrics import evaluate_driver_many_aggs_from_driver_month_frame

OUT_DIR = ROOT / "stage_2" / "output"

# What we want to compare
DM_KS = (10, 20, 30, 50, 100, 200)
DRV_KS = (5, 10, 20, 30, 50, 100)


@dataclass(frozen=True)
class ModelArtifacts:
    model: str
    scored_path: Path


def _infer_model_name(path: Path) -> str:
    # expected: "<model>_driver_month_scored.csv"
    name = path.name
    if name.endswith("_driver_month_scored.csv"):
        return name[: -len("_driver_month_scored.csv")]
    return path.stem


def _load_scored(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"driver_id", "month", "y_true", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    df["month"] = pd.to_datetime(df["month"])
    df["y_true"] = df["y_true"].astype(int)
    df["score"] = df["score"].astype(float)
    return df


def _driver_month_summary_wide(scored_dm: pd.DataFrame, model: str) -> pd.Series:
    y = scored_dm["y_true"].to_numpy()
    s = scored_dm["score"].to_numpy()
    rep = evaluate_driver_month(y, s, tag=f"{model}_dm", ks=DM_KS)

    keep = ["n", "positives", "base_rate", "roc_auc", "ap"]
    for k in DM_KS:
        keep += [f"tp@{k}", f"fn@{k}", f"miss_rate@{k}", f"precision@{k}", f"recall@{k}"]
    rep = rep[keep].copy()
    rep["model"] = model
    return rep


def _driver_level_summary_wide(scored_dm: pd.DataFrame, model: str) -> pd.DataFrame:
    metrics_aggs, _ = evaluate_driver_many_aggs_from_driver_month_frame(
        scored_dm[["driver_id", "month", "y_true", "score"]].copy(),
        ks=tuple(k for k in DRV_KS if k <= scored_dm["driver_id"].nunique()),
        q=0.2,
        tag_prefix=f"{model}_driver",
    )

    metrics_aggs = metrics_aggs.copy()
    metrics_aggs["model"] = model
    metrics_aggs["agg"] = metrics_aggs.index.astype(str)
    metrics_aggs.reset_index(drop=True, inplace=True)

    keep = ["model", "agg", "n_drivers", "positives", "base_rate", "roc_auc", "ap"]
    for k in DRV_KS:
        cols_k = [f"tp@{k}", f"fn@{k}", f"miss_rate@{k}", f"precision@{k}", f"recall@{k}"]
        keep += [c for c in cols_k if c in metrics_aggs.columns]
    return metrics_aggs[keep]


def _melt_at_k(
    df: pd.DataFrame,
    *,
    ks: tuple[int, ...],
    id_cols: list[str],
) -> pd.DataFrame:
    """
    Convert wide columns like tp@10, fn@10, ... into a long table with a row per k.
    """
    rows: list[pd.DataFrame] = []
    metrics = ("tp", "fn", "miss_rate", "precision", "recall", "lift")

    for k in ks:
        base = df[id_cols].copy()
        base["k"] = int(k)
        for m in metrics:
            col = f"{m}@{k}"
            if col in df.columns:
                base[m] = df[col]
        rows.append(base)

    out = pd.concat(rows, ignore_index=True)
    return out


def main() -> None:
    if not OUT_DIR.exists():
        raise SystemExit(f"Output dir does not exist: {OUT_DIR}")

    scored_files = sorted(OUT_DIR.glob("*_driver_month_scored.csv"))
    if not scored_files:
        raise SystemExit(f"No '*_driver_month_scored.csv' found in {OUT_DIR}")

    artifacts = [ModelArtifacts(model=_infer_model_name(p), scored_path=p) for p in scored_files]

    dm_rows: list[pd.Series] = []
    drv_rows: list[pd.DataFrame] = []

    for art in artifacts:
        scored = _load_scored(art.scored_path)
        dm_rows.append(_driver_month_summary_wide(scored, art.model))
        drv_rows.append(_driver_level_summary_wide(scored, art.model))

    dm_wide = pd.DataFrame(dm_rows)
    drv_wide = pd.concat(drv_rows, ignore_index=True)

    # Save raw wide summaries (nice for Excel)
    out_dm = OUT_DIR / "summary_driver_month_metrics.csv"
    out_drv = OUT_DIR / "summary_driver_metrics.csv"
    dm_wide.to_csv(out_dm, index=False)
    drv_wide.to_csv(out_drv, index=False)

    # --------- Long / readable prints ----------
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.expand_frame_repr", False)

    # DRIVER-MONTH long view
    dm_long = _melt_at_k(
        dm_wide,
        ks=DM_KS,
        id_cols=["model", "ap", "roc_auc", "positives"],
    )

    # Print grouped by k so you can compare models at each budget
    print("\n=== LEADERBOARD: DRIVER-MONTH (long format) ===")
    for k in DM_KS:
        block = dm_long[dm_long["k"] == k].copy()
        # primary: FN (miss fewer positives), secondary: AP
        sort_cols = [c for c in ["fn", "ap"] if c in block.columns]
        if sort_cols:
            block = block.sort_values(sort_cols, ascending=[True, False][: len(sort_cols)])
        cols = ["model", "k", "ap", "roc_auc", "tp", "fn", "miss_rate", "precision", "recall"]
        cols = [c for c in cols if c in block.columns]
        print(f"\n--- DRIVER-MONTH @K={k} (sorted by FN asc, then AP desc) ---")
        print(block[cols].to_string(index=False))

    # DRIVER-level long view by agg
    preferred_aggs = ["topq_mean", "max", "mean", "last"]
    drv_filtered = drv_wide[drv_wide["agg"].isin(preferred_aggs)].copy()

    drv_long = _melt_at_k(
        drv_filtered,
        ks=DRV_KS,
        id_cols=["model", "agg", "ap", "roc_auc", "positives"],
    )

    print("\n=== LEADERBOARD: DRIVER (long format) ===")
    for agg in preferred_aggs:
        for k in DRV_KS:
            block = drv_long[(drv_long["agg"] == agg) & (drv_long["k"] == k)].copy()
            if block.empty:
                continue

            sort_cols = [c for c in ["fn", "ap"] if c in block.columns]
            if sort_cols:
                block = block.sort_values(sort_cols, ascending=[True, False][: len(sort_cols)])

            cols = ["model", "agg", "k", "ap", "roc_auc", "tp", "fn", "miss_rate", "precision", "recall"]
            cols = [c for c in cols if c in block.columns]

            print(f"\n--- DRIVER agg={agg} @K={k} (sorted by FN asc, then AP desc) ---")
            print(block[cols].to_string(index=False))

    print("\nSaved:", out_dm)
    print("Saved:", out_drv)


if __name__ == "__main__":
    main()