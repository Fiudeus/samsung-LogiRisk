from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets"
IN_DIR = ROOT / "stage_1" / "output"
OUT_DIR = ROOT / "stage_1" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# helpers
# -------------------------
def _prep_incidents(incidents: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    inc = incidents[[entity_col, "incident_date", "at_fault_flag", "preventable_flag", "injury_flag",
                     "vehicle_damage_cost", "cargo_damage_cost", "claim_amount", "incident_type"]].copy()
    inc = inc.dropna(subset=[entity_col, "incident_date"])
    inc = inc.sort_values("incident_date")

    # flags -> 0/1
    for c in ["at_fault_flag", "preventable_flag", "injury_flag"]:
        if c in inc.columns:
            inc[c] = pd.to_numeric(inc[c], errors="coerce").fillna(0).astype(int)

    # costs -> numeric
    for c in ["vehicle_damage_cost", "cargo_damage_cost", "claim_amount"]:
        if c in inc.columns:
            inc[c] = pd.to_numeric(inc[c], errors="coerce").fillna(0.0).astype(float)

    inc["damage_cost_total"] = inc.get("vehicle_damage_cost", 0.0) + inc.get("cargo_damage_cost", 0.0)
    return inc


def _asof_cumcount(
    trips: pd.DataFrame,
    inc: pd.DataFrame,
    *,
    entity_col: str,
    out_col: str,
    mask: pd.Series | None = None,
) -> pd.DataFrame:
    """
    cumulative count of incidents strictly before dispatch_date
    optionally with mask (e.g., at_fault only)
    """
    x = inc[[entity_col, "incident_date"]].copy()
    if mask is not None:
        x = x[mask.values].copy()

    if x.empty:
        out = trips.copy()
        out[out_col] = 0.0
        return out

    x = x.sort_values("incident_date")
    x["cnt"] = 1
    x["cum"] = x.groupby(entity_col)["cnt"].cumsum()

    out = pd.merge_asof(
        trips.sort_values("dispatch_date"),
        x[["incident_date", entity_col, "cum"]].sort_values("incident_date"),
        left_on="dispatch_date",
        right_on="incident_date",
        by=entity_col,
        direction="backward",
        allow_exact_matches=False,
    )
    out = out.rename(columns={"cum": out_col}).drop(columns=["incident_date"])
    out[out_col] = out[out_col].fillna(0.0)
    return out


def _asof_cumsum(
    trips: pd.DataFrame,
    inc: pd.DataFrame,
    *,
    entity_col: str,
    value_col: str,
    out_col: str,
    mask: pd.Series | None = None,
) -> pd.DataFrame:
    """
    cumulative sum(value_col) strictly before dispatch_date
    """
    x = inc[[entity_col, "incident_date", value_col]].copy()
    if mask is not None:
        x = x[mask.values].copy()

    if x.empty:
        out = trips.copy()
        out[out_col] = 0.0
        return out

    x = x.sort_values("incident_date")
    x["cum"] = x.groupby(entity_col)[value_col].cumsum()

    out = pd.merge_asof(
        trips.sort_values("dispatch_date"),
        x[["incident_date", entity_col, "cum"]].sort_values("incident_date"),
        left_on="dispatch_date",
        right_on="incident_date",
        by=entity_col,
        direction="backward",
        allow_exact_matches=False,
    )
    out = out.rename(columns={"cum": out_col}).drop(columns=["incident_date"])
    out[out_col] = out[out_col].fillna(0.0)
    return out


def _asof_last_date(
    trips: pd.DataFrame,
    inc: pd.DataFrame,
    *,
    entity_col: str,
    out_col: str,
    mask: pd.Series | None = None,
) -> pd.DataFrame:
    """
    last incident_date strictly before dispatch_date
    """
    x = inc[[entity_col, "incident_date"]].copy()
    if mask is not None:
        x = x[mask.values].copy()

    out = pd.merge_asof(
        trips.sort_values("dispatch_date"),
        x.sort_values("incident_date"),
        left_on="dispatch_date",
        right_on="incident_date",
        by=entity_col,
        direction="backward",
        allow_exact_matches=False,
    )
    out = out.rename(columns={"incident_date": out_col})
    return out


def _add_time_window_counts(
    trips: pd.DataFrame,
    inc: pd.DataFrame,
    *,
    entity_col: str,
    base_name: str,
    windows_days: list[int],
    mask: pd.Series | None = None,
) -> pd.DataFrame:
    """
    For each window W:
      count_last_Wd = cumcount_before(dispatch_date) - cumcount_before(dispatch_date - W days)
    """
    out = trips.copy()

    # cumulative to dispatch_date
    out = _asof_cumcount(out, inc, entity_col=entity_col, out_col=f"{base_name}_cum", mask=mask)

    for w in windows_days:
        tmp_col = f"{base_name}_cum_shift_{w}"

        shifted = out[[entity_col, "dispatch_date"]].copy()
        shifted["dispatch_date"] = shifted["dispatch_date"] - pd.Timedelta(days=int(w))

        # attach shifted cumulative counts aligned by index
        shifted = _asof_cumcount(
            shifted,
            inc,
            entity_col=entity_col,
            out_col=tmp_col,
            mask=mask,
        )

        out[tmp_col] = shifted[tmp_col].values

        out[f"{base_name}_last_{w}d"] = out[f"{base_name}_cum"] - out[tmp_col]
        out[f"{base_name}_last_{w}d"] = out[f"{base_name}_last_{w}d"].clip(lower=0).fillna(0.0)

        out = out.drop(columns=[tmp_col])

    out = out.drop(columns=[f"{base_name}_cum"])
    return out


def add_entity_trip_exposure(trips: pd.DataFrame, *, entity_col: str, out_col: str) -> pd.DataFrame:
    out = trips.sort_values("dispatch_date").copy()
    out[out_col] = out.groupby(entity_col).cumcount().astype(float)
    return out


def _asof_type_nunique(
    trips: pd.DataFrame,
    inc: pd.DataFrame,
    *,
    entity_col: str,
    out_col: str,
) -> pd.DataFrame:
    """
    Nunique incident_type strictly before dispatch_date.
    Implemented via expanding set size using cumulative nunique on sorted incidents.
    """
    x = inc[[entity_col, "incident_date", "incident_type"]].dropna(subset=[entity_col, "incident_date", "incident_type"]).copy()
    if x.empty:
        out = trips.copy()
        out[out_col] = 0.0
        return out

    x = x.sort_values("incident_date")

    # cumulative nunique per entity: use factorize + seen counts trick
    # mark first occurrence of (entity, type) as 1, then cumulative sum
    x["_type_key"] = x[entity_col].astype(str) + "||" + x["incident_type"].astype(str)
    x["_is_first"] = (~x.duplicated("_type_key")).astype(int)
    x["_cum_nunique"] = x.groupby(entity_col)["_is_first"].cumsum()

    out = pd.merge_asof(
        trips.sort_values("dispatch_date"),
        x[[entity_col, "incident_date", "_cum_nunique"]].sort_values("incident_date"),
        left_on="dispatch_date",
        right_on="incident_date",
        by=entity_col,
        direction="backward",
        allow_exact_matches=False,
    )
    out = out.rename(columns={"_cum_nunique": out_col}).drop(columns=["incident_date"])
    out[out_col] = out[out_col].fillna(0.0)
    return out


def main() -> None:
    df = pd.read_parquet(IN_DIR / "trips_features_base.parquet")
    df["dispatch_date"] = pd.to_datetime(df["dispatch_date"], errors="coerce")
    df = df.dropna(subset=["dispatch_date", "driver_id", "truck_id"]).copy()
    required = {"dispatch_date", "has_incident", "driver_id", "truck_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"trips_features_base.parquet missing columns: {sorted(missing)}")

    df = df.dropna(subset=["dispatch_date"]).sort_values("dispatch_date").reset_index(drop=True)

    incidents = pd.read_csv(DATA_DIR / "safety_incidents.csv")
    incidents["incident_date"] = pd.to_datetime(incidents["incident_date"], errors="coerce")
    incidents = incidents.dropna(subset=["incident_date"])

    # ------------ driver history ------------
    inc_driver = _prep_incidents(incidents, "driver_id")
    df_hist = df.copy()

    # lifetime counts
    df_hist = _asof_cumcount(df_hist, inc_driver, entity_col="driver_id", out_col="driver_incidents_before_trip")
    df_hist = _asof_cumcount(
        df_hist, inc_driver, entity_col="driver_id", out_col="driver_at_fault_incidents_before_trip",
        mask=(inc_driver["at_fault_flag"] == 1),
    )
    df_hist = _asof_cumcount(
        df_hist, inc_driver, entity_col="driver_id", out_col="driver_preventable_incidents_before_trip",
        mask=(inc_driver["preventable_flag"] == 1),
    )
    df_hist = _asof_type_nunique(df_hist, inc_driver, entity_col="driver_id", out_col="driver_incident_type_nunique_before_trip")

    # time windows (counts)
    df_hist = _add_time_window_counts(
        df_hist, inc_driver, entity_col="driver_id", base_name="driver_incidents", windows_days=[30, 90, 180, 365]
    )
    df_hist = _add_time_window_counts(
        df_hist, inc_driver, entity_col="driver_id", base_name="driver_at_fault_incidents", windows_days=[365],
        mask=(inc_driver["at_fault_flag"] == 1),
    )
    df_hist = _add_time_window_counts(
        df_hist, inc_driver, entity_col="driver_id", base_name="driver_preventable_incidents", windows_days=[365],
        mask=(inc_driver["preventable_flag"] == 1),
    )
    df_hist = _add_time_window_counts(
        df_hist, inc_driver, entity_col="driver_id", base_name="driver_injury_incidents", windows_days=[365],
        mask=(inc_driver["injury_flag"] == 1),
    )

    # time windows (cost sums)
    df_hist = _asof_cumsum(df_hist, inc_driver, entity_col="driver_id", value_col="claim_amount", out_col="driver_claim_amount_sum_before_trip")
    df_hist = _asof_cumsum(df_hist, inc_driver, entity_col="driver_id", value_col="damage_cost_total", out_col="driver_damage_cost_sum_before_trip")

    # last date / recency
    df_hist = _asof_last_date(df_hist, inc_driver, entity_col="driver_id", out_col="driver_last_incident_date")
    df_hist["driver_days_since_last_incident"] = (
        (df_hist["dispatch_date"] - df_hist["driver_last_incident_date"]).dt.days
    )
    df_hist["driver_days_since_last_incident"] = df_hist["driver_days_since_last_incident"].fillna(9999).astype(float)
    df_hist = df_hist.drop(columns=["driver_last_incident_date"])

    # exposure
    df_hist = add_entity_trip_exposure(df_hist, entity_col="driver_id", out_col="driver_trips_before_trip")

    # rates
    df_hist["driver_incident_rate_before_trip"] = df_hist["driver_incidents_before_trip"] / (
        df_hist["driver_trips_before_trip"].replace(0, np.nan)
    )
    df_hist["driver_incident_rate_before_trip"] = df_hist["driver_incident_rate_before_trip"].fillna(0.0)

    df_hist["driver_incident_rate_smoothed_before_trip"] = (
        (df_hist["driver_incidents_before_trip"] + 1.0) / (df_hist["driver_trips_before_trip"] + 20.0)
    )

    # ------------ truck history ------------
    inc_truck = _prep_incidents(incidents, "truck_id")

    df_hist = _asof_cumcount(df_hist, inc_truck, entity_col="truck_id", out_col="truck_incidents_before_trip")
    df_hist = _asof_cumcount(
        df_hist, inc_truck, entity_col="truck_id", out_col="truck_at_fault_incidents_before_trip",
        mask=(inc_truck["at_fault_flag"] == 1),
    )
    df_hist = _asof_cumcount(
        df_hist, inc_truck, entity_col="truck_id", out_col="truck_preventable_incidents_before_trip",
        mask=(inc_truck["preventable_flag"] == 1),
    )
    df_hist = _asof_type_nunique(df_hist, inc_truck, entity_col="truck_id", out_col="truck_incident_type_nunique_before_trip")

    df_hist = _add_time_window_counts(
        df_hist, inc_truck, entity_col="truck_id", base_name="truck_incidents", windows_days=[30, 90, 180, 365]
    )
    df_hist = _add_time_window_counts(
        df_hist, inc_truck, entity_col="truck_id", base_name="truck_at_fault_incidents", windows_days=[365],
        mask=(inc_truck["at_fault_flag"] == 1),
    )
    df_hist = _add_time_window_counts(
        df_hist, inc_truck, entity_col="truck_id", base_name="truck_preventable_incidents", windows_days=[365],
        mask=(inc_truck["preventable_flag"] == 1),
    )
    df_hist = _add_time_window_counts(
        df_hist, inc_truck, entity_col="truck_id", base_name="truck_injury_incidents", windows_days=[365],
        mask=(inc_truck["injury_flag"] == 1),
    )

    df_hist = _asof_cumsum(df_hist, inc_truck, entity_col="truck_id", value_col="claim_amount", out_col="truck_claim_amount_sum_before_trip")
    df_hist = _asof_cumsum(df_hist, inc_truck, entity_col="truck_id", value_col="damage_cost_total", out_col="truck_damage_cost_sum_before_trip")

    df_hist = _asof_last_date(df_hist, inc_truck, entity_col="truck_id", out_col="truck_last_incident_date")
    df_hist["truck_days_since_last_incident"] = (
        (df_hist["dispatch_date"] - df_hist["truck_last_incident_date"]).dt.days
    )
    df_hist["truck_days_since_last_incident"] = df_hist["truck_days_since_last_incident"].fillna(9999).astype(float)
    df_hist = df_hist.drop(columns=["truck_last_incident_date"])

    df_hist = add_entity_trip_exposure(df_hist, entity_col="truck_id", out_col="truck_trips_before_trip")

    df_hist["truck_incident_rate_before_trip"] = df_hist["truck_incidents_before_trip"] / (
        df_hist["truck_trips_before_trip"].replace(0, np.nan)
    )
    df_hist["truck_incident_rate_before_trip"] = df_hist["truck_incident_rate_before_trip"].fillna(0.0)

    df_hist["truck_incident_rate_smoothed_before_trip"] = (
        (df_hist["truck_incidents_before_trip"] + 1.0) / (df_hist["truck_trips_before_trip"] + 50.0)
    )

    # ------------ final cleanup ------------
    # replace inf/nan just in case
    df_hist = df_hist.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # sanity prints: show means for ALL newly created history features
    print("ROWS:", len(df_hist))
    print("incident_rate (has_incident mean):", float(df_hist["has_incident"].mean()))

    # detect new columns added by this script
    base_cols = set(df.columns)
    new_cols = [c for c in df_hist.columns if c not in base_cols]

    # keep only numeric new columns for mean()
    new_num_cols = [c for c in new_cols if pd.api.types.is_numeric_dtype(df_hist[c])]
    new_num_cols = sorted(new_num_cols)

    print(f"New columns added: {len(new_cols)} total, {len(new_num_cols)} numeric.")
    for c in new_num_cols:
        print(f"mean {c}:", float(df_hist[c].mean()))

    for id_col in ["trip_id", "driver_id", "truck_id"]:
        if id_col in df_hist.columns:
            df_hist[id_col] = df_hist[id_col].astype("string").fillna("")
    obj_cols = df_hist.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        df_hist[c] = df_hist[c].astype("string").fillna("")

    out_path = OUT_DIR / "trips_features_history.parquet"
    df_hist.to_parquet(out_path, index=False)
    print("saved:", out_path)


if __name__ == "__main__":
    main()