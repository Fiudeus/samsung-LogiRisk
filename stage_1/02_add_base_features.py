from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "stage_1" / "output"
DATA_DIR = ROOT / "datasets"
OUT_DIR = ROOT / "stage_1" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def pick_existing(df: pd.DataFrame, cols: list[str], *, name: str) -> pd.DataFrame:
    """Безопасно выбирает только существующие колонки; печатает какие отсутствуют."""
    existing = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"{name}: missing columns (ok): {missing}")
    return df[existing].copy()


def main() -> None:
    # ===== base =====
    df = pd.read_parquet(IN_DIR / "trips_labeled.parquet")

    loads = pd.read_csv(DATA_DIR / "loads.csv")
    routes = pd.read_csv(DATA_DIR / "routes.csv")
    drivers = pd.read_csv(DATA_DIR / "drivers.csv")
    trucks = pd.read_csv(DATA_DIR / "trucks.csv")

    # dispatch_date нужен для time-features и truck_age
    if "dispatch_date" not in df.columns:
        raise KeyError("trips_labeled.parquet missing column: dispatch_date")
    df["dispatch_date"] = pd.to_datetime(df["dispatch_date"], errors="coerce")

    # ===== join: trips -> loads =====
    loads_cols_wanted = [
        "load_id",
        "route_id",
        "weight_lbs",
        "pieces",
        "revenue",
        "fuel_surcharge",
        "accessorial_charges",
        "fuel_surcharge_rate",
        "trailer_id",
    ]
    loads_small = pick_existing(loads, loads_cols_wanted, name="loads.csv")

    # Если fuel_surcharge_rate нет в сырье — досчитываем (как долю от revenue)
    if "fuel_surcharge_rate" not in loads_small.columns:
        if "fuel_surcharge" in loads_small.columns and "revenue" in loads_small.columns:
            denom = loads_small["revenue"].replace(0, pd.NA)
            loads_small["fuel_surcharge_rate"] = loads_small["fuel_surcharge"] / denom
        else:
            loads_small["fuel_surcharge_rate"] = pd.NA

    df = df.merge(loads_small, on="load_id", how="left")

    # ===== join: loads -> routes =====
    routes_cols_wanted = [
        "route_id",
        "typical_distance_miles",
        "typical_transit_days",
        "base_rate_per_mile",
    ]
    routes_small = pick_existing(routes, routes_cols_wanted, name="routes.csv")
    df = df.merge(routes_small, on="route_id", how="left")

    # ===== join: driver =====
    drivers_cols_wanted = [
        "driver_id",
        "years_experience",
    ]
    drivers_small = pick_existing(drivers, drivers_cols_wanted, name="drivers.csv")
    df = df.merge(drivers_small, on="driver_id", how="left")

    # ===== join: truck =====
    trucks_cols_wanted = [
        "truck_id",
        "model_year",
        "acquisition_mileage",
        "tank_capacity_gallons",
    ]
    trucks_small = pick_existing(trucks, trucks_cols_wanted, name="trucks.csv")
    df = df.merge(trucks_small, on="truck_id", how="left")

    # ===== time features =====
    df["dispatch_month"] = df["dispatch_date"].dt.month
    df["dispatch_dow"] = df["dispatch_date"].dt.dayofweek
    df["is_weekend"] = (df["dispatch_dow"] >= 5).astype(int)

    # ===== derived =====
    if "model_year" in df.columns:
        df["truck_age"] = df["dispatch_date"].dt.year - df["model_year"]
    else:
        df["truck_age"] = pd.NA

    # revenue_per_mile: если есть actual_distance_miles — используем её, иначе typical_distance_miles
    denom = None
    if "actual_distance_miles" in df.columns:
        denom = df["actual_distance_miles"]
    elif "typical_distance_miles" in df.columns:
        denom = df["typical_distance_miles"]

    if denom is not None and "revenue" in df.columns:
        df["revenue_per_mile"] = df["revenue"] / denom.replace(0, pd.NA)
    else:
        df["revenue_per_mile"] = pd.NA

    # ===== sanity checks =====
    if "trip_id" in df.columns:
        dup_rate = float(df["trip_id"].duplicated().mean())
        print("dup trip_id rate:", dup_rate)

    print("ROWS:", len(df))
    print("has_incident:", float(df["has_incident"].mean()) if "has_incident" in df.columns else "N/A")
    if "years_experience" in df.columns:
        print("NaN years_experience:", float(df["years_experience"].isna().mean()))
    print("NaN truck_age:", float(df["truck_age"].isna().mean()))
    print("NaN fuel_surcharge_rate:", float(df["fuel_surcharge_rate"].isna().mean()) if "fuel_surcharge_rate" in df.columns else "N/A")

    out_path = OUT_DIR / "trips_features_base.parquet"
    df.to_parquet(out_path, index=False)
    print("saved:", out_path)


if __name__ == "__main__":
    main()