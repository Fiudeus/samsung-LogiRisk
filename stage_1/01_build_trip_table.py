from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets"
OUT_DIR = ROOT / "stage_1" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    trips = pd.read_csv(DATA_DIR / "trips.csv")
    incidents = pd.read_csv(DATA_DIR / "safety_incidents.csv")

    # базовые проверки
    required_trips_cols = {"trip_id", "dispatch_date"}
    missing = required_trips_cols - set(trips.columns)
    if missing:
        raise KeyError(f"trips.csv is missing columns: {sorted(missing)}")

    if "trip_id" not in incidents.columns:
        raise KeyError("safety_incidents.csv is missing column: trip_id")

    # парсим дату отправки
    trips["dispatch_date"] = pd.to_datetime(trips["dispatch_date"], errors="coerce")

    # таргет
    trips["has_incident"] = trips["trip_id"].isin(incidents["trip_id"].dropna()).astype(int)

    # статистика
    print("TRIPS:", len(trips))
    print("dispatch_date:", trips["dispatch_date"].min(), "→", trips["dispatch_date"].max())
    print("incident_rate:", float(trips["has_incident"].mean()))
    print("incidents:", int(trips["has_incident"].sum()))

    out_path = OUT_DIR / "trips_labeled.parquet"
    trips.to_parquet(out_path, index=False)
    print("saved:", out_path)


if __name__ == "__main__":
    main()