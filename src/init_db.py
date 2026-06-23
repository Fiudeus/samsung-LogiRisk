from pathlib import Path
import clickhouse_connect
import pandas as pd
import os


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"


def init_clickhouse():
    print("Подключение к ClickHouse...")

    ch_host = os.getenv('CLICKHOUSE_HOST', '127.0.0.1')
    client = clickhouse_connect.get_client(host=ch_host, port=8123, username='admin', password='admin')

    print("Создание таблиц...")

    client.command("DROP TABLE IF EXISTS default.trips")
    client.command("""
                   CREATE TABLE default.trips
                   (
                       trip_id               String,
                       load_id               Nullable(String),
                       driver_id             String,
                       truck_id              Nullable(String),
                       trailer_id            Nullable(String),
                       dispatch_date         Date32,
                       actual_distance_miles Nullable(Float32),
                       actual_duration_hours Nullable(Float32),
                       fuel_gallons_used     Nullable(Float32),
                       average_mpg           Nullable(Float32),
                       idle_time_hours       Nullable(Float32),
                       trip_status           Nullable(String)
                   ) ENGINE = MergeTree()
        ORDER BY (driver_id, dispatch_date)
                   """)

    client.command("DROP TABLE IF EXISTS default.safety_incidents")
    client.command("""
                   CREATE TABLE default.safety_incidents
                   (
                       incident_id         String,
                       trip_id             Nullable(String),
                       truck_id            Nullable(String),
                       driver_id           Nullable(String),
                       incident_date       DateTime,
                       incident_type       Nullable(String),
                       location_city       Nullable(String),
                       location_state      Nullable(String),
                       at_fault_flag       UInt8,
                       injury_flag         UInt8,
                       vehicle_damage_cost Nullable(Float32),
                       cargo_damage_cost   Nullable(Float32),
                       claim_amount        Nullable(Float32),
                       preventable_flag    UInt8,
                       description         Nullable(String)
                   ) ENGINE = MergeTree()
        ORDER BY incident_id
                   """)

    client.command("DROP TABLE IF EXISTS default.delivery_events")
    client.command("""
                   CREATE TABLE default.delivery_events
                   (
                       event_id           String,
                       load_id            Nullable(String),
                       trip_id            Nullable(String),
                       event_type         Nullable(String),
                       facility_id        Nullable(String),
                       scheduled_datetime Nullable(DateTime),
                       actual_datetime    Nullable(DateTime),
                       detention_minutes  Nullable(Float32),
                       on_time_flag       UInt8,
                       location_city      Nullable(String),
                       location_state     Nullable(String)
                   ) ENGINE = MergeTree()
        ORDER BY event_id
                   """)

    client.command("DROP TABLE IF EXISTS default.driver_monthly_metrics")
    client.command("""
                   CREATE TABLE default.driver_monthly_metrics
                   (
                       driver_id             String,
                       month                 Date32,
                       trips_completed       Nullable(Int32),
                       total_miles           Nullable(Float32),
                       total_revenue         Nullable(Float32),
                       average_mpg           Nullable(Float32),
                       total_fuel_gallons    Nullable(Float32),
                       on_time_delivery_rate Nullable(Float32),
                       average_idle_hours    Nullable(Float32)
                   ) ENGINE = MergeTree()
            ORDER BY (driver_id, month)
                   """)

    client.command("DROP TABLE IF EXISTS default.drivers")
    client.command("""
                   CREATE TABLE default.drivers
                   (
                       driver_id         String,
                       first_name        Nullable(String),
                       last_name         Nullable(String),
                       hire_date         Nullable(Date32),
                       termination_date  Nullable(Date32),
                       license_number    Nullable(String),
                       license_state     Nullable(String),
                       date_of_birth     Nullable(Date32),
                       home_terminal     Nullable(String),
                       employment_status Nullable(String),
                       cdl_class         Nullable(String),
                       years_experience  Nullable(Int32)
                   ) ENGINE = MergeTree()
            ORDER BY driver_id
                   """)

    print("Загрузка CSV файлов в базу данных (это займет пару секунд)...")

    # Загрузка Trips
    trips_df = pd.read_csv(DATASETS_DIR / "trips.csv")
    trips_df = trips_df.dropna(subset=["dispatch_date", "driver_id", "trip_id"])
    trips_df['dispatch_date'] = pd.to_datetime(trips_df['dispatch_date']).dt.date
    client.insert_df('default.trips', trips_df)
    print(f"✓ Загружено поездок: {len(trips_df)}")

    # Загрузка Incidents
    incidents_df = pd.read_csv(DATASETS_DIR / "safety_incidents.csv")
    incidents_df = incidents_df.dropna(subset=["incident_id"])
    incidents_df['incident_date'] = pd.to_datetime(incidents_df['incident_date'])
    for col in ['at_fault_flag', 'injury_flag', 'preventable_flag']:
        incidents_df[col] = incidents_df[col].fillna(False).astype(bool).astype(int)
    client.insert_df('default.safety_incidents', incidents_df)
    print(f"✓ Загружено инцидентов: {len(incidents_df)}")

    # Загрузка Delivery Events
    delivery_df = pd.read_csv(DATASETS_DIR / "delivery_events.csv")
    delivery_df = delivery_df.dropna(subset=["event_id"])
    delivery_df['scheduled_datetime'] = pd.to_datetime(delivery_df['scheduled_datetime'])
    delivery_df['actual_datetime'] = pd.to_datetime(delivery_df['actual_datetime'])
    delivery_df['on_time_flag'] = delivery_df['on_time_flag'].fillna(False).astype(bool).astype(int)
    client.insert_df('default.delivery_events', delivery_df)
    print(f"✓ Загружено логов доставок: {len(delivery_df)}")

    # Загрузка Metrics
    metrics_df = pd.read_csv(DATASETS_DIR / "driver_monthly_metrics.csv")
    metrics_df = metrics_df.dropna(subset=["driver_id", "month"])
    metrics_df['month'] = pd.to_datetime(metrics_df['month']).dt.date
    client.insert_df('default.driver_monthly_metrics', metrics_df)
    print(f"✓ Загружено ежемесячных метрик: {len(metrics_df)}")

    # Загрузка Drivers
    drivers_df = pd.read_csv(DATASETS_DIR / "drivers.csv")
    drivers_df = drivers_df.dropna(subset=["driver_id"])
    drivers_df['hire_date'] = pd.to_datetime(drivers_df['hire_date']).dt.date
    drivers_df['termination_date'] = pd.to_datetime(drivers_df['termination_date']).dt.date
    drivers_df['date_of_birth'] = pd.to_datetime(drivers_df['date_of_birth']).dt.date
    client.insert_df('default.drivers', drivers_df)
    print(f"✓ Загружено профилей водителей: {len(drivers_df)}")

    print("\n[УСПЕХ] Все данные успешно мигрировали в ClickHouse!")

if __name__ == "__main__":
    init_clickhouse()