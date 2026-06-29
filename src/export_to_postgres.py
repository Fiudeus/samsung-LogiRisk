from __future__ import annotations
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "src" / "output"
DATASETS_DIR = ROOT / "datasets"


def main():
    print("=== ЭКСПОРТ ДАННЫХ В POSTGRESQL ===")

    # --- 1. ЗАГРУЗКА ПРЕДИКТОВ ---
    preds_path = OUTPUT_DIR / "hybrid_ensemble_scored.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Файл {preds_path} не найден! Запустите ансамбль.")
    df_preds = pd.read_csv(preds_path)

    # --- 2. ЗАГРУЗКА СПРАВОЧНИКА ВОДИТЕЛЕЙ ---
    # Проверяем Parquet из Data Lake, если его еще нет - берем базовый CSV
    drivers_parquet = DATASETS_DIR / "persistent_store" / "drivers.parquet"
    drivers_csv = DATASETS_DIR / "drivers.csv"

    if drivers_parquet.exists():
        df_drivers = pd.read_parquet(drivers_parquet)
    elif drivers_csv.exists():
        df_drivers = pd.read_csv(drivers_csv)
    else:
        print("⚠ Внимание: Справочник водителей не найден. Создаем пустую структуру.")
        df_drivers = pd.DataFrame(columns=['driver_id', 'first_name', 'last_name'])

    # --- 3. ПОДКЛЮЧЕНИЕ К БАЗЕ ---
    engine = create_engine('postgresql+psycopg2://admin:admin_password@postgres_main:5432/driver_risks')

    # Временно удаляем зависимую витрину
    with engine.begin() as conn:
        conn.execute(text("DROP VIEW IF EXISTS driver_display_risks CASCADE;"))
        print("✓ Старая витрина временно отключена.")

    # ЗАЛИВАЕМ СПРАВОЧНИК ВОДИТЕЛЕЙ (Фикс для Streamlit)
    df_drivers.to_sql('drivers', engine, if_exists='replace', index=False)
    print(f"✓ Загружено {len(df_drivers)} строк в справочник drivers.")

    # ЗАЛИВАЕМ ПРЕДИКТЫ
    df_preds.to_sql('driver_predictions', engine, if_exists='replace', index=False)
    print(f"✓ Загружено {len(df_preds)} строк в таблицу driver_predictions.")

    # Пересоздаем таблицу кулдаунов и накатываем умную витрину обратно
    with engine.begin() as conn:
        conn.execute(text("""
                          CREATE TABLE IF NOT EXISTS driver_cooldowns
                          (
                              driver_id
                              VARCHAR
                          (
                              50
                          ) PRIMARY KEY,
                              checked_at TIMESTAMP,
                              cooldown_until TIMESTAMP
                              );
                          """))

        conn.execute(text("""
                          CREATE
                          OR REPLACE VIEW driver_display_risks AS
                          SELECT p.driver_id,
                                 p.month,
                                 CASE
                                     WHEN c.cooldown_until IS NOT NULL AND NOW() < c.cooldown_until THEN 0.0
                                     ELSE p.display_score_100
                                     END AS display_risk_score,
                                 CASE
                                     WHEN c.cooldown_until IS NOT NULL AND NOW() < c.cooldown_until THEN 0
                                     ELSE p.has_critical_alert
                                     END AS display_alert
                          FROM driver_predictions p
                                   LEFT JOIN driver_cooldowns c
                                             ON CAST(p.driver_id AS TEXT) = CAST(c.driver_id AS TEXT);
                          """))

    print("✓ SQL-витрина driver_display_risks успешно пересоздана и готова к работе.")


if __name__ == "__main__":
    main()