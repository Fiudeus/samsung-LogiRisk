from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
ENSEMBLE_CSV = ROOT / "src" / "output" / "hybrid_ensemble_scored.csv"
DATABASE_URL = "postgresql://admin:admin_password@postgres_main:5432/driver_risks"


def export_predictions():
    if not ENSEMBLE_CSV.exists():
        print("[ОШИБКА] Запусти сначала пайплайн обучения.")
        return

    df = pd.read_csv(ENSEMBLE_CSV)

    export_df = pd.DataFrame({
        'driver_id': df['driver_id'],
        'month': pd.to_datetime(df['month']).dt.date,
        'raw_risk_score': df['ensemble_mean'],
        'raw_alert': df['has_critical_alert'].astype(int)
    })

    engine = create_engine(DATABASE_URL)

    with engine.begin() as conn:
        # 1. Таблица сырых предсказаний
        conn.execute(text("""
                          CREATE TABLE IF NOT EXISTS driver_predictions
                          (
                              driver_id
                              VARCHAR
                          (
                              50
                          ) PRIMARY KEY,
                              month DATE NOT NULL,
                              raw_risk_score FLOAT NOT NULL,
                              raw_alert INT NOT NULL,
                              updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                              );
                          """))

        # 2. Таблица кулдаунов (пополняется фронтендом/бэкендом)
        conn.execute(text("""
                          CREATE TABLE IF NOT EXISTS driver_cooldowns
                          (
                              driver_id
                              VARCHAR
                          (
                              50
                          ) PRIMARY KEY,
                              checked_at TIMESTAMP NOT NULL,
                              cooldown_until TIMESTAMP NOT NULL
                              );
                          """))

        # 3. УМНАЯ ВИТРИНА ДЛЯ ФРОНТЕНДА (SQL VIEW)
        conn.execute(text("""
                          CREATE
                          OR REPLACE VIEW driver_display_risks AS
                          SELECT p.driver_id,
                                 p.month,
                                 p.raw_risk_score,
                                 p.raw_alert,
                                 c.checked_at,
                                 c.cooldown_until,

                                 -- Расчет множителя M (от 0.0 до 1.0)
                                 CASE
                                     WHEN c.driver_id IS NULL THEN 1.0
                                     WHEN NOW() >= c.cooldown_until THEN 1.0
                                     WHEN NOW() <= c.checked_at THEN 0.0
                                     ELSE EXTRACT(EPOCH FROM (NOW() - c.checked_at)) /
                                          EXTRACT(EPOCH FROM (c.cooldown_until - c.checked_at))
                                     END AS recovery_multiplier,

                                 -- Итоговый скор для UI
                                 p.raw_risk_score * (
                                     CASE
                                         WHEN c.driver_id IS NULL THEN 1.0
                                         WHEN NOW() >= c.cooldown_until THEN 1.0
                                         WHEN NOW() <= c.checked_at THEN 0.0
                                         ELSE EXTRACT(EPOCH FROM (NOW() - c.checked_at)) /
                                              EXTRACT(EPOCH FROM (c.cooldown_until - c.checked_at))
                                         END
                                     )   AS display_risk_score,

                                 -- Гасим алерт, если кулдаун еще активен
                                 CASE
                                     WHEN c.driver_id IS NULL OR NOW() >= c.cooldown_until THEN p.raw_alert
                                     ELSE 0
                                     END AS display_alert

                          FROM driver_predictions p
                                   LEFT JOIN driver_cooldowns c ON p.driver_id = c.driver_id;
                          """))

        # Очищаем только старые предикты (кулдауны не трогаем!)
        conn.execute(text("TRUNCATE TABLE driver_predictions;"))

    print("Заливаем свежие сырые скоры в базу...")
    export_df.to_sql('driver_predictions', con=engine, if_exists='append', index=False)

    print(
        "\n[УСПЕХ] Данные залиты! Фронтенд может забирать готовые данные запросом: SELECT * FROM driver_display_risks")


if __name__ == "__main__":
    export_predictions()