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
        'risk_score': df['ensemble_mean'],
        'has_critical_alert': df['has_critical_alert'].astype(int)
    })

    engine = create_engine(DATABASE_URL)

    with engine.begin() as conn:
        conn.execute(text("""
                          CREATE TABLE IF NOT EXISTS driver_predictions
                          (
                              id
                              SERIAL
                              PRIMARY
                              KEY,
                              driver_id
                              VARCHAR
                          (
                              50
                          ) NOT NULL,
                              month DATE NOT NULL,
                              risk_score FLOAT NOT NULL,
                              has_critical_alert INT NOT NULL,
                              updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                              );
                          """))

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

        conn.execute(text("TRUNCATE TABLE driver_predictions;"))

    print("Заливаем свежие скоры рисков...")
    export_df.to_sql('driver_predictions', con=engine, if_exists='append', index=False)

    with engine.begin() as conn:
        conn.execute(text("""
                          UPDATE driver_predictions p
                          SET has_critical_alert = 0 FROM driver_cooldowns c
                          WHERE p.driver_id = c.driver_id
                            AND NOW()
                              < c.cooldown_until;
                          """))

    print("\n[УСПЕХ] Данные экспортированы с учетом активных кулдаунов профилактических бесед!")


if __name__ == "__main__":
    export_predictions()