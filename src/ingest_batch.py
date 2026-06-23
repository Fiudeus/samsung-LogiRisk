import sys
import os
from pathlib import Path
import pandas as pd
import clickhouse_connect


def ingest_new_csv(csv_path: str, target_table: str):
    path = Path(csv_path)
    if not path.exists():
        print(f"[ОШИБКА] Файл не найден: {path}")
        return

    ch_host = os.getenv('CLICKHOUSE_HOST', '127.0.0.1')
    client = clickhouse_connect.get_client(host=ch_host, port=8123, username='admin', password='admin')

    df = pd.read_csv(path)

    print(f"Обработка новой пачки для таблицы {target_table} ({len(df)} строк)...")

    # Базовая предобработка дат в зависимости от таблицы
    if 'date' in path.name or 'event' in path.name or 'incident' in path.name:
        for col in df.columns:
            if 'date' in col or 'datetime' in col:
                df[col] = pd.to_datetime(df[col])
                if target_table in ['trips', 'driver_monthly_metrics', 'drivers']:
                    df[col] = df[col].dt.date

    # Нативный инсерт в ClickHouse (данные просто добавятся в конец)
    client.insert_df(f'default.{target_table}', df)
    print(f"✓ Пачка успешно добавлена в ClickHouse default.{target_table}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Использование: python -m src.ingest_batch <путь_к_csv> <имя_таблицы>")
    else:
        ingest_new_csv(sys.argv[1], sys.argv[2])