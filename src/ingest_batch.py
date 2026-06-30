import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STORAGE_DIR = ROOT / "datasets" / "persistent_store"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def ingest_new_csv(uploaded_file, target_table: str) -> int:
    """
    Pandas-версия с умным UPSERT: Принимает файл, чистит,
    и безопасно обновляет Parquet-архив (заменяя старые данные новыми).
    """
    uploaded_file.seek(0)
    try:
        df_upload = pd.read_csv(uploaded_file)
    except Exception as e:
        raise ValueError(f"Не удалось прочитать CSV: {e}")

    if df_upload.empty:
        raise ValueError("Пакет данных пуст.")

    # 1. Очистка driver_id
    if "driver_id" in df_upload.columns:
        df_upload = df_upload.dropna(subset=["driver_id"])
        df_upload["driver_id"] = df_upload["driver_id"].astype(str).str.strip()
        df_upload = df_upload[df_upload["driver_id"] != ""]

    if df_upload.empty:
        raise ValueError("Нет корректных driver_id после очистки.")

    # 2. Приведение дат
    for col in df_upload.columns:
        if "date" in col.lower() or "time" in col.lower():
            df_upload[col] = pd.to_datetime(df_upload[col], errors="coerce").dt.strftime("%Y-%m-%d")

    parquet_target_path = STORAGE_DIR / f"{target_table}.parquet"

    # 3. Умное слияние (UPSERT)
    if parquet_target_path.exists():
        df_old = pd.read_parquet(parquet_target_path)

        # Выравниваем типы данных, чтобы concat не упал
        for col in df_old.columns:
            if col in df_upload.columns:
                df_upload[col] = df_upload[col].astype(df_old[col].dtype)

        df_final = pd.concat([df_old, df_upload], ignore_index=True)

        # Заменяем старые записи новыми по ключу driver_id + month
        if "driver_id" in df_final.columns and "month" in df_final.columns:
            df_final = df_final.drop_duplicates(subset=["driver_id", "month"], keep="last")
        else:
            df_final = df_final.drop_duplicates()
    else:
        df_final = df_upload

    # 4. Сохраняем обратно в Parquet
    df_final.to_parquet(parquet_target_path, index=False, compression="snappy")
    return len(df_upload)