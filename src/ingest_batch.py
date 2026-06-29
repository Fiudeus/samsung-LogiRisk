from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Сюда складываем глобальные персистентные данные (наш аналог диска ClickHouse)
STORAGE_DIR = ROOT / "datasets" / "persistent_store"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def ingest_new_csv(uploaded_file, target_table: str) -> int:
    """
    Принимает файл из UI, валидирует схему, чистит даты
    и инкрементально дописывает в персистентный Parquet-архив.
    """
    df_upload = pd.read_csv(uploaded_file)

    # Очистка пустых строк в driver_id
    if "driver_id" in df_upload.columns:
        df_upload = df_upload.dropna(subset=["driver_id"])
        df_upload = df_upload[df_upload["driver_id"].astype(str).str.strip() != ""]

    if df_upload.empty:
        raise ValueError("Пакет данных пуст или не содержит корректных driver_id.")

    # Приведение дат к единому стандарту
    for col in df_upload.columns:
        if "date" in col.lower() or "time" in col.lower():
            if df_upload[col].dtype != "bool" and not pd.api.types.is_bool_dtype(df_upload[col]):
                df_upload[col] = pd.to_datetime(df_upload[col], errors="coerce").dt.strftime("%Y-%m-%d")

    # Путь к постоянному файлу таблицы
    parquet_target_path = STORAGE_DIR / f"{target_table}.parquet"

    # Инкрементальное добавление (совмещение старого архива с новым пакетом)
    if parquet_target_path.exists():
        df_old = pd.read_parquet(parquet_target_path)
        # Соединяем старую историю и новую пачку
        df_final = pd.concat([df_old, df_upload], ignore_index=True).drop_duplicates()
    else:
        df_final = df_upload

    # Сохраняем на "жесткий диск" проекта
    df_final.to_parquet(parquet_target_path, index=False, compression="snappy")

    return len(df_upload)