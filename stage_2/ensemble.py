from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "stage_2" / "output"


def load_predictions() -> dict[str, pd.DataFrame]:
    """Загружает результаты работы 4 лучших моделей с учетом реальных имен колонок."""
    models = ["xgb", "catboost", "hgb", "extratrees"]
    preds = {}

    for m in models:
        path = OUTPUT_DIR / f"{m}_driver_month_scored.csv"
        if not path.exists():
            raise FileNotFoundError(f"Не найден файл {path}. Сначала запустите модель {m}.")

        df = pd.read_csv(path)
        df = df.rename(columns={"score": f"prob_{m}"})
        preds[m] = df[["driver_id", "month", f"prob_{m}", "y_true"]]

    return preds


def build_hybrid_ensemble(critical_threshold: float = 0.90) -> pd.DataFrame:
    """
    Реализует Вариант Б:
    Основа - средняя вероятность (Mean).
    Алерты - если хотя бы одна модель выдала риск выше critical_threshold.
    """
    preds = load_predictions()
    models = list(preds.keys())

    base_df = preds[models[0]].copy()
    for m in models[1:]:
        base_df = base_df.merge(preds[m].drop(columns=["y_true"]), on=["driver_id", "month"], how="left")

    prob_cols = [f"prob_{m}" for m in models]

    # 1. Средний скор (Mean) для базового ранжирования
    base_df["ensemble_mean"] = base_df[prob_cols].mean(axis=1)

    # 2. Максимальный скор (Max) для выявления критических ситуаций
    base_df["ensemble_max"] = base_df[prob_cols].max(axis=1)

    # 3. Выставляем красные флаги (Critical Alerts)
    base_df["has_critical_alert"] = (base_df["ensemble_max"] >= critical_threshold).astype(int)

    # Определяем источники алертов
    def get_alert_sources(row):
        if not row["has_critical_alert"]:
            return ""
        sources = [m for m in models if row[f"prob_{m}"] >= critical_threshold]
        return ", ".join(sources)

    base_df["alert_source"] = base_df.apply(get_alert_sources, axis=1)

    # 4. Двухуровневая сортировка: сначала алерты, затем по убыванию среднего риска
    base_df = base_df.sort_values(
        by=["has_critical_alert", "ensemble_mean"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return base_df


def main():
    # Настройка парсера аргументов командной строки
    parser = argparse.ArgumentParser(
        description="Гибридный ансамбль (Вариант Б) для ранжирования риска инцидентов водителей."
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.90,
        help="Критический порог для индивидуальных моделей (по умолчанию: 0.90)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Количество выводимых строк в консоль. Передайте -1 для вывода всего топа (по умолчанию: 10)"
    )
    parser.add_argument(
        "--month", "-m",
        type=str,
        default=None,
        help="Фильтрация по конкретному месяцу срезa (например, '2024-09-01'). Если не указан, выводится всё время."
    )

    args = parser.parse_args()

    print("=== HYBRID ENSEMBLE BLENDING (CLI MODE) ===")

    # 1. Строим ансамбль
    final_df = build_hybrid_ensemble(critical_threshold=args.threshold)

    # Сохраняем полный датасет (бизнес-логика требует сохранять всё без обрезки)
    out_path = OUTPUT_DIR / "hybrid_ensemble_scored.csv"
    final_df.to_csv(out_path, index=False)
    print(f"✓ Полный ансамбль сохранен в: {out_path}")

    # 2. Фильтрация по месяцу, если аргумент передан
    if args.month:
        # Приводим к строковому формату для надежного сравнения, если в df тип object/string
        final_df["month_str"] = final_df["month"].astype(str)
        # Поддерживаем форматы ГГГГ-ММ-ДД и ГГГГ-ММ
        filtered_df = final_df[final_df["month_str"].str.startswith(args.month)].copy()
        filtered_df = filtered_df.drop(columns=["month_str"])

        if filtered_df.empty:
            print(f"⚠ Внимание: Данные за месяц '{args.month}' не найдены в датасете.")
            return
        print(f"✓ Применен фильтр по месяцу: {args.month}")
    else:
        filtered_df = final_df

    # 3. Определение размера вывода (Top-K или Всё)
    if args.top_k == -1:
        print(f"\nВывод полного рейтинга (всего строк: {len(filtered_df)}):")
        display_df = filtered_df
    else:
        k = min(args.top_k, len(filtered_df))
        print(f"\nВывод ТОП-{k} водителей на проверку:")
        display_df = filtered_df.head(k)

    # Печать результатов в консоль
    cols_to_show = ["driver_id", "month", "has_critical_alert", "alert_source", "ensemble_mean", "y_true"]
    print(display_df[cols_to_show].to_string(index=False))

    # Вывод краткой сводки по алертам
    total_alerts = filtered_df["has_critical_alert"].sum()
    print(f"\n[Статистика]: Всего критических алертов в выбранном срезе (риск >= {args.threshold}): {total_alerts}")


if __name__ == "__main__":
    main()