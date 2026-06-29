from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "src" / "output"


def load_predictions() -> dict[str, pd.DataFrame]:
    """Загружает результаты работы моделей."""
    models = ["logreg", "catboost", "xgb", "extratrees"]
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
    Реализует каскадный условный ансамбль (Tiered Cascade) на базе двух таблиц
    с последующим нормированием индексов риска (Risk Banding) от 0 до 100.
    """
    preds = load_predictions()
    models = list(preds.keys())

    # Собираем все предсказания в один датафрейм
    base_df = preds["catboost"].copy()
    for m in ["logreg", "xgb", "extratrees"]:
        base_df = base_df.merge(preds[m].drop(columns=["y_true"]), on=["driver_id", "month"], how="left")

    prob_cols = [f"prob_{m}" for m in models]

    # 1. Расчет базового взвешенного скора для остального автопарка
    weights = {"catboost": 0.50, "xgb": 0.35, "logreg": 0.12, "extratrees": 0.03}
    base_df["weighted_score"] = (
            base_df["prob_catboost"] * weights["catboost"] +
            base_df["prob_xgb"] * weights["xgb"] +
            base_df["prob_logreg"] * weights["logreg"] +
            base_df["prob_extratrees"] * weights["extratrees"]
    )

    # 2. Строим жесткий каскад по рангу CatBoost внутри каждого месяца
    base_df["catboost_rank"] = base_df.groupby("month")["prob_catboost"].rank(ascending=False, method="first")

    # Глобальные экстремумы для честного MinMax-масштабирования второго яруса
    global_max = base_df["weighted_score"].max()
    global_min = base_df["weighted_score"].min()
    safe_denom = (global_max - global_min) if (global_max - global_min) > 0 else 1.0

    # === РЕЛИЗ ИДЕИ: ДВЕ ТАБЛИЧКИ (STITCHING) ===

    # Таблица А: Жесткий Топ-20 от CatBoost (Tier 1) -> Грейд A
    t1_df = base_df[base_df["catboost_rank"] <= 20].copy()
    t1_df["ensemble_tier"] = 1
    t1_df["risk_grade"] = "A (Critical)"

    # Масштабируем внутренний топ Кэтбуста в верхнюю половину монотонной шкалы [0.5, 1.0]
    t1_max = t1_df["prob_catboost"].max() if not t1_df.empty else 1.0
    t1_min = t1_df["prob_catboost"].min() if not t1_df.empty else 0.0
    denom_t1 = (t1_max - t1_min) if (t1_max - t1_min) > 0 else 1.0
    t1_df["ensemble_mean"] = 0.50 + ((t1_df["prob_catboost"] - t1_min) / denom_t1) * 0.50

    # Таблица Б: Все остальные водители (Tier 2) -> Грейды B, C, D
    t2_df = base_df[base_df["catboost_rank"] > 20].copy()
    t2_df["ensemble_tier"] = 2
    t2_df["risk_grade"] = "D (Low)"  # Значение по умолчанию

    # Масштабируем взвешенный скор остатка в нижнюю половину монотонной шкалы [0.0, 0.499]
    t2_df["ensemble_mean"] = ((t2_df["weighted_score"] - global_min) / safe_denom) * 0.499

    # Динамическая нарезка грейдов внутри Tier 2 через процентили (Квантили)
    if not t2_df.empty:
        # Топ-5% риска из остатка получают Грейд B, следующие 15% — Грейд C, остальным оставляем D
        quantiles = pd.qcut(
            t2_df["ensemble_mean"],
            q=[0, 0.80, 0.95, 1.0],
            labels=["D (Low)", "C (Medium)", "B (High)"],
            duplicates='drop'
        )
        t2_df["risk_grade"] = quantiles

    # Склеиваем две таблицы обратно в единый консистентный датасет
    final_df = pd.concat([t1_df, t2_df], ignore_index=True)

    # Линейный перевод монотонного скора в стобалльный Индекс Риска для UI дашборда
    final_df["display_score_100"] = (final_df["ensemble_mean"] * 100).round(1)

    # 3. Алерты безопасности (вычисляются по сырым пикам моделей до сжатия)
    final_df["ensemble_max"] = final_df[prob_cols].max(axis=1)
    final_df["has_critical_alert"] = (final_df["ensemble_max"] >= critical_threshold).astype(int)

    def get_alert_sources(row):
        if not row["has_critical_alert"]:
            return ""
        return ", ".join([m for m in models if row[f"prob_{m}"] >= critical_threshold])

    final_df["alert_source"] = final_df.apply(get_alert_sources, axis=1)

    # Жесткая финальная сортировка: алерты -> ярус каскада -> итоговый балл -> стабильный ID
    final_df = final_df.sort_values(
        by=["has_critical_alert", "ensemble_tier", "ensemble_mean", "driver_id"],
        ascending=[False, True, False, True]
    ).reset_index(drop=True)

    return final_df


def main():
    parser = argparse.ArgumentParser(description="Каскадный условный ансамбль LogiRisk.")
    parser.add_argument("--threshold", "-t", type=float, default=0.90)
    parser.add_argument("--top-k", "-k", type=int, default=15)
    parser.add_argument("--month", "-m", type=str, default=None)

    args = parser.parse_args()
    print("=== HYBRID ENSEMBLE BLENDING (CASCADE TIERED MODE) ===")

    final_df = build_hybrid_ensemble(critical_threshold=args.threshold)

    out_path = OUTPUT_DIR / "hybrid_ensemble_scored.csv"
    final_df.to_csv(out_path, index=False)
    print(f"✓ Каскадный ансамбль сохранен в: {out_path}")

    if args.month:
        final_df["month_str"] = final_df["month"].astype(str)
        filtered_df = final_df[final_df["month_str"].str.startswith(args.month)].copy()
        filtered_df = filtered_df.drop(columns=["month_str"])
        if filtered_df.empty:
            print(f"⚠ Данные за месяц '{args.month}' не найдены.")
            return
        print(f"✓ Фильтр по месяцу: {args.month}")
    else:
        filtered_df = final_df

    if args.top_k == -1:
        display_df = filtered_df
    else:
        k = min(args.top_k, len(filtered_df))
        print(f"\nВывод ТОП-{k} водителей повышенного риска:")
        display_df = filtered_df.head(k)

    # Выводим в консоль красивую бизнес-таблицу
    cols_to_show = ["driver_id", "month", "risk_grade", "display_score_100", "has_critical_alert", "y_true"]
    print(display_df[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()