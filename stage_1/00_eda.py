from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "datasets").exists():
            return p
    return start


def main() -> None:
    try:
        root = Path(__file__).resolve().parents[1]
    except NameError:
        root = find_project_root(Path.cwd())

    in_path = root / "stage_1" / "output" / "trips_features_base.parquet"
    out_dir = root / "stage_1" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("ROOT =", root)
    print("IN_PATH =", in_path)

    df = pd.read_parquet(in_path)

    # даты
    if "dispatch_date" in df.columns:
        df["dispatch_date"] = pd.to_datetime(df["dispatch_date"], errors="coerce")

    # размеры/период/класс
    print("\nРАЗМЕРЫ ИТОГОВОЙ ТАБЛИЦЫ:")
    print(f"df: {df.shape[0]} строк × {df.shape[1]} колонок")
    if "dispatch_date" in df.columns:
        print(f"Период данных: {df['dispatch_date'].min()} — {df['dispatch_date'].max()}")
    if "has_incident" in df.columns:
        print(f"Доля инцидентов: {df['has_incident'].mean():.3%}")
        print("Распределение классов:")
        print(df["has_incident"].value_counts(dropna=False))

    # пропуски по ключам
    print("\n=============== Пропуски после объединения (ключи) ===========================")
    for col in ["load_id", "route_id", "driver_id", "truck_id"]:
        if col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                print(f"  {col}: {missing} пропусков ({missing/len(df)*100:.2f}%)")

    # топ пропусков/нулей
    print("\nТоп-10 по пропускам:")
    print(df.isnull().mean().sort_values(ascending=False).head(10))

    numeric_cols = df.select_dtypes(include="number").columns
    print("\nТоп-10 по нулям (numeric):")
    print(((df[numeric_cols] == 0).mean()).sort_values(ascending=False).head(10))

    # years_experience (без падений)
    if "years_experience" in df.columns:
        s = df["years_experience"].dropna()
        if len(s) > 0:
            mode_value = s.mode().iloc[0]
            count = (df["years_experience"] == mode_value).sum()
            total = len(df)
            print("\n=== years_experience ===")
            print("mean:", s.mean())
            print(f"mode={mode_value:.2f} count={count}/{total} ({count/total*100:.3f}%)")
            print("top-10:")
            print(df["years_experience"].value_counts().head(10))
            print("describe:")
            print(df["years_experience"].describe())
    else:
        print("\nКолонка years_experience отсутствует в trips_features_base.parquet.")

    # корреляция внутри df
    corr = df.select_dtypes(include="number").corr(method="spearman")
    corr_path = out_dir / "correlation_matrix_trips_features_base.csv"
    corr.to_csv(corr_path)
    print("\nsaved:", corr_path)

    # ===== Полный heatmap корреляции (все numeric колонки) =====
    corr_full = corr.copy()
    mask = np.tril(np.ones_like(corr_full, dtype=bool))

    labels = [str(c).replace("_", "\n") for c in corr_full.columns]
    corr_plot = corr_full.copy()
    corr_plot.columns = labels
    corr_plot.index = labels

    n = corr_plot.shape[0]
    fig_w = max(22, n * 0.55)
    fig_h = max(18, n * 0.55)

    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        corr_plot,
        mask=mask,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.15,
        square=True,
        cbar_kws={"shrink": 0.6},
    )

    plt.title("Матрица корреляции", fontsize=24, pad=22)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()

    png_path = out_dir / "correlation_heatmap_full.png"
    svg_path = out_dir / "correlation_heatmap_full.svg"
    plt.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.show()

    print("saved:", png_path)
    print("saved:", svg_path)

    # гистограммы
    plt.figure(figsize=(12, 10))
    for i, col in enumerate(["typical_distance_miles", "weight_lbs", "years_experience", "truck_age", ], 1):
        if col not in df.columns:
            continue
        plt.subplot(2, 2, i)
        plt.hist(df[col].dropna(), bins=30, edgecolor="black", alpha=0.7)
        plt.title(f"Распределение: {col}")
        plt.xlabel(col)
        plt.ylabel("Количество")

    plt.tight_layout()
    dist_path = out_dir / "distributions.png"
    plt.savefig(dist_path, dpi=300, bbox_inches="tight")
    plt.show()
    print("saved:", dist_path)


if __name__ == "__main__":
    main()