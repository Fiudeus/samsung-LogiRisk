from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# Определяем корень проекта для вывода красивых путей
ROOT = Path(__file__).resolve().parents[1]


def run_step(module_name: str, description: str, *args) -> None:
    print(f"\n{'=' * 80}")
    print(f"▶ Шаг: {description}")
    print(f"▶ Команда: python -m {module_name} {' '.join(args)}")
    print(f"{'=' * 80}\n")

    start_time = time.time()

    try:
        subprocess.run([sys.executable, "-m", module_name, *args], check=True)
        elapsed = time.time() - start_time
        print(f"\n[УСПЕХ] Шаг '{description}' завершен за {elapsed:.2f} сек.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ОШИБКА] Шаг '{description}' упал с кодом {e.returncode}.")
        print("Остановка пайплайна.")
        sys.exit(1)


def main():
    print(f"\n🚀 ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА РАНЖИРОВАНИЯ LOGIRISK 🚀")
    print(f"Рабочая директория: {ROOT}")

    total_start_time = time.time()

    # 0. Подготовка данных (ETL & Feature Engineering)
    run_step("src.spark_features", "Генерация признаков (Apache Spark -> Parquet)")

    # 1. Обучение базовых древесных моделей (участников ансамбля)
    run_step("src.models.xgb", "Обучение XGBoost")
    run_step("src.models.catbst", "Обучение CatBoost")
    # run_step("src.models.hgb", "Обучение HistGradientBoosting")
    run_step("src.models.extratrees", "Обучение ExtraTrees")
    run_step("src.models.logreg", "Обучение LogReg")
    # run_step("src.models.easyensemble", "Обучение EasyEnsemble")

    # 2. Сборка гибридного ансамбля
    run_step("src.ensemble", "Сборка гибридного ансамбля", "--threshold", "0.90", "--top-k", "-1")

    # 3. Сводка метрик для отчета (Leaderboard) - фокус на Average Precision и Precision@K
    run_step("src.summarize_metrics", "Генерация финального отчета по метрикам")

    # 4. Serving Layer (Опционально, если файл уже есть)
    # Выгрузка финальных скоров в БД для работы оператора с аналитическим инструментом
    run_step("src.export_to_postgres", "Экспорт витрины рисков в PostgreSQL")

    total_elapsed = time.time() - total_start_time
    print(f"\n{'=' * 80}")
    print(f"🎉 ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН 🎉")
    print(f"Общее время выполнения: {total_elapsed / 60:.2f} минут.")
    print(f"Все артефакты (модели, метрики, ансамбль) сохранены в: src/output/")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()