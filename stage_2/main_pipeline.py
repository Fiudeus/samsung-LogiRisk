from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# Определяем корень проекта для вывода красивых путей
ROOT = Path(__file__).resolve().parents[1]


def run_step(module_name: str, description: str, *args) -> None:
    """
    Запускает Python-модуль как отдельный процесс.
    Использование sys.executable гарантирует запуск в текущем виртуальном окружении (.venv).
    """
    print(f"\n{'=' * 80}")
    print(f"▶ Шаг: {description}")
    print(f"▶ Команда: python -m {module_name} {' '.join(args)}")
    print(f"{'=' * 80}\n")

    start_time = time.time()

    try:
        # Запускаем процесс, вывод автоматически транслируется в консоль
        subprocess.run([sys.executable, "-m", module_name, *args], check=True)

        elapsed = time.time() - start_time
        print(f"\n[УСПЕХ] Шаг '{description}' завершен за {elapsed:.2f} сек.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ОШИБКА] Шаг '{description}' упал с кодом {e.returncode}.")
        print("Остановка пайплайна.")
        sys.exit(1)


def main():
    print(f"\n🚀 ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА РАНЖИРОВАНИЯ (STAGE 2) 🚀")
    print(f"Рабочая директория: {ROOT}")

    total_start_time = time.time()

    # 1. Обучение базовых древесных моделей (участников ансамбля)
    # LogReg пропускаем, так как он показал слабые результаты и не нужен в ансамбле
    run_step("stage_2.models.xgb", "Обучение XGBoost")
    run_step("stage_2.models.catbst", "Обучение CatBoost")
    run_step("stage_2.models.hgb", "Обучение HistGradientBoosting")
    run_step("stage_2.models.extratrees", "Обучение ExtraTrees")

    # Можно раскомментировать, если хотите всегда пересчитывать EasyEnsemble
    # run_step("stage_2.models.easyensemble", "Обучение EasyEnsemble")

    # 2. Сборка гибридного ансамбля (Вариант Б: Mean + Max Alerts)
    run_step("stage_2.ensemble", "Сборка гибридного ансамбля", "--threshold", "0.90", "--top-k", "-1")

    # 3. Сводка метрик для отчета (Leaderboard)
    run_step("stage_2.summarize_metrics", "Генерация финального отчета по метрикам")

    total_elapsed = time.time() - total_start_time
    print(f"\n{'=' * 80}")
    print(f"🎉 ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН 🎉")
    print(f"Общее время выполнения: {total_elapsed / 60:.2f} минут.")
    print(f"Все артефакты (модели, метрики, ансамбль) сохранены в: stage_2/output/")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()