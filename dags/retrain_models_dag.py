from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Базовые аргументы для DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Инициализация DAG
with DAG(
        'logirisk_analytical_pipeline',
        default_args=default_args,
        description='Pipeline for LogiRisk: Spark ETL, Model Training, Cascade Ensemble, and Postgres Export',
        schedule_interval='0 2 * * 0',
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['logirisk', 'ml', 'spark'],
) as dag:
    # 1. Слой подготовки данных (Apache Spark)
    spark_etl = BashOperator(
        task_id='spark_feature_engineering',
        bash_command='python -m src.spark_features',
        cwd='/opt/airflow'
    )

    # 2. Слой обучения моделей (Можно запустить параллельно в Airflow!)
    train_catboost = BashOperator(
        task_id='train_catboost',
        bash_command='python -m src.models.catbst',
        cwd='/opt/airflow'
    )

    train_xgb = BashOperator(
        task_id='train_xgboost',
        bash_command='python -m src.models.xgb',
        cwd='/opt/airflow'
    )

    train_logreg = BashOperator(
        task_id='train_logreg',
        bash_command='python -m src.models.logreg',
        cwd='/opt/airflow'
    )

    train_extratrees = BashOperator(
        task_id='train_extratrees',
        bash_command='python -m src.models.extratrees',
        cwd='/opt/airflow'
    )

    # 3. Сборка каскадного ансамбля
    build_ensemble = BashOperator(
        task_id='build_cascade_ensemble',
        bash_command='python -m src.ensemble --threshold 0.90 --top-k -1',
        cwd='/opt/airflow'
    )

    # 4. Выгрузка витрины для аналитика в PostgreSQL
    export_to_db = BashOperator(
        task_id='export_to_postgres',
        bash_command='python -m src.export_to_postgres',
        cwd='/opt/airflow'
    )

    # Выстраиваем граф зависимостей
    # Сначала Spark -> потом все модели параллельно -> потом ансамбль -> потом экспорт
    spark_etl >> [train_catboost, train_xgb, train_logreg, train_extratrees] >> build_ensemble >> export_to_db