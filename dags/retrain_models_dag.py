from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'data_ml_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'driver_risk_weekly_retrain',
    default_args=default_args,
    description='Еженедельное автоматическое переобучение моделей рисков',
    schedule='0 2 * * 0',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['ml', 'clickhouse', 'postgres'],
) as dag:

    train_models = BashOperator(
        task_id='run_main_pipeline',
        bash_command='python -m src.main_pipeline',
    )

    export_to_postgres = BashOperator(
        task_id='export_predictions_to_postgres',
        bash_command='python -m src.export_to_postgres',
    )

    train_models >> export_to_postgres