FROM apache/airflow:2.9.1

# Копируем наши требования
COPY requirements.txt /requirements.txt

# Устанавливаем пакеты безопасно, используя официальные ограничения версий Airflow
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.12.txt"