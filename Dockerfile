FROM apache/airflow:2.9.1

USER root

# Устанавливаем Java для Spark
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Копируем наши требования
COPY requirements.txt /requirements.txt

# Устанавливаем пакеты с учетом ограничений Airflow
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.11.txt"