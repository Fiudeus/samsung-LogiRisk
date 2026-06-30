import os
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def main():
    spark = SparkSession.builder \
        .appName("LogiRisk_Feature_Engineering") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    print("=== SPARK SESSION STARTED ===")

    # Динамическое определение путей
    if os.path.exists("/opt/airflow/datasets"):
        datasets_dir = "/opt/airflow/datasets"
        output_dir = "/opt/airflow/src/output"
    else:
        project_root = Path(__file__).resolve().parent.parent
        datasets_dir = str(project_root / "datasets")
        output_dir = str(project_root / "src" / "output")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Загрузка данных
    store_dir = f"{datasets_dir}/persistent_store"

    def load_table(table_name: str):
        parquet_path = f"{store_dir}/{table_name}.parquet"
        csv_path = f"{datasets_dir}/{table_name}.csv"

        if os.path.exists(parquet_path) or os.path.exists(parquet_path.replace("file://", "")):
            return spark.read.parquet(parquet_path)
        else:
            print(f"Внимание: {parquet_path} не найден. Читаем базовый {csv_path}")
            return spark.read.csv(csv_path, header=True, inferSchema=True)

    trips = load_table("trips")
    incidents = load_table("safety_incidents")
    metrics = load_table("driver_monthly_metrics")
    delivery = load_table("delivery_events")
    drivers = load_table("drivers")

    # 2. Предобработка
    trips = trips.filter(
        F.col("dispatch_date").isNotNull() & F.col("driver_id").isNotNull() & F.col("trip_id").isNotNull()) \
        .withColumn("month", F.trunc(F.to_date(F.col("dispatch_date")), "MM"))

    incident_trips = incidents.select("trip_id").distinct().withColumn("has_incident", F.lit(1))
    trips_with_inc = trips.join(incident_trips, on="trip_id", how="left").na.fill({"has_incident": 0})

    # 3. Агрегация логов на уровне driver-month + Кросс-доменные фичи (скорость)
    driver_month_trips = trips_with_inc.groupBy("driver_id", "month").agg(
        F.count("trip_id").cast("double").alias("trips_cnt"),
        F.sum("has_incident").cast("double").alias("incidents_cnt"),
        F.countDistinct("truck_id").cast("double").alias("unique_trucks"),
        F.sum("actual_distance_miles").cast("double").alias("total_distance"),
        F.sum("actual_duration_hours").cast("double").alias("total_duration")
    ).withColumn("truck_churn_rate", F.col("unique_trucks") / F.col("trips_cnt")) \
     .withColumn("avg_speed", F.when(F.col("total_duration") > 0, F.col("total_distance") / F.col("total_duration")).otherwise(0.0)) \
     .drop("total_distance", "total_duration")

    # Агреграция простоев (Detention)
    delivery_clean = delivery.filter(F.col("trip_id").isNotNull() & F.col("detention_minutes").isNotNull())
    detention_features = trips.join(delivery_clean, on="trip_id", how="inner") \
        .groupBy("driver_id", "month").agg(
        F.sum("detention_minutes").cast("double").alias("detention_minutes_sum"),
        F.avg("detention_minutes").cast("double").alias("detention_minutes_avg")
    )

    # 4. Сборка Time Grid
    metrics_base = metrics.filter(F.col("driver_id").isNotNull() & F.col("month").isNotNull()) \
        .withColumn("month", F.trunc(F.to_date(F.col("month")), "MM")) \
        .select("driver_id", "month", "average_mpg", "average_idle_hours", "on_time_delivery_rate")

    panel = metrics_base \
        .join(driver_month_trips, on=["driver_id", "month"], how="left") \
        .join(detention_features, on=["driver_id", "month"], how="left")

    # Сохраняем sparsity там, где это важно, но базовые счетчики добиваем нулями
    panel = panel.na.fill({
        "trips_cnt": 0.0, "incidents_cnt": 0.0, "unique_trucks": 0.0, "truck_churn_rate": 0.0,
        "detention_minutes_sum": 0.0, "detention_minutes_avg": 0.0, "avg_speed": 0.0
    })

    # Кросс-доменная фича: простои на одну поездку в месяц
    panel = panel.withColumn("detention_per_trip",
        F.when(F.col("trips_cnt") > 0, F.col("detention_minutes_sum") / F.col("trips_cnt")).otherwise(0.0))

    # 5. Оконные функции
    win_driver = Window.partitionBy("driver_id").orderBy("month")

    # Исторические окна (-w до -1)
    for w in [3, 6, 12]:
        win_history = win_driver.rowsBetween(-w, -1)
        panel = panel \
            .withColumn(f"incidents_prev_{w}m", F.sum("incidents_cnt").over(win_history)) \
            .withColumn(f"trips_prev_{w}m", F.sum("trips_cnt").over(win_history)) \
            .withColumn(f"incident_rate_prev_{w}m",
                        F.when(F.col(f"trips_prev_{w}m") > 0,
                               F.col(f"incidents_prev_{w}m") / F.col(f"trips_prev_{w}m")).otherwise(0.0))

    # Флаг недавней активности (Sparsity Signal)
    panel = panel.withColumn("is_active_last_3m", F.when(F.col("trips_prev_3m") > 0, 1.0).otherwise(0.0))

    # Накопительные метрики
    win_lifetime = win_driver.rowsBetween(Window.unboundedPreceding, -1)
    panel = panel \
        .withColumn("lifetime_trips_cnt", F.sum("trips_cnt").over(win_lifetime)) \
        .withColumn("lifetime_incidents_cnt", F.sum("incidents_cnt").over(win_lifetime)) \
        .withColumn("lifetime_incident_rate",
                    F.when(F.col("lifetime_trips_cnt") > 0,
                           F.col("lifetime_incidents_cnt") / F.col("lifetime_trips_cnt")).otherwise(0.0)) \
        .na.fill({"lifetime_trips_cnt": 0.0, "lifetime_incidents_cnt": 0.0})

    # Месяцы с последнего инцидента
    panel = panel.withColumn("month_idx", F.row_number().over(win_driver))
    panel = panel.withColumn("incident_marker",
                             F.when(F.col("incidents_cnt") > 0, F.col("month_idx")).otherwise(F.lit(None)))

    win_last_inc = win_driver.rowsBetween(Window.unboundedPreceding, -1)
    panel = panel.withColumn("last_incident_month_idx", F.last("incident_marker", ignorenulls=True).over(win_last_inc))
    panel = panel.withColumn("months_since_last_incident",
                             F.when(F.col("last_incident_month_idx").isNotNull(),
                                    F.col("month_idx") - F.col("last_incident_month_idx"))
                             .otherwise(F.lit(999.0)))

    # Динамика и Волатильность (Rolling StdDev)
    win_roll_3m = win_driver.rowsBetween(-3, -1)
    panel = panel \
        .withColumn("mpg_roll_3m", F.avg("average_mpg").over(win_roll_3m)) \
        .withColumn("idle_roll_3m", F.avg("average_idle_hours").over(win_roll_3m)) \
        .withColumn("ontime_roll_3m", F.avg("on_time_delivery_rate").over(win_roll_3m)) \
        .withColumn("mpg_std_3m", F.stddev("average_mpg").over(win_roll_3m)) \
        .withColumn("idle_std_3m", F.stddev("average_idle_hours").over(win_roll_3m)) \
        .withColumn("detention_std_3m", F.stddev("detention_minutes_avg").over(win_roll_3m)) \
        .withColumn("mpg_drop_ratio",
                    F.when(F.col("mpg_roll_3m") > 0, F.col("average_mpg") / F.col("mpg_roll_3m")).otherwise(1.0)) \
        .withColumn("idle_spike_ratio",
                    F.when(F.col("idle_roll_3m") > 0, F.col("average_idle_hours") / F.col("idle_roll_3m")).otherwise(1.0)) \
        .withColumn("ontime_drop_ratio",
                    F.when(F.col("ontime_roll_3m") > 0, F.col("on_time_delivery_rate") / F.col("ontime_roll_3m")).otherwise(1.0)) \
        .drop("mpg_roll_3m", "idle_roll_3m", "ontime_roll_3m") \
        .na.fill({"mpg_std_3m": 0.0, "idle_std_3m": 0.0, "detention_std_3m": 0.0})

    # Сезонность
    panel = panel \
        .withColumn("month_num", F.month(F.col("month"))) \
        .withColumn("month_sin", F.sin(2 * 3.1415926535 * F.col("month_num") / 12.0)) \
        .withColumn("month_cos", F.cos(2 * 3.1415926535 * F.col("month_num") / 12.0)) \
        .withColumn("quarter", F.quarter(F.col("month"))) \
        .withColumn("is_high_season", F.when(F.col("quarter") == 4, 1.0).otherwise(0.0)) \
        .drop("month_num")

    # Стаж и опыт
    drivers_features = drivers.filter(F.col("driver_id").isNotNull()) \
        .withColumn("hire_date", F.to_date(F.col("hire_date"))) \
        .select("driver_id", "hire_date", "years_experience")

    panel = panel.join(drivers_features, on="driver_id", how="left") \
        .withColumn("months_in_company",
                    F.when(F.col("hire_date").isNotNull(),
                           F.months_between(F.col("month"), F.col("hire_date")).cast("double"))
                    .otherwise(F.lit(999.0))) \
        .drop("hire_date")

    # 6. ТАРГЕТ СМОТРИТ ВПЕРЕД
    win_forward_3m = win_driver.rowsBetween(1, 3)
    panel = panel.withColumn("future_incidents_sum", F.sum("incidents_cnt").over(win_forward_3m)) \
        .withColumn("has_incident_next_3m", F.when(F.col("future_incidents_sum") > 0, 1.0).otherwise(0.0)) \
        .drop("future_incidents_sum")

    # Фильтрация краев. ИСПОЛЬЗУЕТСЯ ПРИ ОБУЧЕНИИ И ПРАВИЛЬНОМ РАССЧЁТЕ МЕТРИК,
    # УДАЛЕНО ИЗ ФИНАЛЬНОЙ ВЕРСИИ ДЛЯ ПРАВИЛЬНОЙ РАБОТЫ ЗАГРУЗКИ НОВЫХ ДАННЫХ
    # max_month_row = panel.select(F.max("month").alias("max_m")).collect()
    # max_month = max_month_row[0]["max_m"]
    # panel = panel.filter(F.col("month") <= F.add_months(F.lit(max_month), -3))

    # 7. Сохранение
    output_path = f"{output_dir}/spark_features.parquet"
    panel.write.mode("overwrite").parquet(output_path)
    print(f"✓ Матрица фичей успешно сохранена в Parquet: {output_path}")

    spark.stop()

if __name__ == "__main__":
    main()