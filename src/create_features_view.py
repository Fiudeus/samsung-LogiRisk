import clickhouse_connect
import os


def create_view():
    print("Подключение к ClickHouse...")

    ch_host = os.getenv('CLICKHOUSE_HOST', '127.0.0.1')
    client = clickhouse_connect.get_client(host=ch_host, port=8123, username='admin', password='admin')

    # Сносим старое, чтобы не было конфликтов
    client.command("DROP VIEW IF EXISTS default.driver_features_view")

    query = """
            CREATE VIEW default.driver_features_view AS
            WITH trip_aggs AS (SELECT driver_id, \
                                      toStartOfMonth(dispatch_date) AS month,
                count (trip_id \
            ) AS trips_cnt,
                uniqExact(truck_id \
            ) AS unique_trucks
                FROM default.trips
                GROUP BY driver_id, month
            ),
                inc_aggs AS (
                SELECT
                driver_id,
                toStartOfMonth(incident_date \
            ) AS month,
                count ( \
            ) AS incidents_cnt
                FROM default.safety_incidents
                GROUP BY driver_id, month
            ),
                det_aggs AS (
                SELECT
                t.driver_id,
                toStartOfMonth(t.dispatch_date \
            ) AS month,
                sum (d.detention_minutes \
            ) AS detention_minutes_sum,
                avg (d.detention_minutes \
            ) AS detention_minutes_avg
                FROM default.trips t
                JOIN default.delivery_events d ON t.trip_id = d.trip_id
                GROUP BY t.driver_id, month
            ),
                base AS (
                SELECT
                m.driver_id AS driver_id,
                m.month AS month,
                m.trips_completed AS trips_completed,
                m.total_miles AS total_miles,
                m.total_revenue AS total_revenue,
                m.average_mpg AS average_mpg,
                m.total_fuel_gallons AS total_fuel_gallons,
                m.on_time_delivery_rate AS on_time_delivery_rate,
                m.average_idle_hours AS average_idle_hours,

                -- Подтягиваем агрегаты
                coalesce (ta.trips_cnt, 0 \
            ) AS trips_cnt,
                coalesce (ta.unique_trucks, 0 \
            ) AS unique_trucks,
                coalesce (ia.incidents_cnt, 0 \
            ) AS incidents_cnt,
                coalesce (da.detention_minutes_sum, 0 \
            ) AS detention_minutes_sum,
                coalesce (da.detention_minutes_avg, 0 \
            ) AS detention_minutes_avg,

                if(coalesce (ta.trips_cnt, 0 \
            ) > 0, coalesce (ta.unique_trucks, 0 \
            ) / coalesce (ta.trips_cnt, 0 \
            ), 0 \
            ) AS truck_churn_rate,

                -- Сезонность
                sin(2 * pi( \
            ) * toMonth(m.month \
            ) / 12 \
            ) AS month_sin,
                cos(2 * pi( \
            ) * toMonth(m.month \
            ) / 12 \
            ) AS month_cos,
                toQuarter(m.month \
            ) AS quarter,
                if(toQuarter(m.month \
            ) = 4, 1, 0 \
            ) AS is_high_season,

                -- Месяцы в компании
                coalesce (dateDiff('month', drv.hire_date, m.month \
            ), 999 \
            ) AS months_in_company \
                FROM default.driver_monthly_metrics m
                LEFT JOIN trip_aggs ta ON m.driver_id = ta.driver_id AND m.month = ta.month
                LEFT JOIN inc_aggs ia ON m.driver_id = ia.driver_id AND m.month = ia.month
                LEFT JOIN det_aggs da ON m.driver_id = da.driver_id AND m.month = da.month
                LEFT JOIN default.drivers drv ON m.driver_id = drv.driver_id
            ),
                windows AS (
                SELECT
                *,
                -- История 3, 6, 12 месяцев (ИНЛАЙН ОКНА)
                sum (incidents_cnt \
            ) OVER (PARTITION BY driver_id ORDER BY month ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING \
            ) AS incidents_prev_3m,
                sum (trips_cnt \
            ) OVER (PARTITION BY driver_id ORDER BY month ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING \
            ) AS trips_prev_3m,
                avg (average_mpg \
            ) OVER (PARTITION BY driver_id ORDER BY month ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING \
            ) AS mpg_roll_3m,
                avg (average_idle_hours \
            ) OVER (PARTITION BY driver_id ORDER BY month ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING \
            ) AS idle_roll_3m,
                avg (on_time_delivery_rate \
            ) OVER (PARTITION BY driver_id ORDER BY month ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING \
            ) AS ontime_roll_3m, \
                sum (incidents_cnt \
            ) OVER (PARTITION BY driver_id ORDER BY month ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING \
            ) AS incidents_prev_6m,
                sum (trips_cnt \
            ) OVER (PARTITION BY driver_id ORDER BY month ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING \
            ) AS trips_prev_6m, \
                sum (incidents_cnt \
            ) OVER (PARTITION BY driver_id ORDER BY month ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING \
            ) AS incidents_prev_12m,
                sum (trips_cnt \
            ) OVER (PARTITION BY driver_id ORDER BY month ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING \
            ) AS trips_prev_12m,

                -- Лайфтайм (за все время работы)
                sum (trips_cnt \
            ) OVER (PARTITION BY driver_id ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING \
            ) AS lifetime_trips_cnt,
                sum (incidents_cnt \
            ) OVER (PARTITION BY driver_id ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING \
            ) AS lifetime_incidents_cnt,

                -- Последний инцидент (с защитой от NULL)
                max (if(incidents_cnt > 0, month, toDate32('1970-01-01' \
            ) \
            ) \
            ) OVER (PARTITION BY driver_id ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING \
            ) AS last_inc_month,

                -- Таргет (заглядываем в будущее)
                sum (incidents_cnt \
            ) OVER (PARTITION BY driver_id ORDER BY month ROWS BETWEEN 1 FOLLOWING AND 3 FOLLOWING \
            ) AS future_incidents \
                FROM base
            )
            SELECT
                -- 1. Базовые фичи
                driver_id, month, trips_completed, total_miles, total_revenue, average_mpg, total_fuel_gallons, on_time_delivery_rate, average_idle_hours, trips_cnt, incidents_cnt, unique_trucks, truck_churn_rate, detention_minutes_sum, detention_minutes_avg,

                -- 2. Оконные агрегаты и рейты
                coalesce (incidents_prev_3m, 0) AS incidents_prev_3m, coalesce (trips_prev_3m, 0) AS trips_prev_3m, if(trips_prev_3m > 0, incidents_prev_3m / trips_prev_3m, 0) AS incident_rate_prev_3m, coalesce (incidents_prev_6m, 0) AS incidents_prev_6m, coalesce (trips_prev_6m, 0) AS trips_prev_6m, if(trips_prev_6m > 0, incidents_prev_6m / trips_prev_6m, 0) AS incident_rate_prev_6m, coalesce (incidents_prev_12m, 0) AS incidents_prev_12m, coalesce (trips_prev_12m, 0) AS trips_prev_12m, if(trips_prev_12m > 0, incidents_prev_12m / trips_prev_12m, 0) AS incident_rate_prev_12m, coalesce (lifetime_trips_cnt, 0) AS lifetime_trips_cnt, coalesce (lifetime_incidents_cnt, 0) AS lifetime_incidents_cnt, if(lifetime_trips_cnt > 0, lifetime_incidents_cnt / lifetime_trips_cnt, 0) AS lifetime_incident_rate,

                -- 3. Хитрые расчеты времени (защита от базовой даты 1970)
                if(toYear(last_inc_month) <= 1970, 999, dateDiff('month', last_inc_month, month)) AS months_since_last_incident, months_in_company,

                -- 4. Сезонность
                month_sin, month_cos, quarter, is_high_season,

                -- 5. Динамика эффективности
                if(mpg_roll_3m > 0, average_mpg / mpg_roll_3m, 1.0) AS mpg_drop_ratio, if(idle_roll_3m > 0, average_idle_hours / idle_roll_3m, 1.0) AS idle_spike_ratio, if(ontime_roll_3m > 0, on_time_delivery_rate / ontime_roll_3m, 1.0) AS ontime_drop_ratio,

                -- 6. Целевая переменная
                if(coalesce (future_incidents, 0) > 0, 1, 0) AS has_incident_next_3m

            FROM windows
            ORDER BY driver_id, month; \
            """

    client.command(query)
    print("Витрина driver_features_view со всеми фичами успешно создана!")


if __name__ == "__main__":
    create_view()