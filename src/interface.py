import streamlit as st
import psycopg2
import pandas as pd
import os
from datetime import datetime

# Импорт нашей функции пакетной загрузки в Data Lake (Parquet)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
try:
    from src.ingest_batch import ingest_new_csv
except ImportError:
    pass  # Обработка ошибки на случай запуска не из корня

# ========== НАСТРОЙКИ СТРАНИЦЫ ==========
st.set_page_config(
    page_title="LogiRisk | Оценка риска",
    layout="wide"
)

# ========== СТИЛИЗАЦИЯ ИНТЕРФЕЙСА ==========
st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
        h1, h2, h3 { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 700; }

        /* Карточки KPI */
        div[data-testid="stMetric"] {
            background-color: #1E1E2F !important;
            padding: 1.2rem !important;
            border-radius: 10px !important;
            border: 1px solid #2D2D44 !important;
        }
        div[data-testid="stMetric"] label, 
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #FFFFFF !important;
            font-weight: bold !important;
        }
    </style>
""", unsafe_allow_html=True)


# ========== БАЗА ДАННЫХ ==========
def get_pg_connection():
    """Умное подключение: работает и локально, и внутри Docker"""
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=os.getenv('POSTGRES_PORT', '5433'),
        user=os.getenv('POSTGRES_USER', 'admin'),
        password=os.getenv('POSTGRES_PASSWORD', 'admin_password'),
        database=os.getenv('POSTGRES_DB', 'driver_risks')
    )


def get_dashboard_stats():
    """Сбор статистики только за ПОСЛЕДНИЙ месяц"""
    conn = get_pg_connection()
    try:
        df_stats = pd.read_sql("""
                               SELECT COUNT(DISTINCT driver_id) as total,
                                      SUM(display_alert)        as high_risk
                               FROM driver_display_risks
                               WHERE month = (SELECT MAX (month) FROM driver_display_risks)
                               """, conn)
        total = df_stats.iloc[0]['total'] or 0
        high = df_stats.iloc[0]['high_risk'] or 0
    except Exception:
        total, high = 0, 0
    finally:
        conn.close()
    return total, high


def get_all_drivers():
    """Получение витрины водителей только за ПОСЛЕДНИЙ месяц"""
    conn = get_pg_connection()
    try:
        query = """
                SELECT v.driver_id, \
                       COALESCE(d.first_name || ' ' || d.last_name, v.driver_id) as driver_name, \
                       v.display_risk_score                                      as risk_score, \
                       v.display_alert                                           as is_alert, \
                       v.month
                FROM driver_display_risks v
                         LEFT JOIN drivers d ON TRIM(CAST(v.driver_id AS TEXT)) = TRIM(CAST(d.driver_id AS TEXT))
                WHERE v.month = (SELECT MAX(month) FROM driver_display_risks)
                ORDER BY v.display_alert DESC, v.display_risk_score DESC; \
                """
        df = pd.read_sql(query, conn)

        if not df.empty:
            df.insert(0, "№", range(1, len(df) + 1))

            df["Крит. Сигнал"] = df["is_alert"].map({1: "🚨", 0: "—"})
            df["Индекс Риска"] = df["risk_score"].round(1)

            # Оставляем только нужные колонки (без колонки "Грейд")
            df = df[["№", "driver_id", "driver_name", "month", "Индекс Риска", "Крит. Сигнал"]]

            # Переименовываем всё на чистый русский
            df = df.rename(columns={
                "driver_id": "ID Водителя",
                "driver_name": "ФИО Водителя",
                "month": "Отчётный месяц"
            })

        return df

    except Exception as e:
        st.error(f"Ошибка при чтении данных из базы: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def add_cooldown_event(driver_id, hours=720):  # 720 часов = 30 дней затухания
    conn = get_pg_connection()
    cursor = conn.cursor()
    try:
        query = """
                INSERT INTO driver_cooldowns (driver_id, checked_at, cooldown_until)
                VALUES (%s, NOW(), NOW() + INTERVAL '%s hour') ON CONFLICT (driver_id) 
            DO \
                UPDATE SET checked_at = NOW(), cooldown_until = NOW() + INTERVAL '%s hour'; \
                """
        cursor.execute(query, (driver_id, hours, hours))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Ошибка при сохранении кулдауна: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


# ========== ОТРИСОВКА ИНТЕРФЕЙСА ==========
st.title("LogiRisk | Инструмент ранжирования водителей по риску инцидентов")

total_drv, high_risk_drv = get_dashboard_stats()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Водителей на контроле", f"{total_drv:,}")
with col2:
    st.metric("Критический уровень (Alerts)", int(high_risk_drv))
with col3:
    st.metric("Синхронизация витрины", datetime.now().strftime("%d.%m.%Y %H:%M"))

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Рейтинг рисков", "🛡️ Панель корректировок", "📁 Загрузка логов (Data Lake)"])

# ----- ВКЛАДКА 1: ОСНОВНОЙ РЕЙТИНГ -----
with tab1:
    st.header("Список водителей (Smart Ranking)")
    df_all = get_all_drivers()

    if df_all.empty:
        st.warning("В базе PostgreSQL пока нет данных. Запустите пайплайн Airflow.")
    else:
        st.dataframe(
            df_all,
            use_container_width=True,
            hide_index=True,
            column_config={
                "№": st.column_config.NumberColumn(
                    "№",
                    width="small"
                ),
                "ID Водителя": st.column_config.TextColumn(
                    "ID Водителя",
                    width="small"
                ),
                "ФИО Водителя": st.column_config.TextColumn(
                    "ФИО Водителя",
                    width="medium"
                ),
                "Отчётный месяц": st.column_config.DateColumn(
                    "Отчётный месяц",
                    format="YYYY-MM",
                    width="small"
                ),
                "Индекс Риска": st.column_config.ProgressColumn(
                    "Индекс Риска (0-100)",
                    format="%.1f",
                    min_value=0.0,
                    max_value=100.0,
                    width="medium"
                ),
                "Крит. Сигнал": st.column_config.TextColumn(
                    "Крит. Сигнал",
                    width="small"
                )
            }
        )

# ----- ВКЛАДКА 2: ПАНЕЛЬ КОРРЕКТИРОВОК -----
with tab2:
    st.header("Регистрация профилактических мероприятий")
    st.markdown("Фиксация беседы активирует 30-дневное плавное затухание риска.")

    conn = get_pg_connection()
    try:
        # Выводим ВСЕХ водителей за последний месяц, сортируя от самых опасных к безопасным.
        df_for_select = pd.read_sql("""
                                    SELECT driver_id
                                    FROM driver_display_risks
                                    WHERE month = (SELECT MAX (month) FROM driver_display_risks)
                                    ORDER BY display_risk_score DESC;
                                    """, conn)
    except Exception:
        df_for_select = pd.DataFrame()
    finally:
        conn.close()

    if df_for_select.empty:
        st.info("В текущем месяце нет данных по водителям.")
    else:
        selected_driver = st.selectbox(
            "Выберите ID водителя для фиксации инструктажа:",
            options=df_for_select["driver_id"].tolist()
        )

        comment = st.text_area(
            "Детали проведенной работы:",
            placeholder="Например: Проведен внеплановый инструктаж ПДД, водитель предупрежден..."
        )

        if st.button("Зафиксировать мероприятие", type="primary"):
            if add_cooldown_event(selected_driver):
                # Строгое, корпоративное уведомление об успехе
                st.success(
                    f"Системный статус обновлен. Риск водителя {selected_driver} отправлен в фазу затухания на 30 дней.")

# ----- ВКЛАДКА 3: ЗАГРУЗКА НОВЫХ ЛОГОВ (DATA LAKE) -----
with tab3:
    st.header("Загрузка данных в Data Lake (Parquet Store)")
    st.markdown("Пакетный инкрементальный импорт операционных логов для Apache Spark.")

    target_table = st.selectbox(
        "Выберите целевой набор данных:",
        options=["trips", "safety_incidents", "delivery_events", "driver_monthly_metrics", "drivers"]
    )

    uploaded_file = st.file_uploader("Загрузите файл .csv от телематики", type=["csv"])

    if uploaded_file is not None:
        df_preview = pd.read_csv(uploaded_file)
        st.write("📋 **Предпросмотр (первые 3 строки):**")
        st.dataframe(df_preview.head(3), use_container_width=True)

        if st.button("🚀 Отправить в Data Lake", type="primary"):
            with st.spinner("Склеиваем историю и обновляем Parquet-архив..."):
                try:
                    rows = ingest_new_csv(uploaded_file, target_table)
                    st.success(f"Успешно! {rows} строк сжато и добавлено в {target_table}.parquet!")
                    st.info("Данные на диске. Теперь Airflow может запускать Spark ETL без перегрузки памяти.")
                except Exception as e:
                    st.error(f"Ошибка при обновлении Data Lake: {e}")