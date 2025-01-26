import pandas as pd
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
import psycopg2

# Глобальная переменная для хранения состояния подключения
connection_established = False

# Функция для загрузки переменных окружения
def load_env(env_path=None):
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

# Функция для подключения к базе данных
def connect_to_db():
    global connection_established
    try:
        db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        engine = create_engine(db_url)
        if not connection_established:
            print("Подключение к базе данных успешно установлено")
            connection_established = True
        return engine
    except Exception as e:
        print(f"Ошибка подключения к базе данных: {e}")
        return None

# Функция для выполнения SQL-запросов
def execute_query(query):
    engine = connect_to_db()
    if engine:
        try:
            df = pd.read_sql_query(query, engine)
            return df
        except Exception as e:
            print(f"Ошибка выполнения запроса для {query}: {e}")
            return None

# Функция для создания таблицы и загрузки данных из CSV
def create_and_load_table(csv_file_path):
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )
        cursor = conn.cursor()
        cursor.execute("""
            DROP TABLE IF EXISTS public."Ecommerce";
            CREATE TABLE public."Ecommerce" (
                "Customer ID" INTEGER,
                "Purchase Date" DATE,
                "Product Category" TEXT,
                "Product Price" INTEGER,
                "Quantity" INTEGER,
                "Total Purchase Amount" INTEGER,
                "Payment Method" TEXT,
                "Customer Age" INTEGER,
                "Returns" INTEGER,
                "Customer Name" TEXT,
                "Age" INTEGER,
                "Gender" TEXT,
                "Churn" INTEGER
            );
        """)
        conn.commit()
        cursor.close()
        print("Table created successfully")

        # Используем метод COPY для загрузки данных из CSV-файла
        with open(csv_file_path, 'r') as f:
            cursor = conn.cursor()
            cursor.copy_expert("""
                COPY public."Ecommerce" ("Customer ID", "Purchase Date", "Product Category", "Product Price", "Quantity",
                "Total Purchase Amount", "Payment Method", "Customer Age", "Returns", "Customer Name", "Age", "Gender", "Churn")
                FROM STDIN WITH CSV HEADER DELIMITER ',' QUOTE '"' ESCAPE ''''
            """, f)
            conn.commit()
            cursor.close()
        conn.close()
        print("Data loaded successfully")
    except Exception as e:
        print(f"Error creating and loading table: {e}")
