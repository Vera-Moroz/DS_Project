import pandas as pd
from db_utils import execute_query

# Функция для получения количества видов товаров
def get_product_types_qty():
    query = """
    SELECT COUNT(*) AS product_types_qty
    FROM (
        SELECT DISTINCT "Product Category", "Product Price"
        FROM public."Ecommerce"
    ) AS unique_pairs;
    """
    result = execute_query(query)
    return result.iloc[0]['product_types_qty'] if result is not None else None

# Функция для получения количества категорий товаров
def get_product_categories_qty():
    query = "SELECT COUNT(DISTINCT \"Product Category\") AS product_categories_qty FROM public.\"Ecommerce\";"
    result = execute_query(query)
    return result.iloc[0]['product_categories_qty'] if result is not None else None

# Функция для получения общего количества товаров
def get_total_goods_quantity():
    query = "SELECT SUM(\"Quantity\") AS total_goods_quantity FROM public.\"Ecommerce\";"
    result = execute_query(query)
    return result.iloc[0]['total_goods_quantity'] if result is not None else None

# Функция для получения общей суммы товаров
def get_total_goods_cost():
    query = "SELECT SUM(\"Product Price\" * \"Quantity\") AS total_goods_cost FROM public.\"Ecommerce\";"
    result = execute_query(query)
    return result.iloc[0]['total_goods_cost'] if result is not None else None

# Функция для получения количества уникальных клиентов
def get_unique_customers():
    query = "SELECT COUNT(DISTINCT \"Customer ID\") AS unique_customers FROM public.\"Ecommerce\";"
    result = execute_query(query)
    return result.iloc[0]['unique_customers'] if result is not None else None

# Функция для получения количества покупок
def get_total_purchases():
    query = "SELECT COUNT(*) AS total_purchases FROM public.\"Ecommerce\";"
    result = execute_query(query)
    return result.iloc[0]['total_purchases'] if result is not None else None

# Функция для получения общей суммы покупок
def get_total_purchases_amount():
    query = "SELECT SUM(\"Total Purchase Amount\") AS total_purchases_amount FROM public.\"Ecommerce\";"
    result = execute_query(query)
    return result.iloc[0]['total_purchases_amount'] if result is not None else None

# Функция для получения метрик по категориям
def get_category_metrics():
    try:
        query = """
        SELECT "Product Category",
               COUNT(*) AS "Количество",
               AVG("Product Price") AS "Средняя цена",
               STDDEV("Product Price") AS "Стандартное отклонение",
               MIN("Product Price") AS "Минимальная цена",
               PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY "Product Price") AS "Медианная цена",
               MAX("Product Price") AS "Максимальная цена"
        FROM public."Ecommerce"
        GROUP BY "Product Category"
        """
        result = execute_query(query)

        if result is not None:
            total_row = {
                "Product Category": "Итого",
                "Количество": result["Количество"].sum(),
                "Средняя цена": result["Средняя цена"].mean(),
                "Стандартное отклонение": result["Стандартное отклонение"].mean(),
                "Минимальная цена": result["Минимальная цена"].min(),
                "Медианная цена": result["Медианная цена"].median(),
                "Максимальная цена": result["Максимальная цена"].max()
            }
            result = pd.concat([result, pd.DataFrame([total_row])], ignore_index=True)
            result = result.set_index("Product Category").T
        return result
    except Exception as e:
        print(f"Error in get_category_metrics: {e}")
        return None

# Функция для получения среза базы данных
def get_data_summary():
    product_types_qty = get_product_types_qty()
    product_categories_qty = get_product_categories_qty()
    total_goods_quantity = get_total_goods_quantity()
    total_goods_cost = get_total_goods_cost()
    unique_customers = get_unique_customers()
    total_purchases = get_total_purchases()
    total_purchases_amount = get_total_purchases_amount()

    summary = (
        f"Продуктовый каталог содержит {product_types_qty:,} видов товаров по {product_categories_qty:,} категориям "
        f"\nв общем количестве {total_goods_quantity:,} штук на сумму {total_goods_cost:,} руб. "
        f"\nВ базе зарегистрировано {unique_customers:,} уникальных клиентов, "
        f"\nсовершивших {total_purchases:,} покупок на {total_purchases_amount:,} руб."
    )
    return summary
