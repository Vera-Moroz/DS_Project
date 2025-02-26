# eda.py

import pandas as pd

def perform_eda(df):
    """
    Выполняет анализ данных (EDA) для переданного DataFrame.
    """
    # 1. Поля и типы данных
    print("\n**Поля и типы данных:**\n")
    print(df.info())

    # 2. Примеры значений (первые строки)
    print("\n**Первые строки данных:**\n")
    print(df.head())

    # 3. Статистический срез данных
    print("\n**Статистический срез данных:**\n")
    print(df.describe())

    # 4. Проверка пропущенных значений
    print("\n**Проверка пропущенных значений:**\n")
    print(df.isnull().sum())

    # Дополнительный анализ
    print("\n**Количество уникальных значений в столбцах:**\n")
    print(f"Уникальных стран: {df['Country'].nunique()}")
    print(f"Уникальных клиентов: {df['CustomerID'].nunique()}")
    print(f"Уникальных описаний товаров: {df['Description'].nunique()}")

    # Количество заказов и возвратов
    print("\n**Количество заказов и возвратов:**\n")
    total_invoices = df['InvoiceNo'].nunique()
    cancellations = df[df['InvoiceNo'].str.startswith('C')]['InvoiceNo'].nunique()
    print(f"Всего заказов: {total_invoices}")
    print(f"Возвратов: {cancellations}")

    # Проверка временного интервала данных
    min_date = pd.to_datetime(df['InvoiceDate']).min()
    max_date = pd.to_datetime(df['InvoiceDate']).max()
    print(f"\n**Временной интервал данных:**\n")
    print(f"Минимальная дата в датасете: {min_date}")
    print(f"Максимальная дата в датасете: {max_date}")

def compare_statistics(df1, df2):
    """
    Сравнение статистических срезов двух DataFrame и вывод краткого заключения.
    df1 - меньший датасет (например, без пропусков)
    df2 - больший датасет (например, весь датасет целиком)
    """
    description1 = df1.describe(include='all')
    description2 = df2.describe(include='all')
    
    significant_diff = {}
    exclude_columns = ['CustomerID', 'Country', 'CustomerID_missing']
    
    for col in description1.columns:
        if col in description2.columns and col not in exclude_columns:
            try:
                if pd.api.types.is_numeric_dtype(description1[col]):
                    mean_diff = round(description1.at['mean', col] - description2.at['mean', col], 2)
                    std_diff = round(description1.at['std', col] - description2.at['std', col], 2)
                    mean_diff_percent = round((mean_diff / description2.at['mean', col]) * 100)
                    std_diff_percent = round((std_diff / description2.at['std', col]) * 100)
                    
                    mean_diff_percent_str = f"+{mean_diff_percent}%" if mean_diff_percent > 0 else f"{mean_diff_percent}%"
                    std_diff_percent_str = f"+{std_diff_percent}%" if std_diff_percent > 0 else f"{std_diff_percent}%"

                    if abs(mean_diff_percent) > 10 or abs(std_diff_percent) > 10:
                        significant_diff[col] = {
                            'mean_diff': mean_diff,
                            'mean_diff_percent': mean_diff_percent_str,
                            'std_diff': std_diff,
                            'std_diff_percent': std_diff_percent_str
                        }
            except Exception:
                continue
    
    return significant_diff

def print_comparison_results(df1, df2):
    """
    Печать результатов сравнения статистических срезов двух DataFrame.
    df1 - меньший датасет (например, без пропусков)
    df2 - больший датасет (например, весь датасет целиком)
    """
    print("\nЗапуск compare_statistics для сравнения df1 и df2:")
    differences = compare_statistics(df1, df2)

    print("\nНа какие показатели серьезно повлиял выброс пропусков:")
    if differences:
        for col, diff in differences.items():
            print(f"{col}: разница в среднем значении = {diff['mean_diff']} / {diff['mean_diff_percent']}, разница в стандартном отклонении = {diff['std_diff']} / {diff['std_diff_percent']}")
    else:
        print("Существенных различий не обнаружено.")

def analyze_missing_values(df):
    """
    Анализ данных с пропущенными значениями и без них, сравнение статистических срезов.
    """
    # Данные без пропусков в CustomerID
    no_missing = df[df['CustomerID_missing'] == 0]
    print("\nАнализ данных без пропусков в CustomerID:")
    print("Количество строк в df_no_missing:", len(no_missing))
    perform_eda(no_missing)

    # Сравнение статистических срезов и вывод результатов
    print("\nСравнение статистических срезов данных без пропусков и данных целиком:")

    print("\n**Без пропусков:**\n")
    print(no_missing.describe())

    print("\n**С пропусками:**\n")
    print(df.describe())

    print_comparison_results(no_missing, df)

def analyze_returns(df):
    """
    Анализ доли возвратов в датасете, подсчет количества и суммы возвратов и продаж.
    """
    # Создание копии DataFrame, чтобы избежать SettingWithCopyWarning
    df = df.copy()

    # Разделение на возвраты и продажи
    df['IsReturn'] = df['InvoiceNo'].apply(lambda x: x.startswith('C'))
    print("Столбец IsReturn добавлен:")
    print(df.head())

    df_returns = df[df['IsReturn'] == True]
    df_sales = df[df['IsReturn'] == False]

    # Подсчет количества и суммы возвратов
    num_returns = df_returns.shape[0]
    sum_returns = (df_returns['Quantity'] * df_returns['UnitPrice']).sum()

    # Подсчет количества и суммы продаж
    num_sales = df_sales.shape[0]
    sum_sales = (df_sales['Quantity'] * df_sales['UnitPrice']).sum()

    # Расчет процента возвратов по количеству и по сумме
    percent_returns_quantity = (num_returns / (num_returns + num_sales)) * 100
    percent_returns_amount = (sum_returns / (sum_returns + sum_sales)) * 100

    print(f"Количество возвратов: {num_returns}")
    print(f"Сумма возвратов: {sum_returns:.2f}")
    print(f"Количество продаж: {num_sales}")
    print(f"Сумма продаж: {sum_sales:.2f}")
    print(f"Процент возвратов по количеству: {percent_returns_quantity:.2f}%")
    print(f"Процент возвратов по сумме: {percent_returns_amount:.2f}%")
    
    return df_sales
