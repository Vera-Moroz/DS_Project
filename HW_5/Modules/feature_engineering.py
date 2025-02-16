import pandas as pd
from sklearn.preprocessing import StandardScaler

def calculate_amount(df):
    """
    Расчитывает столбец 'Amount' как произведение 'Quantity' и 'UnitPrice'.
    """
    df['Amount'] = df['Quantity'] * df['UnitPrice']
    return df

def create_invoice_features(df):
    """
    Создает фичи для каждого чека на основе группировки по InvoiceNo.
    """
    
    invoice_features = df.groupby('InvoiceNo').agg(
        TotalAmount=('Amount', 'sum'),
        MinUnitPrice=('UnitPrice', 'min'),
        MaxUnitPrice=('UnitPrice', 'max'),
        MedianUnitPrice=('UnitPrice', 'median'),
        MinQuantity=('Quantity', 'min'),
        MaxQuantity=('Quantity', 'max'),
        TotalQuantity=('Quantity', 'sum'),
        InvoiceDate=('InvoiceDate', 'first'),  # Предполагаем, что дата чека одинакова для всех строк в чеке
        CustomerID=('CustomerID', 'first')  # Предполагаем, что CustomerID одинаков для всех строк в чеке
    ).reset_index()
    
    # Удаление ненужных полей
    df = df.drop(columns=['StockCode', 'Description', 'IsReturn'], errors='ignore')
    
    return invoice_features

def create_customer_features(df):
    """
    Создает фичи для каждого абонента на основе группировки по CustomerID.
    """
    # Преобразование столбца 'InvoiceDate' в тип datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    customer_features = df.groupby('CustomerID').agg(
        FirstPurchaseDate=('InvoiceDate', 'min'),
        LastPurchaseDate=('InvoiceDate', 'max'),
        TotalPurchaseCount=('InvoiceNo', 'nunique'),
        TotalPurchaseAmount=('TotalAmount', 'sum'),
        AveragePurchaseAmount=('TotalAmount', 'mean'),
        TotalQuantity=('TotalQuantity', 'sum')
    ).reset_index()
    
    # Вычисляем количество дней с момента первой покупки
    customer_features['DaysSinceFirstPurchase'] = (pd.to_datetime(df['InvoiceDate']).max() - customer_features['FirstPurchaseDate']).dt.days
    
    # Сумма покупок за последний месяц
    one_month_ago = pd.to_datetime(df['InvoiceDate']).max() - pd.DateOffset(months=1)
    last_month_purchases = df[df['InvoiceDate'] >= one_month_ago].groupby('CustomerID').agg(
        LastMonthPurchaseAmount=('TotalAmount', 'sum')
    ).reset_index()
    
    customer_features = customer_features.merge(last_month_purchases, on='CustomerID', how='left')
    customer_features['LastMonthPurchaseAmount'] = customer_features['LastMonthPurchaseAmount'].fillna(0)
    
    return customer_features

def create_date_features(df):
    """
    Создает фичи на основе группировки по дате (без времени).
    """
    # Создание столбца 'Date' без времени
    df['Date'] = df['InvoiceDate'].dt.date
    
    # Группировка по дате и создание фичей
    date_features = df.groupby('Date').agg(
        TotalSalesAmount=('Amount', 'sum'),  # Общая сумма продаж за день
        TotalSalesQuantity=('Quantity', 'sum'),  # Общее количество товаров, проданных за день
        AverageSalesAmount=('Amount', 'mean'),  # Средняя сумма продаж за день
        AverageSalesQuantity=('Quantity', 'mean'),  # Среднее количество товаров, проданных за день
        NumberOfTransactions=('InvoiceNo', 'nunique'),  # Количество транзакций за день
        DayOfWeek=('InvoiceDate', lambda x: x.dt.dayofweek.mode()[0]),  # День недели (0 = понедельник, 1 = вторник и т.д.)
        WeekOfYear=('InvoiceDate', lambda x: x.dt.isocalendar().week.mode()[0]),  # Номер недели в году
        MonthOfYear=('InvoiceDate', lambda x: x.dt.month.mode()[0])  # Месяц года
    ).reset_index()

    return date_features

def normalize_features(df, columns):
    """
    Нормализует указанные фичи в DataFrame.
    """
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df
