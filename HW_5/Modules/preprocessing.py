# preprocessing.py

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_extraction import FeatureHasher
from Modules.feature_engineering import calculate_amount

def preprocess_data(df):
    """
    Предобработка данных: создание индикаторной переменной для пропусков в CustomerID, 
    присвоение пропускам индекса 0 при энкодировании и добавление столбца Amount.
    """
    # Создание копии исходного DataFrame
    df1 = df.copy()
    
    # Преобразование столбца 'InvoiceDate' из типа 'object' в 'datetime64'
    df1['InvoiceDate'] = pd.to_datetime(df1['InvoiceDate'])

    # Создание индикаторной переменной для пропусков в 'CustomerID'
    df1['CustomerID_missing'] = df1['CustomerID'].isnull().astype(int)

    # Присвоение пропускам в 'CustomerID' значения 0
    df1['CustomerID'] = df1['CustomerID'].fillna(0)

    # Преобразование категориальных переменных и сохранение соответствий
    label_encoders = {}
    for column in ['Country', 'CustomerID', 'Description']:
        unique_values = df1[column].nunique()
        if unique_values < 10000:
            label_encoder = LabelEncoder()
            df1[column] = label_encoder.fit_transform(df1[column].astype(str))
            label_encoders[column] = {index: label for index, label in enumerate(label_encoder.classes_)}
        else:
            hasher = FeatureHasher(input_type='string', n_features=10)
            hashed_features = hasher.transform(df1[column].astype(str))
            df1 = df1.drop(columns=[column])
            for i in range(hashed_features.shape[1]):
                df1[f'{column}_hash_{i}'] = hashed_features[:, i].toarray()

    # Сохранение соответствий категориальных переменных
    save_label_encodings(label_encoders)
    
    return df1, label_encoders

def save_label_encodings(label_encoders):
    """
    Сохранение соответствий категориальных переменных в файл с кодировкой utf-8.
    """
    data_dir = os.path.join(os.getcwd(), 'Data')
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, 'label_encodings.txt')

    with open(file_path, 'w', encoding='utf-8') as file:
        for column, mapping in label_encoders.items():
            file.write(f"{column}:\n")
            for index, label in mapping.items():
                file.write(f"{index}: {label}\n")
            file.write("\n")
    print(f"\nТрансформация данных и энкодирование категориальных переменных завершены. Файл сохранен в {file_path}.")

def clean_data(df):
    """
    Очищает данные, выполняя следующие шаги:
    1. Поиск и удаление инвойсов с отрицательными значениями Quantity.
    2. Поиск и удаление инвойсов с отрицательными значениями UnitPrice.
    3. Вывод и удаление инвойсов, не соответствующих условию "a 6-digit integral number".
    4. Вывод и удаление инвойсов с разными InvoiceDate или CustomerID для одного InvoiceNo.
    """
    # Временное вычисление столбца 'Amount'
    df = calculate_amount(df)
    
    total_quantity = abs(df['Quantity'].sum())
    total_amount = abs(df['Amount'].sum())
    total_removed_quantity = 0
    total_removed_amount = 0
    print(f"\nОбщее количество товаров до очистки: {total_quantity}")
    print(f"Общая сумма продаж до очистки: {total_amount:.2f}")
    
    # 1. Поиск и удаление инвойсов с отрицательными значениями Quantity
    negative_quantity_invoices = df[df['Quantity'] < 0]['InvoiceNo'].unique()
    if len(negative_quantity_invoices) > 0:
        print(f"\nИнвойсы с отрицательными значениями Quantity:")
        print(negative_quantity_invoices)
        print(f"Количество удаленных инвойсов: {len(negative_quantity_invoices)}")
        
        # Удаление всех данных по инвойсу, если хоть одна строка бракованная
        removed_quantity = abs(df[df['InvoiceNo'].isin(negative_quantity_invoices)]['Quantity'].sum())
        removed_amount = abs(df[df['InvoiceNo'].isin(negative_quantity_invoices)]['Amount'].sum())
        total_removed_quantity += removed_quantity
        total_removed_amount += removed_amount
        percent_removed_quantity = removed_quantity / total_quantity * 100
        percent_removed_amount = removed_amount / total_amount * 100
        
        print(f"Количество удаленых товаров: {removed_quantity}")
        print(f"Доля от общего количества товаров: {percent_removed_quantity:.2f}%")
        print(f"Сумма удаленых продаж: {removed_amount:.2f}")
        print(f"Доля от общей суммы продаж: {percent_removed_amount:.2f}%")
        
        df_clean = df[~df['InvoiceNo'].isin(negative_quantity_invoices)]
    else:
        df_clean = df.copy()

    # 2. Поиск и удаление инвойсов с отрицательными значениями UnitPrice
    negative_unitprice_invoices = df_clean[df_clean['UnitPrice'] < 0]['InvoiceNo'].unique()
    if len(negative_unitprice_invoices) > 0:
        print(f"\nИнвойсы с отрицательными значениями UnitPrice:")
        print(negative_unitprice_invoices)
        print(f"Количество удаленных инвойсов: {len(negative_unitprice_invoices)}")
        
        # Удаление всех данных по инвойсу, если хоть одна строка бракованная
        removed_quantity = abs(df[df['InvoiceNo'].isin(negative_unitprice_invoices)]['Quantity'].sum())
        removed_amount = abs(df[df['InvoiceNo'].isin(negative_unitprice_invoices)]['Amount'].sum())
        total_removed_quantity += removed_quantity
        total_removed_amount += removed_amount
        percent_removed_quantity = removed_quantity / total_quantity * 100
        percent_removed_amount = removed_amount / total_amount * 100
        
        print(f"Количество удаленых товаров: {removed_quantity}")
        print(f"Доля от общего количества товаров: {percent_removed_quantity:.2f}%")
        print(f"Сумма удаленых продаж: {removed_amount:.2f}")
        print(f"Доля от общей суммы продаж: {percent_removed_amount:.2f}%")
        
        df_clean = df_clean[~df_clean['InvoiceNo'].isin(negative_unitprice_invoices)]

    # 3. Вывод и удаление инвойсов, не соответствующих условию "a 6-digit integral number"
    non_conforming_invoices = df_clean[~df_clean['InvoiceNo'].str.match(r'^\d{6}$')]['InvoiceNo'].unique()
    if len(non_conforming_invoices) > 0:
        print(f"\nИнвойсы, не соответствующие условию 'a 6-digit integral number':")
        print(non_conforming_invoices)
        print(f"Количество удаленных инвойсов: {len(non_conforming_invoices)}")
        
        # Удаление всех данных по инвойсу, если хоть одна строка бракованная
        removed_quantity = abs(df[df['InvoiceNo'].isin(non_conforming_invoices)]['Quantity'].sum())
        removed_amount = abs(df[df['InvoiceNo'].isin(non_conforming_invoices)]['Amount'].sum())
        total_removed_quantity += removed_quantity
        total_removed_amount += removed_amount
        percent_removed_quantity = removed_quantity / total_quantity * 100
        percent_removed_amount = removed_amount / total_amount * 100
        
        print(f"Количество удаленых товаров: {removed_quantity}")
        print(f"Доля от общего количества товаров: {percent_removed_quantity:.2f}%")
        print(f"Сумма удаленых продаж: {removed_amount:.2f}")
        print(f"Доля от общей суммы продаж: {percent_removed_amount:.2f}%")
        
        df_clean = df_clean[~df_clean['InvoiceNo'].isin(non_conforming_invoices)]

    # 4. Вывод и удаление инвойсов с разными InvoiceDate или CustomerID для одного InvoiceNo
    inconsistent_invoices = df_clean.groupby('InvoiceNo').filter(lambda x: x['InvoiceDate'].nunique() > 1 or x['CustomerID'].nunique() > 1)['InvoiceNo'].unique()
    if len(inconsistent_invoices) > 0:
        print(f"\nИнвойсы с разными InvoiceDate или CustomerID для одного InvoiceNo:")
        print(inconsistent_invoices)
        print(f"Количество удаленных инвойсов: {len(inconsistent_invoices)}")

        # Удаление всех данных по инвойсу, если хоть одна строка бракованная
        removed_quantity = abs(df[df['InvoiceNo'].isin(inconsistent_invoices)]['Quantity'].sum())
        removed_amount = abs(df[df['InvoiceNo'].isin(inconsistent_invoices)]['Amount'].sum())
        total_removed_quantity += removed_quantity
        total_removed_amount += removed_amount
        percent_removed_quantity = removed_quantity / total_quantity * 100
        percent_removed_amount = removed_amount / total_amount * 100

        print(f"Количество удаленых товаров: {removed_quantity}")
        print(f"Доля от общего количества товаров: {percent_removed_quantity:.2f}%")
        print(f"Сумма удаленых продаж: {removed_amount:.2f}")
        print(f"Доля от общей суммы продаж: {percent_removed_amount:.2f}%")

        df_clean = df_clean[~df_clean['InvoiceNo'].isin(inconsistent_invoices)]

    # Итоговая информация по очищенным данным
    total_quantity_clean = abs(df_clean['Quantity'].sum())
    total_amount_clean = abs(df_clean['Amount'].sum())
    print(f"\nИтоговое количество товаров после очистки: {total_quantity_clean}")
    print(f"Итоговая сумма продаж после очистки: {total_amount_clean:.2f}")

    # Итоговая информация о проценте удаленных данных
    total_removed_quantity_percent = total_removed_quantity / total_quantity * 100
    total_removed_amount_percent = total_removed_amount / total_amount * 100
    print(f"\nИтого удалено {total_removed_quantity_percent:.2f}% от общего количества товаров и {total_removed_amount_percent:.2f}% от общей суммы продаж")

    return df_clean

def normalize_data(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    return scaler, data_scaled

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets)

def fill_missing_sales(df):
    """
    Заполняет пропущенные дни (например, субботы) средним арифметическим продаж за неделю и обрабатывает NaN и бесконечные значения.
    """
    # Убедимся, что 'Date' имеет тип datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Создаем полный набор дат от минимальной до максимальной даты
    full_date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
    
    # Включаем полный набор дат в DataFrame
    df_full = df.set_index('Date').reindex(full_date_range).reset_index().rename(columns={'index': 'Date'})
    
    # Найдем пропущенные значения
    missing_values = df_full[df_full['TotalSalesAmount'].isna()].copy()

    # Заполняем пропущенные значения средним арифметическим продаж за неделю
    df_full['TotalSalesAmount'] = df_full.groupby(df_full['Date'].dt.isocalendar().week)['TotalSalesAmount'].transform(lambda x: x.fillna(x.mean()))

    # Создаем DataFrame для хранения замененных значений пропущенных дней
    replaced_missing = missing_values.copy()
    replaced_missing['TotalSalesAmount_new'] = df_full.loc[missing_values.index, 'TotalSalesAmount']
    
    # Обработка NaN значений
    replaced_nans = df_full[df_full['TotalSalesAmount'].isna()].copy()
    df_full['TotalSalesAmount'] = df_full['TotalSalesAmount'].ffill().bfill()

    # Создаем DataFrame для хранения замененных значений NaN
    replaced_nans['TotalSalesAmount_new'] = df_full.loc[replaced_nans.index, 'TotalSalesAmount']

    # Обработка бесконечных значений
    df_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    replaced_infs = df_full[df_full['TotalSalesAmount'].isna()].copy()
    df_full['TotalSalesAmount'] = df_full['TotalSalesAmount'].ffill().bfill()

    # Создаем DataFrame для хранения замененных значений бесконечных значений
    replaced_infs['TotalSalesAmount_new'] = df_full.loc[replaced_infs.index, 'TotalSalesAmount']
    
    # Вывод замененных значений
    print("Пропущенные значения (по дням):\n", replaced_missing[['Date', 'TotalSalesAmount', 'TotalSalesAmount_new']])
    print("Значения, замененные по причине NaN:\n", replaced_nans[['Date', 'TotalSalesAmount', 'TotalSalesAmount_new']])
    print("Значения, замененные по причине бесконечных значений:\n", replaced_infs[['Date', 'TotalSalesAmount', 'TotalSalesAmount_new']])
    
    return df_full
