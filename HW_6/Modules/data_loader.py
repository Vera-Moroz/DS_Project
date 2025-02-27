# data_loader.py

import pandas as pd
import os
import zipfile

def load_data_from_site():
    """
    Загрузка датасета 'Online Retail' из репозитория UCI и сохранение в файл.
    Возвращает DataFrame с данными.
    """
    base_path = os.path.dirname(__file__)
    features_path = os.path.join(base_path, '..', 'Data', 'online_retail_features.csv')
    
    if os.path.exists(features_path):
        # Загрузка данных из файла
        X = pd.read_csv(features_path)
        print(f"Данные загружены из файла: {features_path}")
    else:
        # Загрузка данных из репозитория UCI
        online_retail = fetch_ucirepo(id=352)
        
        if online_retail is not None:
            X = online_retail.data.features
            
            # Сохранение данных в CSV файл
            X.to_csv(features_path, index=False)
            print(f"Данные загружены из репозитория и сохранены в файл: {features_path}")
        else:
            print("Ошибка: Датасет не загружен.")
            return None
    
    return X

def load_data_from_zip():
    """
    Загрузка датасета 'Online Retail' из архива и сохранение в файл.
    Возвращает DataFrame с данными.
    """
    base_path = os.path.dirname(__file__)
    data_dir = os.path.join(base_path, '..', 'Data')
    zip_path = os.path.join(data_dir, 'online+retail.zip')
    excel_path = os.path.join(data_dir, 'Online Retail.xlsx')
    csv_path = os.path.join(data_dir, 'online_retail_csv.csv')  # Новый CSV файл для данных из архива
    
    if os.path.exists(csv_path):
        # Загрузка данных из файла CSV
        df = pd.read_csv(csv_path)
        print(f"Данные загружены из файла: {csv_path}")
    else:
        # Проверка наличия Excel файла, если его нет, извлекаем из архива
        if not os.path.exists(excel_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
                print(f"Данные извлечены из zip-архива в каталог {data_dir}.\nИдет чтение и загрузка xlsx-файла.")
        
        # Загрузка данных из файла Excel с использованием openpyxl
        df = pd.read_excel(excel_path, engine='openpyxl')
        
        # Сохранение данных в CSV файл для ускорения будущих загрузок
        df.to_csv(csv_path, index=False)
        print(f"Данные загружены из файла: {excel_path} и сохранены в файл: {csv_path} для ускорения будущих загрузок.")
    
    return df

def data_load():
    """
    Запрос у пользователя о выборе источника данных и загрузка данных.
    """
    data_source = input(f"""Откуда вы хотите загрузить данные?
    Предпочтительнее загрузка данных из архива, т.к. на сайте отсутствуют поля ['InvoiceNo', 'StockCode'].
    Введите 'site' для загрузки с сайта или 'zip' для загрузки из архива: """)

    if data_source.lower() == 'site':
        df = load_data_from_site()
    elif data_source.lower() == 'zip':
        df = load_data_from_zip()
    else:
        # Если введено некорректное значение, по умолчанию загружаем из архива
        print("Ошибка: некорректный выбор источника данных. Применено значение по умолчанию 'zip'.")
        df = load_data_from_zip()

    return df
