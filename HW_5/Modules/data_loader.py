# Modules/data_loader.py

import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

def load_data():
    """
    Загрузка датасета 'Online Retail' из репозитория UCI и сохранение в файл.
    Возвращает DataFrame с features.
    """
    # Указываем путь к файлам относительно расположения модуля
    base_path = os.path.dirname(__file__)
    features_path = os.path.join(base_path, '..', 'Data', 'online_retail_features.csv')
    
    if os.path.exists(features_path):
        # Загрузка данных из файла
        X = pd.read_csv(features_path)
        print("Данные загружены из файла.")
    else:
        # Загрузка данных из репозитория UCI
        online_retail = fetch_ucirepo(id=352)
        
        if online_retail is not None:
            # Получение features из датасета
            X = online_retail.data.features
            
            # Сохранение данных в CSV файл
            X.to_csv(features_path, index=False)
            
            print("Данные загружены из репозитория и сохранены в файл.")
        else:
            print("Ошибка: Датасет не загружен.")
            return None
    
    return X
