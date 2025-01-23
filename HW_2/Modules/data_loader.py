#data_loader.py
import kagglehub
import pandas as pd
import os

class DataLoader:
    @staticmethod
    def load_csv(file_path):
        """
        Загрузка данных из CSV файла.
        :param file_path: Путь к CSV файлу.
        :return: DataFrame с данными.
        """
        return pd.read_csv(file_path)

    @staticmethod
    def load_kaggle_dataset(dataset_name, file_name):
        """
        Загрузка данных напрямую с сайта Kaggle.
        :param dataset_name: Название датасета на Kaggle.
        :param file_name: Имя файла CSV.
        :return: Путь к загруженному файлу CSV.
        """
        dataset_path = kagglehub.dataset_download(dataset_name)
        csv_file_path = os.path.join(dataset_path, file_name)
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV файл '{file_name}' не найден в загруженном датасете.")
        return csv_file_path
