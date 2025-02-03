# Modules/data_loader.py
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

class DataLoader:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_dir = os.path.join(self.base_dir, 'Data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.api = KaggleApi()
        self.api.authenticate()

    def download_dataset(self, dataset):
        self.api.dataset_download_files(dataset, path=self.data_dir, unzip=False)
        print(f"Dataset '{dataset}' downloaded to '{self.data_dir}'")

    def unzip_dataset(self, zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        print(f"Dataset '{zip_file}' extracted to '{self.data_dir}'")

# Основная функция для загрузки и распаковки данных
def load_har_dataset(base_dir):
    dataset = 'uciml/human-activity-recognition-with-smartphones'
    data_loader = DataLoader(base_dir)
    data_loader.download_dataset(dataset)
    zip_file = os.path.join(data_loader.data_dir, 'human-activity-recognition-with-smartphones.zip')
    data_loader.unzip_dataset(zip_file)

if __name__ == "__main__":
    # Определяем базовую директорию на основе местоположения текущего файла
    base_dir = os.path.dirname(os.path.abspath(__file__))
    load_har_dataset(base_dir)
