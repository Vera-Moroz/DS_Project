# Modules/preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}

    def encode_categorical(self, train_data: pd.DataFrame, test_data: pd.DataFrame, columns: list):
        for column in columns:
            le = LabelEncoder()
            train_data[column] = le.fit_transform(train_data[column])
            test_data[column] = le.transform(test_data[column])
            self.label_encoders[column] = le
        print("Категориальные признаки закодированы с помощью LabelEncoder.")
        self.print_label_mapping(columns)
        return train_data, test_data

    def print_label_mapping(self, columns: list):
        for column in columns:
            label_mapping = dict(zip(self.label_encoders[column].classes_, self.label_encoders[column].transform(self.label_encoders[column].classes_)))
            print(f"Новая кодировка для '{column}':")
            for key, value in label_mapping.items():
                print(f"{key} = {value}")
