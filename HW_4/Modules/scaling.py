# Modules/scaling.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer

class DataScaler:
    def __init__(self):
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.maxabs_scaler = MaxAbsScaler()
        self.robust_scaler = RobustScaler()
        self.normalizer = Normalizer()

    def standard_scale(self, data: pd.DataFrame):
        features = data.drop(columns=['Activity'])  # Исключаем целевую переменную
        scaled_features = pd.DataFrame(self.standard_scaler.fit_transform(features), columns=features.columns, index=data.index)
        scaled_data = pd.concat([scaled_features, data['Activity']], axis=1)  # Добавляем обратно целевую переменную
        return scaled_data

    def minmax_scale(self, data: pd.DataFrame):
        features = data.drop(columns=['Activity'])  # Исключаем целевую переменную
        scaled_features = pd.DataFrame(self.minmax_scaler.fit_transform(features), columns=features.columns, index=data.index)
        scaled_data = pd.concat([scaled_features, data['Activity']], axis=1)  # Добавляем обратно целевую переменную
        return scaled_data

    def maxabs_scale(self, data: pd.DataFrame):
        features = data.drop(columns=['Activity'])  # Исключаем целевую переменную
        scaled_features = pd.DataFrame(self.maxabs_scaler.fit_transform(features), columns=features.columns, index=data.index)
        scaled_data = pd.concat([scaled_features, data['Activity']], axis=1)  # Добавляем обратно целевую переменную
        return scaled_data

    def normalize(self, data: pd.DataFrame):
        features = data.drop(columns=['Activity'])  # Исключаем целевую переменную
        scaled_features = pd.DataFrame(self.normalizer.fit_transform(features), columns=features.columns, index=data.index)
        scaled_data = pd.concat([scaled_features, data['Activity']], axis=1)  # Добавляем обратно целевую переменную
        return scaled_data
