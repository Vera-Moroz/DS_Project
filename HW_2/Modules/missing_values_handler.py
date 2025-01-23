#missing_values_handler.py
import pandas as pd

class MissingValuesHandler:
    @staticmethod
    def count_missing_values(data):
        """
        Подсчет пустых или пропущенных значений в каждом столбце.
        :param data: DataFrame с данными.
        :return: DataFrame с количеством пропущенных значений в каждом столбце и процент пропущенных значений.
        """
        missing_values = data.isnull().sum()
        missing_percentage = (missing_values / len(data)) * 100
        return pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})

    @staticmethod
    def report_missing_values(data):
        """
        Вывод отчета с информацией о пропущенных значениях.
        :param data: DataFrame с данными.
        :return: DataFrame с пропущенными значениями.
        """
        missing_columns = data.columns[data.isnull().any()].tolist()
        missing_rows = data[missing_columns].isnull().any(axis=1)
        return data.loc[missing_rows, missing_columns]

    @staticmethod
    def fill_missing_values(data, default_method='mean', **column_methods):
        """
        Заполнение пропущенных значений.
        :param data: DataFrame с данными.
        :param default_method: Метод заполнения по умолчанию ('mean', 'median', 'mode').
        :param column_methods: Опциональные параметры для конкретных столбцов со своим методом заполнения или значением по умолчанию.
        :return: DataFrame с заполненными пропущенными значениями.
        """
        for column in data.columns:
            method = column_methods.get(column, default_method)
            if data[column].isnull().sum() > 0:
                if method == 'mean' and data[column].dtype in ['float64', 'int64']:
                    data[column] = data[column].fillna(data[column].mean())
                elif method == 'median' and data[column].dtype in ['float64', 'int64']:
                    data[column] = data[column].fillna(data[column].median())
                elif method == 'mode':
                    data[column] = data[column].fillna(data[column].mode().iloc[0])
                elif isinstance(method, (int, float, str)):
                    data[column] = data[column].fillna(method)
                else:
                    raise ValueError(f"Некорректный метод заполнения для столбца '{column}': {method}")
        return data
