# Modules/eda.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Modules.preprocessing import DataPreprocessor
from Modules.feature_engineering import FeatureEngineering

class EDA:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_dir = os.path.join(self.base_dir, 'Data')
        self.load_data()
        self.top_features = []
        self.preprocessor = DataPreprocessor()
        self.feature_engineering = FeatureEngineering()

    def load_data(self):
        self.train_data = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        self.test_data = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))

    def display_basic_info(self):
        print("Train data shape:", self.train_data.shape)
        print("Test data shape:", self.test_data.shape)
        print("\nColumns and their data types:\n", self.train_data.dtypes)
        print("\nFirst few rows of train data:\n", self.train_data.head())

    def check_missing_values(self):
        print("Missing values in train data:")
        print(self.train_data.isnull().sum().sum())

        print("Missing values in test data:")
        print(self.test_data.isnull().sum().sum())

    def analyze_target_variable(self):
        # Вывод таблицы с уникальными значениями и их частотами
        activity_counts = self.train_data['Activity'].value_counts()
        print("Activity counts:\n", activity_counts)

        # Гистограмма распределения целевой переменной
        plt.figure(figsize=(8, 6))
        sns.countplot(x=self.train_data['Activity'])
        plt.title('Distribution of Activity Labels')
        plt.xlabel('Activity')
        plt.ylabel('Count')
        plt.show()

    def detect_outliers(self):
        # Метод для выявления выбросов с использованием среднего и стандартного отклонения
        def detect_outliers(data, n_std=3):
            mean = np.mean(data)
            std = np.std(data)
            cutoff = std * n_std
            lower, upper = mean - cutoff, mean + cutoff
            n_below = (data < lower).sum()
            n_above = (data > upper).sum()
            return lower, upper, n_below, n_above

        print("Выбросы по топ-12 признакам:")
        numeric_features = self.train_data[self.top_features]
        for column in numeric_features.columns:
            lower, upper, n_below, n_above = detect_outliers(self.train_data[column])
            print(f"{column} - Доверительный интервал: [{lower:.2f}, {upper:.2f}] - Выходят ниже: {n_below}, Выходят выше: {n_above}")

        # Визуализация выбросов с использованием ящиков с усами для топ-12 признаков
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(self.top_features[:12], 1):
            plt.subplot(4, 3, i)
            sns.boxplot(x=self.train_data[column])
            plt.title(f'Boxplot для {column}')
        plt.tight_layout()
        plt.show()

    def analyze_features(self):
        # Display statistical summary for numerical features
        numeric_features = self.train_data.select_dtypes(include=[np.number])
        print("Statistical summary of numerical features:")
        print(numeric_features.describe())

        # Select top k features using FeatureEngineering's select_k_best_features method
        X = numeric_features.drop(columns=['subject'])
        y = self.train_data['Activity']
        k = 12  # Количество признаков для отбора
        selected_features = self.feature_engineering.select_k_best_features(X, y, k=k)
        self.top_features = selected_features.columns

        # Вывод выбранных признаков
        print(f"\nTop {k} features selected based on ANOVA F-test:")
        print(self.top_features.tolist())

        # Построение корреляционной матрицы для отобранных признаков
        selected_data = selected_features
        corr_matrix = selected_data.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Корреляционная Матрица Отобранных Признаков')
        plt.show()

    def visualize_features(self):
        # Визуализация топ-12 признаков плиткой 4x3
        num_features_to_plot = 12
        num_rows = 4
        num_cols = 3
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        
        for i, feature in enumerate(self.top_features[:num_features_to_plot]):
            row = i // num_cols
            col = i % num_cols
            sns.histplot(self.train_data[feature], bins=30, kde=True, ax=axes[row, col])
            axes[row, col].set_title(f'Distribution of {feature}')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('Frequency')
        
        # Отключить оставшиеся пустые подграфики
        for j in range(i + 1, num_rows * num_cols):
            fig.delaxes(axes.flatten()[j])
            
        plt.show()
