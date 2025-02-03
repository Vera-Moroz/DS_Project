# Modules/model_training.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

class ModelTraining:
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
        # Построение модели логистической регрессии
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # Применение модели к тестовым данным
        y_pred = model.predict(X_test)
        
        # Оценка производительности модели
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Имена классов
        class_names = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']
        
        # Визуализация матрицы ошибок с именами классов
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Матрица ошибок')
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Истинные значения')
        plt.show()
        
        return model, accuracy, report, conf_matrix

    def train_knn(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
        # Построение модели K-ближайших соседей
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        
        # Применение модели к тестовым данным
        y_pred = model.predict(X_test)
        
        # Оценка производительности модели
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Имена классов
        class_names = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']
        
        # Визуализация матрицы ошибок с именами классов
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Матрица ошибок')
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Истинные значения')
        plt.show()
        
        return model, accuracy, report, conf_matrix

    def train_svm(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
        # Построение модели SVM
        model = SVC()
        model.fit(X_train, y_train)
        
        # Применение модели к тестовым данным
        y_pred = model.predict(X_test)
        
        # Оценка производительности модели
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Имена классов
        class_names = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']
        
        # Визуализация матрицы ошибок с именами классов
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Матрица ошибок')
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Истинные значения')
        plt.show()
        
        return model, accuracy, report, conf_matrix

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
        # Построение модели Случайного леса
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        
        # Применение модели к тестовым данным
        y_pred = model.predict(X_test)
        
        # Оценка производительности модели
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Имена классов
        class_names = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']
        
        # Визуализация матрицы ошибок с именами классов
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Матрица ошибок')
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Истинные значения')
        plt.show()
        
        return model, accuracy, report, conf_matrix

    def train_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
        # Построение модели Градиентного бустинга с использованием XGBoost и всех доступных ядер процессора
        model = xgb.XGBClassifier(n_estimators=100, tree_method='hist', n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Применение модели к тестовым данным
        y_pred = model.predict(X_test)
        
        # Оценка производительности модели
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Имена классов
        class_names = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']
        
        # Визуализация матрицы ошибок с именами классов
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Матрица ошибок')
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Истинные значения')
        plt.show()
        
        return model, accuracy, report, conf_matrix

    def train_naive_bayes(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
        # Построение модели Наивного байесовского классификатора
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        # Применение модели к тестовым данным
        y_pred = model.predict(X_test)
        
        # Оценка производительности модели
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Имена классов
        class_names = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']
        
        # Визуализация матрицы ошибок с именами классов
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Матрица ошибок')
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Истинные значения')
        plt.show()
        
        return model, accuracy, report, conf_matrix
