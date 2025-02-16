# visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Установка стиля Seaborn
sns.set(style="whitegrid")

def plot_line_chart(data, x, y, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=x, y=y, data=data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def visualize_missing_data(df_date_features):
    # Создаем копию DataFrame для работы
    temp_df = df_date_features.copy()

    # Получаем минимальную и максимальную дату в данных
    min_date = temp_df['Date'].min()
    max_date = temp_df['Date'].max()

    # Создаем полный диапазон дат
    full_date_range = pd.date_range(start=min_date, end=max_date)

    # Проверяем, какие даты отсутствуют в данных
    missing_dates = full_date_range.difference(temp_df['Date'])

    # Создаем DataFrame для пропущенных дат
    missing_data_df = pd.DataFrame({'Date': missing_dates})
    missing_data_df['Year'] = missing_data_df['Date'].dt.year
    missing_data_df['Week'] = missing_data_df['Date'].dt.isocalendar().week
    missing_data_df['DayOfWeek'] = missing_data_df['Date'].dt.day_name()

    # Фильтруем пропуски для указанных недель
    filtered_missing_data = missing_data_df[((missing_data_df['Year'] == 2010) & (missing_data_df['Week'].isin([51, 52]))) | 
                                            ((missing_data_df['Year'] == 2011) & (missing_data_df['Week'].isin([1, 16, 17, 18, 22, 35, 52])))]

    # Выводим пропущенные даты
    print("Пропущенные даты на 51-52 неделях 2010г и 16-17 неделях 2011г:")
    print(filtered_missing_data)

    # Подсчитываем количество пропусков по неделям и годам
    missing_data_by_week = missing_data_df.groupby(['Year', 'Week']).size().reset_index(name='MissingDays')
    missing_data_by_week = missing_data_by_week[missing_data_by_week['MissingDays'] > 0]

    # Визуализация пропусков по неделям и годам
    plt.figure(figsize=(7, 3))
    for year in missing_data_by_week['Year'].unique():
        data = missing_data_by_week[missing_data_by_week['Year'] == year]
        plt.scatter(data['Week'], data['MissingDays'], label=str(year))

    plt.title('Пропуски в данных по неделям и годам')
    plt.xlabel('Неделя')
    plt.ylabel('Количество дней с пропусками')
    plt.ylim(0, 7)  # Показать только недели с пропусками
    plt.legend(title='Год')
    plt.show()

    # Устанавливаем порядок дней недели, начиная с понедельника
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    missing_data_df['DayOfWeek'] = pd.Categorical(missing_data_df['DayOfWeek'], categories=ordered_days, ordered=True)

    # Подсчитываем количество пропусков по дням недели
    missing_data_by_day = missing_data_df['DayOfWeek'].value_counts().reset_index()
    missing_data_by_day.columns = ['DayOfWeek', 'MissingDays']

    # Визуализация пропусков по дням недели
    plt.figure(figsize=(7, 3))
    bars = plt.bar(missing_data_by_day['DayOfWeek'], missing_data_by_day['MissingDays'], color='skyblue', edgecolor='black')
    plt.title('Пропуски в данных по дням недели')
    plt.xlabel('День недели')
    plt.ylabel('Количество пропусков')
    plt.yticks(ticks=range(0, 53, 10))  # Устанавливаем шаг 10 по оси Y
    plt.show()

def plot_heatmap(df_date_features, title, xlabel, ylabel):
    # Создаем копию DataFrame для работы
    temp_df = df_date_features.copy()

    # Подготовка данных для тепловой карты
    pivot_table = temp_df.pivot_table(index='DayOfWeek', columns='MonthOfYear', values='TotalSalesAmount', aggfunc='sum')

    # Обработка NaN значений, заменяем их на 0
    pivot_table = pivot_table.fillna(0)

    # Визуализация тепловой карты
    plt.figure(figsize=(15, 10))
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='viridis', linewidths=.5, cbar_kws={'label': 'Total Sales Amount'})
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def correlation_matrix(df):
    # Исключаем столбцы с некорректными для корреляции значениями (например, datetime и строковые столбцы)
    numeric_df = df.select_dtypes(include=[float, int])

    # Создание корреляционной матрицы
    corr = numeric_df.corr()

    # Настройка размеров графика
    plt.figure(figsize=(12, 8))

    # Визуализация корреляционной матрицы с помощью тепловой карты
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Корреляционная матрица')
    plt.show()

def visualize_time_series(ts, title='Временной ряд'):
    """
    Визуализирует временной ряд.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(ts, label='Временной ряд')
    plt.title(title)
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_residuals(residuals, title='Остатки модели'):
    """
    Визуализирует остатки модели.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(residuals, label='Остатки')
    plt.title(title)
    plt.axhline(0, linestyle='--', color='red', label='Нулевая линия')
    plt.xlabel('Дата')
    plt.ylabel('Остатки')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_forecast(train_data, test_data, forecast, title='Прогноз временного ряда'):
    """
    Визуализирует обучающие данные, тестовые данные и прогноз.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_data, label='Обучающие данные')
    plt.plot(test_data.index, test_data, label='Тестовые данные')
    plt.plot(test_data.index, forecast, color='green', label='Прогноз')
    plt.title(title)
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_residuals_test(residuals, title='Остатки на тестовой выборке'):
    """
    Визуализирует остатки на тестовой выборке.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(residuals, label='Остатки')
    plt.title(title)
    plt.axhline(0, linestyle='--', color='red', label='Нулевая линия')
    plt.xlabel('Дата')
    plt.ylabel('Остатки')
    plt.legend()
    plt.grid(True)
    plt.show()
