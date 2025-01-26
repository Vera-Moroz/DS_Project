import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from db_utils import execute_query
from get_data_from_db import get_category_metrics

# Функция для визуализации распределения цен товаров по категориям
def plot_price_distribution_by_category():
    try:
        query = "SELECT \"Product Category\", \"Product Price\" FROM public.\"Ecommerce\";"
        result = execute_query(query)
        if result is not None:
            categories = result['Product Category'].unique()
            num_categories = len(categories)
            num_cols = 2
            num_rows = (num_categories + 1) // num_cols

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
            axes = axes.flatten()

            for i, category in enumerate(categories):
                category_data = result[result['Product Category'] == category]
                sns.histplot(category_data['Product Price'], kde=True, ax=axes[i])
                axes[i].set_title(f'{category}')
                axes[i].set_xlabel('Цена товара')
                axes[i].set_ylabel('Количество')

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.show()

            # Вывод таблицы с метриками
            metrics = get_category_metrics()
            print(metrics)
    except Exception as e:
        print(f"Error in plot_price_distribution_by_category: {e}")

# Функция для визуализации диаграмм рассеивания по категориям товаров
def plot_scatter_by_category():
    query = """
    SELECT "Product Category", "Product Price", SUM("Quantity") AS "Total Quantity"
    FROM public."Ecommerce"
    GROUP BY "Product Category", "Product Price"
    LIMIT 5000
    """
    result = execute_query(query)
    if result is not None:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=result, x='Product Price', y='Total Quantity', hue='Product Category')
        plt.title('Диаграмма рассеивания по категориям товаров')
        plt.xlabel('Цена товара')
        plt.ylabel('Количество товаров')
        plt.legend(title='Категория товара')
        plt.show()

# Функция для визуализации распределения покупок в зависимости от пола
def plot_scatter_by_gender():
    query = """
    SELECT "Product Category", "Product Price", "Quantity", "Gender"
    FROM public."Ecommerce"
    """
    result = execute_query(query)
    if result is not None:
        # Разбиваем цену на промежутки
        bins = [0, 100, 200, 300, 400, 500]
        result['Price Range'] = pd.cut(result['Product Price'], bins=bins, labels=['0-100', '100-200', '200-300', '300-400', '400-500'], include_lowest=True)

        # Агрегируем данные по промежуткам цены, категории и полу
        aggregated_data = result.groupby(['Price Range', 'Product Category', 'Gender'], observed=True).agg({'Quantity': 'sum'}).reset_index()

        # Плитка 1*4 из маленьких диаграмм по категориям
        categories = aggregated_data['Product Category'].unique()
        fig, axes = plt.subplots(1, len(categories), figsize=(24, 6), sharey=True)
        for i, category in enumerate(categories):
            category_data = aggregated_data[aggregated_data['Product Category'] == category]
            category_data_pivot = category_data.pivot(index='Price Range', columns='Gender', values='Quantity').fillna(0)
            category_data_pivot.plot(kind='bar', stacked=True, ax=axes[i], color=['red', 'blue'])
            axes[i].set_title(f'Категория: {category}')
            axes[i].set_xlabel('Диапазон цены')
            axes[i].set_ylabel('Количество товаров')
            axes[i].legend(title='Пол', loc='upper right')

        plt.tight_layout()
        plt.show()

        # Вывод текстовой таблицы
        total_data = aggregated_data.groupby(['Product Category', 'Gender']).agg({'Quantity': 'sum'}).unstack().fillna(0)
        total_data['Total'] = total_data.sum(axis=1)
        total_data.loc['Total'] = total_data.sum()
        total_data = total_data.astype(int)

        table_data = pd.DataFrame(columns=categories, index=['Мужчины', 'Женщины', 'Итого'])
        for category in categories:
            male_quantity = total_data.loc[category, ('Quantity', 'Male')].item()
            female_quantity = total_data.loc[category, ('Quantity', 'Female')].item()
            total_quantity = total_data.loc[category, 'Total'].item()
            male_percentage = f'{(male_quantity / total_quantity * 100):.1f}%' if total_quantity > 0 else '0%'
            female_percentage = f'{(female_quantity / total_quantity * 100):.1f}%' if total_quantity > 0 else '0%'
            table_data.loc['Мужчины', category] = f'{male_quantity} / {male_percentage}'
            table_data.loc['Женщины', category] = f'{female_quantity} / {female_percentage}'
            table_data.loc['Итого', category] = f'{total_quantity}'

        total_male_quantity = total_data.loc['Total', ('Quantity', 'Male')].item()
        total_female_quantity = total_data.loc['Total', ('Quantity', 'Female')].item()
        total_quantity = total_data.loc['Total', 'Total'].item()
        table_data['Total'] = [f'{total_male_quantity} / {(total_male_quantity / total_quantity * 100):.1f}%',
                               f'{total_female_quantity} / {(total_female_quantity / total_quantity * 100):.1f}%',
                               f'{total_quantity}']

        print(table_data.to_string(index=True, justify='left'))