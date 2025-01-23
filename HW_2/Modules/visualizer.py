#visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self):
        sns.set(style="whitegrid")

    def add_histogram(self, data, column):
        """
        Добавление гистограммы с подписями количества.
        :param data: DataFrame с данными.
        :param column: Название столбца для построения гистограммы.
        """
        plt.figure(figsize=(15, 10))
        ax = sns.histplot(data[column], bins=len(data[column].unique()), color='skyblue', edgecolor='black')
        
        # Добавление подписей с количеством
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.0f}', 
                        xy=(p.get_x() + p.get_width() / 2, height), 
                        xytext=(0, 5),  # Смещение текста
                        textcoords='offset points', 
                        ha='center', va='bottom')
        
        # Добавление названий для осей и заголовка
        ax.set_xlabel('Значение' if column != 'tenure_bins' else 'Интервалы tenure')
        ax.set_ylabel('Частота')
        ax.set_title(f'Гистограмма для {column}')
        
        # Регулировка макета для предотвращения наложения подписей
        plt.tight_layout()
        
        # Показать график
        plt.show()

    def add_line_plot(self, data, x_column, y_column):
        """
        Добавление линейного графика.
        :param data: DataFrame с данными.
        :param x_column: Название столбца для оси X.
        :param y_column: Название столбца для оси Y.
        """
        plt.figure(figsize=(15, 10))
        ax = sns.lineplot(data=data, x=x_column, y=y_column)
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(f'Line Plot of {y_column} vs {x_column}')
        
        # Регулировка макета для предотвращения наложения подписей
        plt.tight_layout()
        
        # Показать график
        plt.show()

    def add_pairplot(self, data, columns, hue):
        """
        Добавление парных диаграмм рассеивания.
        :param data: DataFrame с данными.
        :param columns: Список столбцов для построения парных диаграмм.
        :param hue: Название столбца для цветовой кодировки.
        """
        sns.pairplot(data[columns + [hue]], hue=hue, palette='Set1')
        
        # Показать график
        plt.show()
