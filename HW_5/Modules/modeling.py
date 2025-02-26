# modeling.py

import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Modules.visualization import visualize_predictions 

# ARIMA Model with manual parameters
def arima_model(train_data, test_data, order):
    import warnings
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from Modules.visualization import visualize_predictions  # Импортируем функцию визуализации

    # Отключение предупреждений
    warnings.filterwarnings("ignore")
    
    # Обучение модели
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    
    # Прогнозирование для тестовой выборки
    forecast = model_fit.forecast(steps=len(test_data))
    forecast_index = pd.date_range(start=test_data.index[0], periods=len(test_data), freq='D')
    forecast = pd.Series(forecast, index=forecast_index)
    print(forecast)
    
    # Получение предсказанных значений на обучающей выборке
    fitted_values = model_fit.predict(start=train_data.index[0], end=train_data.index[-1])
    fitted_values = pd.Series(fitted_values, index=train_data.index)

    # Визуализация данных и предсказаний
    model_name = f'ARIMA ({order[0]}, {order[1]}, {order[2]})'
    visualize_predictions(train_data, fitted_values, test_data['TotalSalesAmount'], forecast, model_name)
    
    # Метрики качества
    mae = mean_absolute_error(test_data, forecast)
    mse = mean_squared_error(test_data, forecast)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100

    print(f'Mean Absolute Error (MAE): {mae:.3f}')
    print(f'Mean Squared Error (MSE): {mse:.3f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.3f}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape:.3f}%')
    
    
    return model_fit, forecast, fitted_values



# ARIMA Model with auto ARIMA
def auto_arima_model(train_data, test_data):
    # Автоматический подбор параметров (p, d, q)
    model_auto = auto_arima(train_data, seasonal=False, stepwise=True, trace=True)
    print(model_auto.summary())

    # Сохранение лучших параметров модели
    best_model = model_auto

    # Прогнозирование для тестовой выборки
    forecast_test = model_auto.predict(n_periods=len(test_data))
    forecast_test = pd.Series(forecast_test, index=test_data.index)

    # Получение предсказанных значений на обучающей выборке
    fitted_values = model_auto.predict_in_sample()
    fitted_values = pd.Series(fitted_values, index=train_data.index)
    
    return best_model, forecast_test, fitted_values

# SARIMA Model
def sarima_model(train_data, test_data):
    # Подбор параметров модели
    model_auto = auto_arima(train_data, seasonal=True, m=12, stepwise=True, trace=True)
    order = model_auto.order
    seasonal_order = model_auto.seasonal_order

    # Обучение модели
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test_data))

    return model_fit, forecast

# LSTM Model
def lstm_model(X_train, y_train, X_test, y_test, seq_length=30):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=2)
    forecast = model.predict(X_test)
    return model, forecast

# GRU Model
def gru_model(X_train, y_train, X_test, y_test, seq_length=30):
    model = Sequential()
    model.add(GRU(50, activation='relu', input_shape=(seq_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=2)
    forecast = model.predict(X_test)
    return model, forecast

# CNN Model
def cnn_model(X_train, y_train, X_test, y_test, seq_length=30):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(seq_length, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=2)
    forecast = model.predict(X_test)
    return model, forecast
