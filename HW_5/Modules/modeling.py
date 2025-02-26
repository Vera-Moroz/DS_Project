# modeling.py

import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from Modules.preprocessing import normalize_data, create_sequences
from Modules.visualization import visualize_predictions


# ARIMA Model with manual parameters
def arima_model(train_data, test_data, order):
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

    return model_fit, forecast, fitted_values, mae, mse, rmse, mape


# ARIMA Model with auto parameters
def auto_arima_model(train_data, test_data):
    # Отключение предупреждений
    warnings.filterwarnings("ignore")
    
    # Автоматический подбор параметров (p, d, q)
    model_auto = auto_arima(train_data, seasonal=False, stepwise=True, trace=True)
    print(model_auto.summary())

    # Сохранение лучших параметров модели
    best_model = model_auto

    # Прогнозирование для тестовой выборки
    forecast = model_auto.predict(n_periods=len(test_data))
    forecast = pd.Series(forecast, index=test_data.index)

    # Получение предсказанных значений на обучающей выборке
    fitted_values = model_auto.predict_in_sample()
    fitted_values = pd.Series(fitted_values, index=train_data.index)
    
    # Визуализация данных и предсказаний
    model_name = f'ARIMA({best_model.order[0]},{best_model.order[1]},{best_model.order[2]})'
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
    
    return model_auto, best_model, forecast, fitted_values, mae, mse, rmse, mape


# SARIMA Model with auto parameters
def sarima_model(train_data, test_data):
    # Отключение предупреждений
    warnings.filterwarnings("ignore")

    # Автоподбор параметров order и seasonal_order
    stepwise_model = auto_arima(train_data, seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    
    # Получение наилучших параметров
    order = stepwise_model.order
    seasonal_order = stepwise_model.seasonal_order
    
    # Обучение модели
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
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
    model_name = f'SARIMA ({order[0]}, {order[1]}, {order[2]}) x ({seasonal_order[0]}, {seasonal_order[1]}, {seasonal_order[2]}, {seasonal_order[3]})'
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

    print(f'Best model: SARIMA {order} x {seasonal_order}')

    return model_fit, forecast, fitted_values, mae, mse, rmse, mape, order, seasonal_order

#LSTM Model
def lstm_model(train_data, test_data, seq_length, n_features):
    try:
        # Нормализация данных
        scaler, train_scaled = normalize_data(train_data)
        test_scaled = scaler.transform(test_data.values.reshape(-1, 1))

        # Подготовка данных для LSTM
        X_train, y_train = create_sequences(train_scaled, seq_length)
        X_test, y_test = create_sequences(test_scaled, seq_length)

        # Преобразование данных к форме (samples, timesteps, features)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

        # Создание модели
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(seq_length, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Обучение модели
        model.fit(X_train, y_train, epochs=200, verbose=0)

        # Прогнозирование для тестовой выборки
        forecast = []
        input_seq = X_test[0]
        for i in range(len(X_test)):
            pred = model.predict(input_seq.reshape(1, seq_length, n_features), verbose=0)
            forecast.append(pred[0][0])
            input_seq = np.append(input_seq[1:], pred).reshape(seq_length, n_features)

        forecast_index = pd.date_range(start=test_data.index[seq_length], periods=len(forecast), freq='D')
        forecast = pd.Series(forecast, index=forecast_index)

        # Получение предсказанных значений на обучающей выборке
        fitted_values = model.predict(X_train, verbose=0)
        fitted_values_index = pd.date_range(start=train_data.index[seq_length], periods=len(fitted_values), freq='D')
        fitted_values = pd.Series(fitted_values.flatten(), index=fitted_values_index)

        # Визуализация данных и предсказаний
        model_name = 'LSTM'
        visualize_predictions(train_data, fitted_values, test_data['TotalSalesAmount'], forecast, model_name)

        # Метрики качества
        mae = mean_absolute_error(y_test, forecast)
        mse = mean_squared_error(y_test, forecast)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - forecast) / y_test)) * 100

        print(f'Mean Absolute Error (MAE): {mae:.3f}')
        print(f'Mean Squared Error (MSE): {mse:.3f}')
        print(f'Root Mean Squared Error (RMSE): {rmse:.3f}')
        print(f'Mean Absolute Percentage Error (MAPE): {mape:.3f}%')

        return model, forecast, fitted_values, mae, mse, rmse, mape
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return None, None, None, None, None, None, None

