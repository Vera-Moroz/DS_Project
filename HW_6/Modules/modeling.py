import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from Modules.preprocessing import normalize_data, create_sequences
from Modules.visualization import visualize_predictions

# ARIMA Model with manual parameters
def arima_model(train_data, test_data, order):
    warnings.filterwarnings("ignore")

    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    print(model_fit.summary())

    forecast = model_fit.forecast(steps=len(test_data))
    forecast_index = pd.date_range(start=test_data.index[0], periods=len(test_data), freq='D')
    forecast = pd.Series(forecast, index=forecast_index)
    print(forecast)

    fitted_values = model_fit.predict(start=train_data.index[0], end=train_data.index[-1])
    fitted_values = pd.Series(fitted_values, index=train_data.index)

    model_name = f'ARIMA ({order[0]}, {order[1]}, {order[2]})'
    visualize_predictions(train_data, fitted_values, test_data['TotalSalesAmount'], forecast, model_name)

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
    warnings.filterwarnings("ignore")

    model_auto = auto_arima(train_data, seasonal=False, stepwise=True, trace=True)
    print(model_auto.summary())

    best_model = model_auto

    forecast = model_auto.predict(n_periods=len(test_data))
    forecast = pd.Series(forecast, index=test_data.index)

    fitted_values = model_auto.predict_in_sample()
    fitted_values = pd.Series(fitted_values, index=train_data.index)

    model_name = f'ARIMA({best_model.order[0]},{best_model.order[1]},{best_model.order[2]})'
    visualize_predictions(train_data, fitted_values, test_data['TotalSalesAmount'], forecast, model_name)

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
    warnings.filterwarnings("ignore")

    stepwise_model = auto_arima(train_data, seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)

    order = stepwise_model.order
    seasonal_order = stepwise_model.seasonal_order

    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    print(model_fit.summary())

    forecast = model_fit.forecast(steps=len(test_data))
    forecast_index = pd.date_range(start=test_data.index[0], periods=len(test_data), freq='D')
    forecast = pd.Series(forecast, index=forecast_index)
    print(forecast)

    fitted_values = model_fit.predict(start=train_data.index[0], end=train_data.index[-1])
    fitted_values = pd.Series(fitted_values, index=train_data.index)

    model_name = f'SARIMA ({order[0]}, {order[1]}, {order[2]}) x ({seasonal_order[0]}, {seasonal_order[1]}, {seasonal_order[2]}, {seasonal_order[3]})'
    visualize_predictions(train_data, fitted_values, test_data['TotalSalesAmount'], forecast, model_name)

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

# LSTM Model
def lstm_model(train_data, test_data, seq_length=12, epochs=100, batch_size=32):
    warnings.filterwarnings("ignore")

    # Нормализация данных
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
    test_data_scaled = scaler.transform(test_data.values.reshape(-1, 1))

    # Создание последовательных данных
    if len(train_data_scaled) > seq_length and len(test_data_scaled) > seq_length:
        LSTM_X_train, LSTM_y_train = create_sequences(train_data_scaled, seq_length)
        LSTM_X_test, LSTM_y_test = create_sequences(test_data_scaled, seq_length)
    else:
        raise ValueError("Ошибка: Размер данных недостаточен для создания последовательностей с указанной длиной.")

    # Модель LSTM
    LSTM_model = Sequential()
    LSTM_model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
    LSTM_model.add(Dense(1))
    LSTM_model.compile(optimizer='adam', loss='mean_squared_error')
    LSTM_model.fit(LSTM_X_train, LSTM_y_train, epochs=epochs, batch_size=batch_size)

    # Прогнозирование на тестовых данных
    LSTM_forecast = LSTM_model.predict(LSTM_X_test)

    # Обратное масштабирование нормализованных данных к исходному масштабу
    LSTM_y_train_inv = scaler.inverse_transform(LSTM_y_train.reshape(-1, 1))
    LSTM_y_test_inv = scaler.inverse_transform(LSTM_y_test.reshape(-1, 1))
    LSTM_forecast_inv = scaler.inverse_transform(LSTM_forecast)

    # Преобразование предсказаний и данных в DataFrame для сохранения временных меток
    LSTM_y_train_inv_df = pd.DataFrame(LSTM_y_train_inv, index=train_data.index[:len(LSTM_y_train_inv)], columns=['TotalSalesAmount'])
    LSTM_y_test_inv_df = pd.DataFrame(LSTM_y_test_inv, index=test_data.index[:len(LSTM_y_test_inv)], columns=['TotalSalesAmount'])
    LSTM_forecast_inv_df = pd.DataFrame(LSTM_forecast_inv, index=test_data.index[:len(LSTM_forecast_inv)], columns=['TotalSalesAmount'])

    # Визуализация результатов прогноза
    visualize_predictions(train_data, LSTM_y_train_inv_df, LSTM_y_test_inv_df, LSTM_forecast_inv_df, model_name="LSTM")

    # Метрики качества
    LSTM_mae = mean_absolute_error(LSTM_y_test_inv, LSTM_forecast_inv)
    LSTM_mse = mean_squared_error(LSTM_y_test_inv, LSTM_forecast_inv)
    LSTM_rmse = np.sqrt(LSTM_mse)

    print(f"Mean Absolute Error (MAE): {LSTM_mae:.3f}")
    print(f"Mean Squared Error (MSE): {LSTM_mse:.3f}")
    print(f"Root Mean Squared Error (RMSE): {LSTM_rmse:.3f}")

    return LSTM_model, LSTM_forecast_inv_df, LSTM_y_test_inv_df, LSTM_y_train_inv_df, LSTM_mae, LSTM_mse, LSTM_rmse
