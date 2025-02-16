# modeling.py

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima

# ARIMA Model
def arima_model(train_data, test_data):
    # Проверка на стационарность
    result = adfuller(train_data)
    if result[1] > 0.05:
        train_data = train_data.diff().dropna()

    # Подбор параметров модели
    model_auto = auto_arima(train_data, seasonal=False, stepwise=True, trace=True)
    order = model_auto.order

    # Обучение модели
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test_data))

    return model_fit, forecast

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
