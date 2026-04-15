# Importing libraries
import yfinance as yf
import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

# 1. COLLECTING DATA FROM YAHOO FINANCE
start_date = input("Enter Start Date (YYYY-MM-DD): ")
end_date = input("Enter End Date (YYYY-MM-DD): ")
ticker = input("Enter stock ticker: ")
data = yf.download(ticker, start=start_date, end=end_date)


# Drop the ticker level from multi index
if 'Ticker' in data.columns.names:
    data.columns = data.columns.droplevel('Ticker')
data.reset_index(inplace=True)
if 'index' in data.columns:
    data.rename(columns={"index": "Date"}, inplace=True)

# 2. FEATURE ENGINEERING
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_100'] = data['Close'].rolling(window=100).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()
data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
data['EMA_100'] = data['Close'].ewm(span=100, adjust=False).mean()

delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI_14'] = 100 - (100 / (1 + rs))

gain_21 = (delta.where(delta > 0, 0)).rolling(window=21).mean()
loss_21 = (-delta.where(delta < 0, 0)).rolling(window=21).mean()
rs_21 = gain_21 / loss_21
data['RSI_21'] = 100 - (100 / (1 + rs_21))

short_ema = data['Close'].ewm(span=12, adjust=False).mean()
long_ema = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = short_ema - long_ema
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

data['BB_Middle'] = data['Close'].rolling(window=20).mean()
data['BB_Upper'] = data['BB_Middle'] + (2 * data['Close'].rolling(window=20).std())
data['BB_Lower'] = data['BB_Middle'] - (2 * data['Close'].rolling(window=20).std())

data['BB_Middle_50'] = data['Close'].rolling(window=50).mean()
data['BB_Upper_50'] = data['BB_Middle_50'] + (2 * data['Close'].rolling(window=50).std())
data['BB_Lower_50'] = data['BB_Middle_50'] - (2 * data['Close'].rolling(window=50).std())

data['Voltility'] = data['High'] - data['Low']
data['Daily_Return'] = data['Close'].pct_change()
data['Volume_MA_50'] = data['Volume'].rolling(window=50).mean()
data['Volume_MA_100'] = data['Volume'].rolling(window=100).mean()

data = data.fillna(method='bfill').fillna(method='ffill')

# 3. DATA PREPROCESSING
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Separate scaler for 'Close' price
scaler_close = MinMaxScaler()
scaler_close.fit(data[['Close']])

features_to_normalize = ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_50', 'SMA_100',
                         'SMA_200', 'EMA_50', 'EMA_100', 'RSI_14', 'RSI_21', 'MACD',
                         'Signal_Line', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Middle_50',
                         'BB_Upper_50', 'BB_Lower_50', 'Voltility', 'Daily_Return',
                         'Volume_MA_50', 'Volume_MA_100']

scaler = MinMaxScaler()
data_normalized = data.copy()
data_normalized[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])

features = data_normalized.columns.tolist()
data_values = data_normalized[features].values
sequence_length = 100

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Predict 'Close' (index 0 after normalization)
    return np.array(x), np.array(y)

x, y = create_sequences(data_values, sequence_length)
split_index = int(len(x) * 0.8)
x_train, x_test = x[:split_index], x[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 4. BUILD THE GRU MODEL
model = Sequential([
    GRU(units=100, return_sequences=True, input_shape=(sequence_length, x_train.shape[2])),
    Dropout(0.2),
    GRU(units=100, return_sequences=False),
    Dropout(0.2),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=32,
                    validation_data=(x_test, y_test),
                    verbose=1)

model.save('D:/Project/Sector stock prediction/model1/utilities1.h5')

# Evaluating the model on Test Data
y_pred = model.predict(x_test)
y_pred_actual = scaler_close.inverse_transform(y_pred.reshape(-1, 1))
y_test_actual = scaler_close.inverse_transform(y_test.reshape(-1, 1))

# Calculate regression metrics
mse = mean_squared_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_pred_actual)
tolerance = 0.05  # 5%
within_tolerance = np.abs(y_test_actual - y_pred_actual) / y_test_actual < tolerance
accuracy_like = np.mean(within_tolerance)

# Print regression metrics
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Custom Accuracy (within ±5%): {accuracy_like * 100:.2f}%")

# Classification: Predict price direction (Up/Down)
# Actual direction: 1 (Up) if next day's price > previous day's price, else 0 (Down)
actual_direction = (y_test_actual[1:] > y_test_actual[:-1]).astype(int)
# Predicted direction: 1 (Up) if predicted price > previous predicted price, else 0 (Down)
predicted_direction = (y_pred_actual[1:] > y_pred_actual[:-1]).astype(int)

# Generate classification report
print("\nClassification Report for Price Direction (Up/Down):")
print(classification_report(actual_direction, predicted_direction, target_names=['Down', 'Up']))

# 5. NEXT-DAY PREDICTION
def predict_future_days(model, last_sequence, scaler_close, days=1):
    predictions = []
    current_sequence = last_sequence.copy()  # Shape: (100, 24)
    
    for _ in range(days):
        # Ensure the sequence is reshaped correctly for the model
        current_sequence_reshaped = current_sequence.reshape(1, sequence_length, current_sequence.shape[1])  # (1, 100, 24)
        next_pred = model.predict(current_sequence_reshaped, verbose=0)  # Predict next 'Close'
        predictions.append(next_pred[0, 0])
        
        # Create a new row with the predicted 'Close' and placeholder values for other features
        new_row = np.zeros((1, current_sequence.shape[1]))  # Shape: (1, 24)
        new_row[0, 0] = next_pred[0, 0]  # 'Close' is at index 0
        
        # Shift the sequence and append the new row
        current_sequence = np.concatenate((current_sequence[1:], new_row), axis=0)  # Keep shape (100, 24)
    
    # Convert predictions back to original scale
    predictions = np.array(predictions).reshape(-1, 1)
    return scaler_close.inverse_transform(predictions)

# Next-day prediction
last_sequence = x_test[-1]  # Shape: (100, 24)
next_day_pred = predict_future_days(model, last_sequence, scaler_close, days=1)
print(f"Predicted next day's stock price: ${next_day_pred[0][0]:.2f}")