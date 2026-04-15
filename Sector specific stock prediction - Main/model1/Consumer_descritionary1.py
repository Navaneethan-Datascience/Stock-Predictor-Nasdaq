# Importing libraries
import yfinance as yf
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report

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

# Prepare data for Linear Regression (no sequences, just features and target)
X = data_normalized[features_to_normalize].values  # All features
y = data_normalized['Close'].values  # Target: normalized 'Close'

# Split into training and testing (80% train, 20% test)
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 4. BUILD THE LINEAR REGRESSION MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open('D:/Project/Sector stock prediction/model1/consumer_discretionary.pkl', 'wb') as file:
    pickle.dump(model, file)

# Evaluating the Model on Test Data
y_pred = model.predict(X_test)
y_pred_actual = scaler_close.inverse_transform(y_pred.reshape(-1, 1)).ravel()
y_test_actual = scaler_close.inverse_transform(y_test.reshape(-1, 1)).ravel()

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
def predict_future_days(model, last_features, scaler_close, days=1):
    predictions = []
    current_features = last_features.copy()  # Shape: (24,)
    
    for _ in range(days):
        # Predict the next day's 'Close' price
        next_pred_normalized = model.predict(current_features.reshape(1, -1))  # Shape: (1,)
        next_pred = scaler_close.inverse_transform(next_pred_normalized.reshape(-1, 1))[0, 0]
        predictions.append(next_pred)
        
        # Update features for the next prediction (simplified approach)
        new_features = current_features.copy()
        new_features[0] = next_pred_normalized[0]  # Update 'Close' (normalized)
        
        # For simplicity, other features are not recalculated (e.g., SMA, RSI)
        current_features = new_features
    
    return np.array(predictions).reshape(-1, 1)

# Next-day prediction
last_features = X[-1]  # Last row of features
next_day_pred = predict_future_days(model, last_features, scaler_close, days=1)
print(f"Predicted next day's stock price: ${next_day_pred[0][0]:.2f}")