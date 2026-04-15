from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODEL_MAP = {
    "Energy": {"type": "random_forest", "path": "model1/energy1.pkl"},
    "Materials": {"type": "gradient_boosting", "path": "model1/materials1.pkl"},
    "Industrials": {"type": "rnn", "path": "model1/industrials1.h5"},
    "Utilities": {"type": "gru", "path": "model1/utilities1.h5"},
    "Healthcare": {"type": "cnn", "path": "model1/healthcare1.h5"},
    "Financials": {"type": "lstm", "path": "model1/finance1.h5"},
    "Consumer Discretionary": {"type": "linear_regression", "path": "model1/consumer_discretionary.pkl"},
    "Consumer Staples": {"type": "linear_regression", "path": "model1/consumer_staples.pkl"},
    "Real Estate": {"type": "svm", "path": "model1/real_estate1.pkl"},
    "Technology": {"type": "lstm", "path": "model1/technology1.h5"},
    "Communication Services": {"type": "svm", "path": "model1/communication_services1.pkl"}
}

def get_sector_from_yfinance(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        yf_sector = info.get('sector', 'Unknown')
        logger.debug(f"Sector for {ticker}: {yf_sector}")
        SECTOR_MAPPING = {
            "Energy": ["Energy", "Oil & Gas", "Energy Equipment & Services"],
            "Materials": ["Basic Materials", "Chemicals", "Metals & Mining"],
            "Industrials": ["Industrials", "Aerospace & Defense", "Industrial Goods"],
            "Utilities": ["Utilities", "Utility Services", "Renewable Energy", "Electric Utilities"],
            "Healthcare": ["Healthcare", "Biotechnology", "Pharmaceuticals"],
            "Financials": ["Financial Services", "Banks", "Insurance", "Financials"],
            "Consumer Discretionary": ["Consumer Cyclical", "Automotive", "Retail", "Consumer Discretionary"],
            "Consumer Staples": ["Consumer Defensive", "Food & Beverage", "Household Products", "Consumer Staples"],
            "Real Estate": ["Real Estate", "REITs", "Property Management"],
            "Technology": ["Technology", "Software", "Hardware"],
            "Communication Services": ["Communication Services", "Telecom", "Media", "Communication"]
        }
        for app_sector, yf_sectors in SECTOR_MAPPING.items():
            if yf_sector in yf_sectors:
                return app_sector
        return 'Unknown'
    except Exception as e:
        logger.error(f"Error fetching sector for {ticker}: {e}")
        return 'Unknown'

def preprocess_stock_data(data):
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    elif 'Ticker' in data.columns.names:
        data.columns = data.columns.droplevel('Ticker')

    try:
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_100'] = data['Close'].rolling(window=100).mean()
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
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (2 * bb_std)
        data['BB_Lower'] = data['BB_Middle'] - (2 * bb_std)
        data['BB_Middle_50'] = data['Close'].rolling(window=50).mean()
        bb_std_50 = data['Close'].rolling(window=50).std()
        data['BB_Upper_50'] = data['BB_Middle_50'] + (2 * bb_std_50)
        data['BB_Lower_50'] = data['BB_Middle_50'] - (2 * bb_std_50)
        data['Volatility'] = data['High'] - data['Low']
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volume_MA_50'] = data['Volume'].rolling(window=50).mean()
        data['Volume_MA_100'] = data['Volume'].rolling(window=100).mean()
        data['Dummy_Feature'] = 1.0

        data = data.fillna(method='bfill').fillna(method='ffill')
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise ValueError(f"Error in preprocessing: {str(e)}")

    return data

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sector = data['sector']
    ticker = data['ticker'].upper()
    start_date = data['startDate']
    end_date = data['endDate']
    prediction_range = data['predictionRange']

    actual_sector = get_sector_from_yfinance(ticker)
    if actual_sector == 'Unknown' or actual_sector.lower() != sector.lower():
        return jsonify({"message": f"Ticker {ticker} does not match selected sector {sector}. Actual sector: {actual_sector}"}), 400

    if datetime.strptime(end_date, '%Y-%m-%d') > datetime.now():
        return jsonify({"message": "End date cannot be in the future"}), 400
    if datetime.strptime(start_date, '%Y-%m-%d') >= datetime.strptime(end_date, '%Y-%m-%d'):
        return jsonify({"message": "Start date must be before end date"}), 400

    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            return jsonify({"message": f"No data found for ticker {ticker} in the given date range"}), 400
        if len(stock_data) < 200:
            return jsonify({"message": "Not enough historical data for reliable prediction"}), 400
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return jsonify({"message": f"Error fetching data: {str(e)}"}), 500

    try:
        stock_data = preprocess_stock_data(stock_data)
    except Exception as e:
        return jsonify({"message": f"Preprocessing error: {str(e)}"}), 500

    features_to_normalize = ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_50', 'SMA_100', 'EMA_50', 'EMA_100', 'RSI_14', 'RSI_21', 'MACD', 'Signal_Line', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Middle_50', 'BB_Upper_50', 'BB_Lower_50', 'Volatility', 'Daily_Return', 'Volume_MA_50', 'Volume_MA_100', 'Dummy_Feature']
    scaler = MinMaxScaler()
    scaler_close = MinMaxScaler()
    try:
        scaler_close.fit(stock_data[['Close']])
        stock_data_normalized = stock_data.copy()
        stock_data_normalized[features_to_normalize] = scaler.fit_transform(stock_data[features_to_normalize])
        logger.debug(f"Normalized input shape: {stock_data_normalized[features_to_normalize].shape}")
    except Exception as e:
        logger.error(f"Normalization error: {e}")
        return jsonify({"message": f"Normalization error: {str(e)}"}), 500

    try:
        model_info = MODEL_MAP[sector]
        model_type = model_info['type']
        model_path = model_info['path']
        logger.debug(f"Attempting to load model from {model_path}")
        if model_type in ['random_forest', 'gradient_boosting', 'linear_regression', 'svm']:
            model = joblib.load(model_path)
        else:
            model = load_model(model_path)
        logger.debug("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return jsonify({"message": f"Error loading model: {str(e)}"}), 500

    if model_type in ['rnn', 'gru', 'cnn', 'lstm']:
        sequence_length = 100
        data_values = stock_data_normalized[features_to_normalize].values
        if len(data_values) < sequence_length:
            return jsonify({"message": "Not enough data for sequence-based prediction"}), 400
        x = data_values[-sequence_length:]
        x = x.reshape(1, sequence_length, len(features_to_normalize))
        logger.debug(f"Reshaped input shape for prediction: {x.shape}")
    else:
        x = stock_data_normalized[features_to_normalize].values[-1:]

    if prediction_range == '1week':
        days = 7
    elif prediction_range == '1month':
        days = 30
    else:
        days = 90

    def predict_future_days(model, input_data, scaler_close, days, is_sequence_model):
        predictions = []
        current_input = input_data.copy()
        for _ in range(days):
            if is_sequence_model:
                pred = model.predict(current_input, verbose=0)
                pred_value = float(scaler_close.inverse_transform(pred)[0, 0])
                predictions.append(pred_value)
                new_row = np.zeros((1, current_input.shape[2]))
                new_row[0, 0] = pred[0, 0]
                current_input = np.concatenate((current_input[:, 1:, :], new_row.reshape(1, 1, -1)), axis=1)
            else:
                pred = model.predict(current_input)
                pred_value = float(scaler_close.inverse_transform(pred.reshape(-1, 1))[0, 0])
                predictions.append(pred_value)
                current_input = current_input.copy()
                current_input[0, 0] = pred[0]
            if pred_value < 0:
                predictions[-1] = 0.0
        return predictions

    try:
        is_sequence_model = model_type in ['rnn', 'gru', 'cnn', 'lstm']
        predicted_prices = predict_future_days(model, x, scaler_close, days, is_sequence_model)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"message": f"Prediction error: {str(e)}"}), 500

    try:
        sample_rate = max(1, len(stock_data) // 100)
        hist_dates = stock_data.index.strftime('%Y-%m-%d').tolist()[::sample_rate]
        historical_prices = [float(p) for p in stock_data['Close'].tolist()[::sample_rate]]
        future_dates = [(datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=i)).strftime('%Y-%m-%d')
                       for i in range(1, days + 1)]
        dates = hist_dates + future_dates
        historical = historical_prices + [None] * days
        predicted = [None] * len(historical_prices) + predicted_prices
        last_close = float(stock_data['Close'].iloc[-1]) if not stock_data.empty else None
    except Exception as e:
        logger.error(f"Response preparation error: {e}")
        return jsonify({"message": f"Error preparing response: {str(e)}"}), 500

    return jsonify({
        "ticker": ticker,
        "endDate": end_date,
        "dates": dates,
        "historical": historical,
        "predicted": predicted,
        "lastClose": last_close,
        "sma50": [float(x) for x in stock_data['SMA_50'].tolist()[::sample_rate]],
        "sma100": [float(x) for x in stock_data['SMA_100'].tolist()[::sample_rate]],
        "ema50": [float(x) for x in stock_data['EMA_50'].tolist()[::sample_rate]],
        "ema100": [float(x) for x in stock_data['EMA_100'].tolist()[::sample_rate]],
        "volatility": [float(x) for x in stock_data['Volatility'].tolist()[::sample_rate]]
    })

@app.route('/')
def serve_frontend():
    with open('static1/index1.html', 'r') as file:
        html_content = file.read()
    return render_template_string(html_content)

@app.route('/results')
def show_results():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Stock Predictor - Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f0f2f5; }
                .header { background-color: #5e2bb8; color: white; padding: 10px 20px; display: flex; align-items: center; }
                .header h1 { margin: 0; font-size: 24px; }
                .container { max-width: 1200px; margin: 20px auto; padding: 0 20px; }
                .card { background-color: white; padding: 15px; margin-bottom: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
                .card h3 { color: #1a0dab; margin-top: 0; }
                .card p { color: #1a0dab; margin: 5px 0; }
                .chart-section { margin-bottom: 20px; }
                canvas { width: 100% !important; height: 400px !important; }
                .table-section { margin-bottom: 20px; }
                table { width: 100%; border-collapse: collapse; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .explanation { font-style: italic; color: #666; margin-top: 10px; }
                #error { color: red; text-align: center; margin-bottom: 15px; }
                #historicalData { text-align: center; margin-bottom: 15px; }
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <div class="header">
                <h1>Stock Predictor</h1>
            </div>
            <div class="container">
                <div id="error" style="display: none;"></div>
                <div id="historicalData"></div>

                <!-- Stock Price Chart -->
                <div class="card chart-section">
                    <h3>Stock Price Prediction</h3>
                    <canvas id="stockChart"></canvas>
                </div>

                <!-- Predicted Values Table -->
                <div class="card table-section">
                    <h3>Predicted Values</h3>
                    <table id="predictionTable">
                        <tr><th>Predicted Price ($)</th></tr>
                    </table>
                </div>

                <!-- SMA Chart -->
                <div class="card chart-section">
                    <h3>SMA (Simple Moving Average)</h3>
                    <canvas id="smaChart"></canvas>
                    <p class="explanation">SMA shows the average price over a set period, smoothing price data to identify trends.</p>
                </div>

                <!-- EMA Chart -->
                <div class="card chart-section">
                    <h3>EMA (Exponential Moving Average)</h3>
                    <canvas id="emaChart"></canvas>
                    <p class="explanation">EMA gives more weight to recent prices, making it more responsive to new information.</p>
                </div>

                <!-- Volatility Chart -->
                <div class="card chart-section">
                    <h3>Volatility</h3>
                    <canvas id="volatilityChart"></canvas>
                    <p class="explanation">Volatility measures the range between high and low prices, indicating the stock's price fluctuation and risk level.</p>
                </div>
            </div>

            <script>
                const urlParams = new URLSearchParams(window.location.search);
                const data = JSON.parse(decodeURIComponent(urlParams.get('data')));
                const errorDiv = document.getElementById('error');
                const historicalDataP = document.getElementById('historicalData');
                let stockChart, smaChart, emaChart, volatilityChart;

                if (data.message) {
                    errorDiv.textContent = data.message;
                    errorDiv.style.display = 'block';
                } else {
                    historicalDataP.textContent = `Historical Data for ${data.ticker}: Close Price on ${data.endDate}: $${data.lastClose?.toFixed(2) || 'N/A'}`;
                    if (stockChart) stockChart.destroy();
                    if (smaChart) smaChart.destroy();
                    if (emaChart) emaChart.destroy();
                    if (volatilityChart) volatilityChart.destroy();

                    // Stock Price Chart
                    const ctxStock = document.getElementById('stockChart').getContext('2d');
                    stockChart = new Chart(ctxStock, {
                        type: 'line',
                        data: {
                            labels: data.dates,
                            datasets: [
                                { label: 'Historical Price', data: data.historical, borderColor: 'blue', fill: false, pointRadius: 0, borderWidth: 2 },
                                { label: 'Predicted Price', data: data.predicted, borderColor: 'green', borderDash: [5, 5], fill: false, pointRadius: 3, borderWidth: 2 }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: { x: { title: { display: true, text: 'Date' } }, y: { title: { display: true, text: 'Price ($)' }, beginAtZero: false } },
                            plugins: { legend: { display: true }, title: { display: true, text: `Stock Price Prediction for ${data.ticker}` } }
                        }
                    });

                    // Predicted Values Table
                    const table = document.getElementById('predictionTable');
                    data.predicted.forEach((price, index) => {
                        if (price !== null) {
                            const row = table.insertRow();
                            row.insertCell(0).textContent = price.toFixed(2);
                        }
                    });

                    // SMA Chart
                    const ctxSMA = document.getElementById('smaChart').getContext('2d');
                    smaChart = new Chart(ctxSMA, {
                        type: 'line',
                        data: {
                            labels: data.dates.slice(0, data.historical.length),
                            datasets: [
                                { label: 'SMA 50', data: data.sma50.slice(0, data.historical.length), borderColor: 'orange', fill: false },
                                { label: 'SMA 100', data: data.sma100.slice(0, data.historical.length), borderColor: 'red', fill: false }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: { y: { title: { display: true, text: 'SMA ($)' } } },
                            plugins: { legend: { display: true }, title: { display: true, text: 'SMA for Historical Data' } }
                        }
                    });

                    // EMA Chart
                    const ctxEMA = document.getElementById('emaChart').getContext('2d');
                    emaChart = new Chart(ctxEMA, {
                        type: 'line',
                        data: {
                            labels: data.dates.slice(0, data.historical.length),
                            datasets: [
                                { label: 'EMA 50', data: data.ema50.slice(0, data.historical.length), borderColor: 'purple', fill: false },
                                { label: 'EMA 100', data: data.ema100.slice(0, data.historical.length), borderColor: 'pink', fill: false }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: { y: { title: { display: true, text: 'EMA ($)' } } },
                            plugins: { legend: { display: true }, title: { display: true, text: 'EMA for Historical Data' } }
                        }
                    });

                    // Volatility Chart
                    const ctxVolatility = document.getElementById('volatilityChart').getContext('2d');
                    volatilityChart = new Chart(ctxVolatility, {
                        type: 'line',
                        data: {
                            labels: data.dates.slice(0, data.historical.length),
                            datasets: [
                                { label: 'Volatility', data: data.volatility.slice(0, data.historical.length), borderColor: 'brown', fill: false }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: { y: { title: { display: true, text: 'Volatility ($)' } } },
                            plugins: { legend: { display: true }, title: { display: true, text: 'Volatility for Historical Data' } }
                        }
                    });
                }
            </script>
        </body>
        </html>
    ''')

if __name__ == '__main__':
    app.run(debug=True)