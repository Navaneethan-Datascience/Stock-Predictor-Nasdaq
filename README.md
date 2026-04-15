# Sector-wise Stock Predictor – Nasdaq (Web Application)

## 1. Introduction

This project is a sector-wise stock prediction web application for Nasdaq-listed companies, where separate machine learning (ML) and deep learning (DL) models are trained for different market sectors. Each model runs behind a web interface that dynamically fetches Nasdaq historical data from the Yahoo Finance API, enabling users to interactively select the sector, stock, and date range to obtain model-driven predictions.

## 2. Key Features

- **Sector-specific modeling**: Dedicated ML/DL models for each Nasdaq sector (e.g., Technology, Healthcare, Financials).
- **Web-based interface**: User-friendly application to run predictions without touching the code.
- **Dynamic data retrieval**: Uses Yahoo Finance API to fetch Nasdaq historical price data on demand.
- **Configurable inputs**:
  - Select **Stock Sector**
  - Select **Stock Code / Ticker**
  - Choose **Date Range** for historical data
- **On-the-fly prediction**: Models run in real time on the selected historical window to generate predictions.

## 3. Architecture Overview

1. **Frontend (Web App)**  
   - Page/forms to select:
     - Sector
     - Stock ticker
     - Start and end dates  
   - Triggers backend calls for data fetch and prediction.
2. **Backend**
   - Integrates with Yahoo Finance API for historical OHLCV data.
   - Routes the request to the appropriate sector-specific model.
   - Applies preprocessing and feature engineering consistent with training.
   - Returns predictions to the frontend.
3. **Models**
   - Multiple ML and DL models, trained separately per sector.
   - Can be extended or replaced with improved architectures over time.

## 4. Data & Source

- **Exchange**: Nasdaq  
- **Source**: Yahoo Finance API (historical OHLCV and related fields)  
- **Granularity**: Typically daily data (can be extended if needed).

## 5. Usage

### 5.1 Web Application (User Flow)

1. Open the web application in your browser.
2. Choose a **Stock Sector** from the dropdown (e.g., Technology).
3. Choose a **Stock Code / Ticker** within that sector.
4. Select the **Date Range** for historical data.
5. Click the **Predict** or **Submit** button.
6. The app:
   - Fetches historical data from Yahoo Finance for the given ticker and period.
   - Passes the data through the corresponding sector model.
   - Displays predictions (e.g., next-day price, return, or direction) and any associated plots/tables.

> Update screenshots/URLs once deployment details are finalized.

### 5.2 Local Development (Example)

```bash
git clone https://github.com/Navaneethan-Datascience/Stock-Predictor-Nasdaq.git
cd Stock-Predictor-Nasdaq

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Run the web app (example command; adjust to your framework)
python app.py
# or
uvicorn main:app --reload
```

Open the URL shown in the terminal (e.g., `http://127.0.0.1:8000`) and use the UI.

## 6. Project Structure

> Adapt this section to your actual folders and filenames.

```text
Stock-Predictor-Nasdaq/
│  ├─ static1              # Frontend templates / static file
│  ├─ model1           # Saved models per sector   
├─ app1.py    # Web app entry point
└─ README.md
```

## 7. Models

- **Per-Sector Training**  
  - Each sector has its own ML/DL model (or model ensemble).  
  - Models are trained using only data from that sector’s tickers.
- **Possible Algorithms**  
  - **ML**: Linear/Logistic Regression, Random Forest, Gradient Boosting, etc.  
  - **DL**: LSTM/GRU/Transformer-based time series models, etc.
- **Targets** (examples; specify precisely in your implementation)
  - Next-day closing price
  - Next-day return or direction
  - Multi-day horizon predictions

> Document chosen model types, features, and training details per sector once finalized.

## 8. Configuration

Typical configuration options (via YAML/JSON or Python configs):

- Sector list and mapping to tickers.
- Yahoo Finance symbols for each stock.
- Training/validation/test split dates.
- Model hyperparameters per sector.
- Default date ranges and UI settings.

## 9. Roadmap

- [ ] Finalize web UI flows and components.
- [ ] Harden Yahoo Finance integration (error handling, rate limits).
- [ ] Document all sector-specific models and metrics.
- [ ] Add visualization for predictions (plots, confidence bands, etc.).
- [ ] Add logging/monitoring and basic tests.
- [ ] Containerize and/or deploy (e.g., Docker + cloud platform).
