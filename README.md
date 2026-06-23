#  AI Stock Market Predictor

A machine learning pipeline that fetches (or synthesizes) stock price data, engineers technical indicators, trains predictive models, and forecasts the next 10 trading days — with a visualization of results.

---

##  Features

- **Real or synthetic data** — pulls live data from Yahoo Finance via `yfinance`; falls back to high-fidelity Geometric Brownian Motion (GBM) simulation if unavailable
- **Technical indicator engineering** — lag features, simple moving averages (SMA), RSI, and rolling volatility
- **Two ML models** — Linear Regression and Random Forest Regressor
- **10-day autoregressive forecast** — each predicted price feeds the next day's input
- **Visualization** — saves a chart comparing actual prices, model fits, and future forecast to `prediction_plot.png`

---

##  Requirements

Python 3.8+ is recommended.

```bash
pip install scikit-learn pandas numpy matplotlib yfinance
```

| Package | Purpose |
|---|---|
| `scikit-learn` | ML models and evaluation metrics |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Chart generation |
| `yfinance` | Real stock data (optional) |

> **Note:** `yfinance` is optional. If it's not installed or the ticker lookup fails, the script automatically uses synthetic data.

---

##  Usage

```bash
python predictor.py
```

To change the ticker or date range, edit these lines near the top of `main()` in `predictor.py`:

```python
ticker = "AAPL"
start_date = "2023-01-01"
end_date = datetime.date.today().strftime("%Y-%m-%d")
```

---

##  How It Works

### 1. Data Loading
Attempts to download OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance. On failure, generates synthetic data using Geometric Brownian Motion with a sinusoidal market-cycle overlay.

### 2. Feature Engineering
Computes the following technical indicators from the raw Close price:

| Feature | Description |
|---|---|
| `Lag_1/2/3/5` | Closing prices from N days ago |
| `SMA_5/15/30` | Simple moving averages over 5, 15, 30 days |
| `Volatility_5` | Rolling 5-day standard deviation of returns |
| `RSI_14` | 14-day Relative Strength Index |

### 3. Model Training
Data is split **chronologically** (80% train / 20% test) to avoid look-ahead bias. Two models are trained:

- **Linear Regression** — trained on standardized features via `StandardScaler`
- **Random Forest** — 100 estimators, trained on raw features

Models are evaluated on the test set using MAE, RMSE, and R².

### 4. Forecasting
The Linear Regression model autoregressively predicts the next 10 trading days. Each predicted Close price is fed back as input for the following day. RSI defaults to 50 (neutral) for future steps.

### 5. Output
- Prints a 10-day price forecast table to the console
- Saves `prediction_plot.png` with historical prices, model fits, and the future forecast

---

##  Output Example
--- Linear Regression Results ---

Mean Absolute Error (MAE): $1.23

Root Mean Squared Error (RMSE): $1.87

R-squared Score (R2): 0.9941
--- 10-Day Forecast ---

Date: 2024-07-01 | Predicted Close: $192.45

Date: 2024-07-02 | Predicted Close: $193.10

...
---

##  Project Structure
antigravity/

├── .venv/               # Python virtual environment

├── app.js               # JavaScript logic (frontend interactions)

├── index.html           # Main UI / entry point for the web interface

├── styles.css           # Stylesheet for the web interface

├── predictor.py         # Core ML pipeline (data loading, training, forecasting)

├── prediction_plot.png  # Generated chart output (created after running predictor.py)

└── README.md

> The project combines a **Python ML backend** (`predictor.py`) with a **web frontend** (`index.html`, `app.js`, `styles.css`), displaying prediction results in a browser-based UI.

---

##  Limitations

- **Not financial advice.** This is an educational ML demo, not a trading tool.
- Autoregressive forecasting compounds errors over time, producing overly smooth predictions.
- RSI and volatility are approximated for future steps, reducing forecast realism.
- The model has no awareness of news, earnings, macroeconomic events, or market sentiment.

---


