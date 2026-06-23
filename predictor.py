import os
import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import scikit-learn
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("Warning: 'scikit-learn' is not installed.")
    print("Please install dependencies: pip install scikit-learn pandas numpy matplotlib yfinance")
    sys.exit(1)

# Try to import yfinance for real stock data
YFINANCE_AVAILABLE = False
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    print("Notice: 'yfinance' is not installed. Script will use high-fidelity synthetic stock data.")

def generate_synthetic_data(symbol="MOCK", days=500):
    """
    Generates realistic synthetic stock data using Geometric Brownian Motion (GBM)
    with cyclical components and simulated weekends.
    """
    np.random.seed(42)
    start_date = datetime.date.today() - datetime.timedelta(days=days * 7 // 5)
    
    # GBM Parameters
    mu = 0.0002          # Expected return (drift)
    sigma = 0.015        # Daily volatility
    S0 = 150.0           # Initial price
    
    prices = [S0]
    dates = []
    
    current_date = start_date
    while len(prices) < days:
        if current_date.weekday() < 5:  # Monday to Friday
            # Random walk component
            epsilon = np.random.normal(0, 1)
            # Add a slow wave cycle (market cycles)
            cycle = 0.003 * np.sin(len(prices) / 30.0)
            # Daily return
            pct_change = (mu + cycle) + sigma * epsilon
            next_price = prices[-1] * (1 + pct_change)
            prices.append(next_price)
            dates.append(current_date)
        current_date += datetime.timedelta(days=1)
        
    prices = prices[1:]  # remove S0
    
    # Generate High, Low, Open, Volume based on Close
    df = pd.DataFrame(index=dates)
    df['Close'] = prices
    df['Open'] = df['Close'].shift(1) * (1 + np.random.normal(0, 0.002, days))
    df.iloc[0, df.columns.get_loc('Open')] = S0 * (1 + np.random.normal(0, 0.002))
    
    df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + abs(np.random.normal(0.005, 0.003, days)))
    df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - abs(np.random.normal(0.005, 0.003, days)))
    df['Volume'] = np.random.lognormal(15, 0.5, days).astype(int)
    
    print(f"Generated {days} trading days of synthetic data for {symbol}.")
    return df

def fetch_stock_data(symbol, start_date, end_date):
    """
    Attempts to download stock data using yfinance. Falls back to synthetic if failure/missing.
    """
    if YFINANCE_AVAILABLE:
        try:
            print(f"Attempting to download data for ticker '{symbol}' from Yahoo Finance...")
            df = yf.download(symbol, start=start_date, end=end_date)
            if not df.empty and len(df) > 50:
                print(f"Successfully downloaded {len(df)} rows of data.")
                # yfinance can return multi-index columns, flatten them if needed
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df
            else:
                print("Ticker returned empty data or too few rows. Falling back to synthetic data.")
        except Exception as e:
            print(f"Failed to download data due to network/API error: {e}. Falling back to synthetic data.")
    
    # Fallback to synthetic
    return generate_synthetic_data(symbol, days=600)

def calculate_technical_indicators(df):
    """
    Calculates technical features: Lags, Simple Moving Averages, RSI.
    """
    df = df.copy()
    
    # 1. Price Lags (Autoregressive features)
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_3'] = df['Close'].shift(3)
    df['Lag_5'] = df['Close'].shift(5)
    
    # 2. Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_15'] = df['Close'].rolling(window=15).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    
    # 3. Volatility
    df['Volatility_5'] = df['Close'].pct_change().rolling(window=5).std()
    
    # 4. Relative Strength Index (RSI - 14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Drop rows with NaN values resulting from indicators
    df = df.dropna()
    return df

def prepare_features(df):
    """
    Prepares feature matrix X and target vector y.
    Target: Next day's Close price (Close_t+1)
    """
    # Shift the close price back by 1 day to represent "Next Day's Close"
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()  # drop last row since it doesn't have a target
    
    feature_cols = [
        'Close', 'Open', 'High', 'Low', 'Volume',
        'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5',
        'SMA_5', 'SMA_15', 'SMA_30',
        'Volatility_5', 'RSI_14'
    ]
    
    X = df[feature_cols]
    y = df['Target']
    return X, y, df

def train_and_evaluate(X, y):
    """
    Splits the data sequentially (to respect time series), trains models, and returns metrics.
    """
    # Time series split: Train on first 80%, Test on last 20%
    split_idx = int(len(X) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)
    
    # 2. Random Forest Model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)  # Random forest doesn't strictly need scaling
    rf_preds = rf_model.predict(X_test)
    
    # Evaluation
    metrics = {}
    for name, y_pred in [("Linear Regression", lr_preds), ("Random Forest", rf_preds)]:
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        metrics[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "predictions": y_pred}
        print(f"\n--- {name} Results ---")
        print(f"Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"R-squared Score (R2): {r2:.4f}")
        
    return lr_model, rf_model, scaler, y_test, metrics

def forecast_future(model, scaler, last_row, feature_cols, days_to_forecast=10):
    """
    Forecasts the next N trading days autoregressively.
    For simplicity, we use the Linear Regression model and update lags sequentially.
    """
    forecast = []
    current_features = last_row.copy()
    
    # Prepare a rolling dataframe to calculate moving averages/indicators dynamically
    # We will need at least 35 rows of historical data to compute the technical indicators
    # for each successive day.
    history_df = last_row.to_frame().T if isinstance(last_row, pd.Series) else last_row.copy()
    
    for day in range(days_to_forecast):
        # 1. Calculate technical indicators for the last row in history_df
        # Create features specifically for prediction
        # To do this safely, we will extract the values needed
        last_close = history_df['Close'].iloc[-1]
        
        # Calculate features manually for the next step to avoid rolling complications on predictions
        # Fill features based on historical logs
        pred_features = {}
        pred_features['Close'] = last_close
        pred_features['Open'] = history_df['Open'].iloc[-1]
        pred_features['High'] = history_df['High'].iloc[-1]
        pred_features['Low'] = history_df['Low'].iloc[-1]
        pred_features['Volume'] = history_df['Volume'].iloc[-1]
        
        pred_features['Lag_1'] = history_df['Close'].iloc[-1]
        pred_features['Lag_2'] = history_df['Close'].iloc[-2] if len(history_df) >= 2 else last_close
        pred_features['Lag_3'] = history_df['Close'].iloc[-3] if len(history_df) >= 3 else last_close
        pred_features['Lag_5'] = history_df['Close'].iloc[-5] if len(history_df) >= 5 else last_close
        
        pred_features['SMA_5'] = history_df['Close'].iloc[-5:].mean()
        pred_features['SMA_15'] = history_df['Close'].iloc[-15:].mean() if len(history_df) >= 15 else history_df['Close'].mean()
        pred_features['SMA_30'] = history_df['Close'].iloc[-30:].mean() if len(history_df) >= 30 else history_df['Close'].mean()
        
        # Volatility & RSI approximations
        pred_features['Volatility_5'] = history_df['Close'].iloc[-5:].pct_change().std()
        if np.isnan(pred_features['Volatility_5']) or pred_features['Volatility_5'] == 0:
            pred_features['Volatility_5'] = 0.015
            
        pred_features['RSI_14'] = 50.0  # neutral default for simplicity in future forecasting
        
        # Structure as DataFrame to match training feature names and avoid warnings
        feat_df = pd.DataFrame([pred_features], columns=feature_cols)
        feat_vector_scaled = scaler.transform(feat_df)
        
        # Predict Close price for next day
        next_close = model.predict(feat_vector_scaled)[0]
        forecast.append(next_close)
        
        # Append simulated row to history_df
        next_row = {
            'Close': next_close,
            'Open': last_close,
            'High': max(last_close, next_close) * 1.002,
            'Low': min(last_close, next_close) * 0.998,
            'Volume': int(history_df['Volume'].iloc[-1] * 0.95)
        }
        history_df = pd.concat([history_df, pd.DataFrame(next_row, index=[history_df.index[-1] + datetime.timedelta(days=1)])])
        
    return forecast

def main():
    print("=" * 60)
    print("                 AI Stock Market Predictor                      ")
    print("=" * 60)
    
    # Configuration
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    
    # 1. Load Data
    raw_df = fetch_stock_data(ticker, start_date, end_date)
    
    # 2. Engineer Technical Features
    print("Engineering technical indicators...")
    featured_df = calculate_technical_indicators(raw_df)
    
    # 3. Separate Features and Target
    X, y, clean_df = prepare_features(featured_df)
    
    # 4. Train Models & Evaluate
    lr_model, rf_model, scaler, y_test, metrics = train_and_evaluate(X, y)
    
    # 5. Forecast Future Prices (Next 10 Days)
    feature_cols = list(X.columns)
    last_historical_rows = clean_df.tail(35) # we need history to calculate technicals for forecasting
    forecast_days = 10
    
    print(f"\nForecasting next {forecast_days} days using Linear Regression...")
    future_preds = forecast_future(lr_model, scaler, last_historical_rows, feature_cols, days_to_forecast=forecast_days)
    
    # Generate future dates (skipping weekends is simplified to calendar days here)
    last_date = clean_df.index[-1]
    future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(forecast_days)]
    
    print("\n--- 10-Day Forecast ---")
    for date, price in zip(future_dates, future_preds):
        print(f"Date: {date.strftime('%Y-%m-%d')} | Predicted Close: ${price:.2f}")
        
    # 6. Visualization
    print("\nGenerating prediction plots...")
    plt.figure(figsize=(14, 7))
    plt.style.use('dark_background' if 'dark_background' in plt.style.available else 'default')
    
    # Plot historical data (past 120 days for readability)
    plot_history_limit = min(120, len(clean_df))
    hist_to_plot = clean_df.iloc[-plot_history_limit:]
    plt.plot(hist_to_plot.index, hist_to_plot['Close'], label='Actual Price', color='#38bdf8', linewidth=2)
    
    # Plot predictions over test period
    # Match dates of y_test to see model fits
    test_dates = y_test.index
    plt.plot(test_dates, metrics["Linear Regression"]["predictions"], 
             label='Linear Regression Fit', color='#f43f5e', linestyle='--', alpha=0.8)
    plt.plot(test_dates, metrics["Random Forest"]["predictions"], 
             label='Random Forest Fit', color='#10b981', linestyle=':', alpha=0.8)
    
    # Plot future predictions
    plt.plot(future_dates, future_preds, label='Future Forecast (LR)', color='#eab308', marker='o', linewidth=2.5)
    
    plt.title(f"{ticker} Stock Price Prediction & Future Forecast", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price ($)", fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=10, loc='upper left')
    
    # Save the plot
    output_plot_path = "prediction_plot.png"
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Success! Prediction plot saved as: '{output_plot_path}'")
    print("=" * 60)

if __name__ == "__main__":
    main()
