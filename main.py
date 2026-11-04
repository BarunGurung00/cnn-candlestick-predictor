"""
===============================================================================
CNN Candlestick Pattern Predictor - Compact & Production-Ready
===============================================================================
Description:
    Downloads financial OHLCV data, calculates technical indicators (trend & support SMAs),
    renders candlestick data vectors, assigns categorical returns labels (Up=1, Stay=0, Down=-1),
    and trains a compact TensorFlow/Keras CNN classifier.
===============================================================================
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras import layers, models

# =============================================================================
# 1. Configuration & Parameters
# =============================================================================
class Config:
    TICKER = "BTC-USD"
    START_DATE = "2020-01-01"
    TREND_SMA = 45
    SUPPORT_SMA = 125
    CANDLE_DAYS = 30
    RETURN_DAYS = 10
    THRESHOLD_PCT = 5.0
    STEP_DAYS = 5
    EPOCHS = 10
    BATCH_SIZE = 32
    RANDOM_SEED = 42

np.random.seed(Config.RANDOM_SEED)
tf.random.set_seed(Config.RANDOM_SEED)

# =============================================================================
# 2. Data Pipeline & Feature Engineering
# =============================================================================
def load_and_prepare_data() -> pd.DataFrame:
    print(f"[*] Downloading data for {Config.TICKER}...")
    df = yf.download(Config.TICKER, start=Config.START_DATE, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    
    # Calculate Technical Indicators
    df['trend'] = df['Close'].rolling(window=Config.TREND_SMA).mean()
    df['support'] = df['Close'].rolling(window=Config.SUPPORT_SMA).mean()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[+] Processed {len(df)} trading rows.")
    return df

# =============================================================================
# 3. Dataset Generator (Feature Matrices & Labels)
# =============================================================================
def build_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    n = len(df)
    
    for i in range(Config.CANDLE_DAYS, n - Config.RETURN_DAYS, Config.STEP_DAYS):
        window_slice = df.iloc[i - Config.CANDLE_DAYS : i]
        
        # Normalize price action relative to window start close for CNN matrix stability
        base_price = window_slice['Close'].iloc[0]
        norm_open = window_slice['Open'] / base_price
        norm_high = window_slice['High'] / base_price
        norm_low = window_slice['Low'] / base_price
        norm_close = window_slice['Close'] / base_price
        
        # Construct a 2D feature matrix (Time steps x Features: Open, High, Low, Close, Trend, Support)
        feature_matrix = np.column_stack([
            norm_open, norm_high, norm_low, norm_close,
            window_slice['trend'] / base_price,
            window_slice['support'] / base_price
        ])
        
        # Calculate Future Return Label
        current_close = df.loc[i - 1, 'Close']
        future_close = df.loc[i - 1 + Config.RETURN_DAYS, 'Close']
        pct_change = ((future_close - current_close) / current_close) * 100.0
        
        if pct_change >= Config.THRESHOLD_PCT:
            label = 2  # Bullish / Up
        elif pct_change <= -Config.THRESHOLD_PCT:
            label = 0  # Bearish / Down
        else:
            label = 1  # Neutral / Stay
            
        X.append(feature_matrix)
        y.append(label)
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# =============================================================================
# 4. Convolutional Neural Network Architecture
# =============================================================================
def create_cnn_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')  # 3 Classes: Down, Stay, Up
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# =============================================================================
# 5. Main Execution Flow
# =============================================================================
if __name__ == "__main__":
    df = load_and_prepare_data()
    X, y = build_dataset(df)
    print(f"[+] Generated Dataset shapes -> X: {X.shape}, y: {y.shape}")
    
    # Train / Test Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Initialize and Train Model
    model = create_cnn_model(input_shape=(X.shape[1], X.shape[2]))
    model.summary()
    
    print("[*] Training CNN model...")
    history = model.fit(
        X_train, y_train,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_data=(X_test, y_test)
    )
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[+] Test Accuracy: {acc * 100:.2f}% | Test Loss: {loss:.4f}")
