"""
===============================================================================
Utilities: Data Pipeline & Dataset Builder for CNN Candlestick Predictor
===============================================================================
"""

from typing import Tuple
import numpy as np
import pandas as pd
import yfinance as yf


def load_and_prepare_data(ticker: str, start_date: str, trend_sma: int, support_sma: int) -> pd.DataFrame:
    """Downloads historical OHLCV data and calculates technical indicators."""
    print(f"[*] Downloading data for {ticker} from {start_date}...")
    df = yf.download(ticker, start=start_date, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.reset_index(inplace=True)
    
    # Calculate Technical Indicators (Moving Averages)
    df['trend'] = df['Close'].rolling(window=trend_sma).mean()
    df['support'] = df['Close'].rolling(window=support_window if 'support_window' in locals() else support_sma).mean()
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[+] Successfully processed {len(df)} rows of data.")
    return df


def build_dataset(
    df: pd.DataFrame, 
    candle_days: int, 
    return_days: int, 
    threshold_pct: float, 
    step_days: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Builds normalized feature matrices and multi-class categorical labels."""
    X, y = [], []
    n = len(df)
    
    for i in range(candle_days, n - return_days, step_days):
        window_slice = df.iloc[i - candle_days : i]
        
        # Normalize price action relative to window start close for stability
        base_price = window_slice['Close'].iloc[0]
        norm_open = window_slice['Open'] / base_price
        norm_high = window_slice['High'] / base_price
        norm_low = window_slice['Low'] / base_price
        norm_close = window_slice['Close'] / base_price
        
        # Construct 2D feature matrix (Time steps x Features)
        feature_matrix = np.column_stack([
            norm_open, norm_high, norm_low, norm_close,
            window_slice['trend'] / base_price,
            window_slice['support'] / base_price
        ])
        
        # Calculate Future Return Label
        current_close = df.loc[i - 1, 'Close']
        future_close = df.loc[i - 1 + return_days, 'Close']
        pct_change = ((future_close - current_close) / current_close) * 100.0
        
        if pct_change >= threshold_pct:
            label = 2  # Bullish / Up
        elif pct_change <= -threshold_pct:
            label = 0  # Bearish / Down
        else:
            label = 1  # Neutral / Stay
            
        X.append(feature_matrix)
        y.append(label)
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
