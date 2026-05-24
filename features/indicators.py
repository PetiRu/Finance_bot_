# -*- coding: utf-8 -*-
"""
Advanced technical feature design pipeline in pure Pandas.
"""

import numpy as np
import pandas as pd


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).copy()
    loss = (-delta.where(delta < 0, 0)).copy()
    avg_gain_ewm = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss_ewm = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain_ewm / (avg_loss_ewm + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def detect_swings(high, low, window=5):
    length = len(high)
    swing_highs = np.zeros(length)
    swing_lows = np.zeros(length)
    high_arr = high.values
    low_arr = low.values
    
    for i in range(window, length - window):
        center_high, center_low = high_arr[i], low_arr[i]
        is_high = all(high_arr[i - j] < center_high for j in range(1, window + 1)) and all(high_arr[i + j] <= center_high for j in range(1, window + 1))
        is_low = all(low_arr[i - j] > center_low for j in range(1, window + 1)) and all(low_arr[i + j] >= center_low for j in range(1, window + 1))
        if is_high:
            swing_highs[i+window] = center_high
        if is_low:
            swing_lows[i+window] = center_low
            
    sh_series = pd.Series(swing_highs).replace(0, np.nan).ffill().fillna(high)
    sl_series = pd.Series(swing_lows).replace(0, np.nan).ffill().fillna(low)
    prev_sh, prev_sl = sh_series.shift(1), sl_series.shift(1)
    return sh_series, sl_series, (sh_series > prev_sh).astype(int), (sh_series <= prev_sh).astype(int), (sl_series > prev_sl).astype(int), (sl_series <= prev_sl).astype(int)


def build_advanced_features(df, window_swings=5):
    if len(df) < 50: return df.copy()
    feats = df.copy()
    close, high, low, volume = feats["close"], feats["high"], feats["low"], feats["volume"]
    
    feats["log_ret"] = np.log(close / close.shift(1))
    for p in [1, 3, 5, 10]:
        feats[f"mom_{p}"] = close.pct_change(periods=p)
        
    for s in [20, 50, 200]:
        feats[f"ema_{s}"] = close.ewm(span=s, adjust=False).mean()
        feats[f"close_to_ema{s}"] = close / feats[f"ema_{s}"] - 1.0
        
    feats["rsi_14"] = compute_rsi(close, 14) / 100.0
    m_l, m_s, m_h = compute_macd(close)
    feats["macd_line"], feats["macd_signal"], feats["macd_hist"] = m_l / close, m_s / close, m_h / close
    
    rolling_std = close.rolling(window=20).std()
    feats["bb_mid"] = close.rolling(window=20).mean()
    feats["bb_upper"] = feats["bb_mid"] + 2.0 * rolling_std
    feats["bb_lower"] = feats["bb_mid"] - 2.0 * rolling_std
    feats["bb_width"] = (feats["bb_upper"] - feats["bb_lower"]) / feats["bb_mid"]
    feats["bb_pct"] = (close - feats["bb_lower"]) / (feats["bb_upper"] - feats["bb_lower"] + 1e-10)
    
    feats["atr_14"] = compute_atr(high, low, close, 14)
    feats["atr_pct"] = feats["atr_14"] / close
    
    vol_mean, vol_std = volume.rolling(window=20).mean(), volume.rolling(window=20).std()
    feats["volume_ratio"] = volume / (vol_mean + 1e-10)
    
    sh, sl, hh, lh, hl, ll = detect_swings(high, low, window_swings)
    feats["swing_high"], feats["swing_low"] = sh / close - 1.0, sl / close - 1.0
    feats["trend_strength"] = (((feats["ema_20"] > feats["ema_50"]).astype(float) * 2 - 1) + (close > feats["ema_200"]).astype(float) * 2 - 1 + (feats["rsi_14"] - 0.5) * 2) / 3.0
    
    atr_rolling_median = feats["atr_pct"].rolling(window=100).median()
    atr_rolling_mad = (feats["atr_pct"] - atr_rolling_median).abs().rolling(window=100).median()
    z_vol = (feats["atr_pct"] - atr_rolling_median) / (atr_rolling_mad + 1e-10)
    feats["vol_regime"] = np.where(z_vol > 1.5, 2, np.where(z_vol < -1.0, 0, 1))
    
    return feats.dropna()
