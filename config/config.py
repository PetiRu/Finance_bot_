# -*- coding: utf-8 -*-
"""
Configuration settings for the Crypto ML Trading System.
"""

import os

# Core system directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Ensure directories exist
for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# List of assets (TOP 10 EXCLUDING BTC)
SYMBOLS = [
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "DOT/USDT",
    "MATIC/USDT"
]

# Primary list of timeframes
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]

# Target configuration for modeling
LOOKAHEAD_PERIODS = 24  # candles ahead to evaluate target outcomes
ATR_MULTIPLIER_SL = 2.0  # SL = Price - (2.0 * ATR) for LONG
ATR_MULTIPLIER_TP1 = 1.5
ATR_MULTIPLIER_TP2 = 3.0
ATR_MULTIPLIER_TP3 = 5.0

# Partial Take-Profit allocation sizes (must sum to 1.0)
TP_ALLOCATIONS = {
    "TP1": 0.30,  # Sell 30% of position
    "TP2": 0.30,  # Sell 30% of position
    "TP3": 0.40   # Sell remaining 40% of position
}

# General trading system settings
RISK_PER_TRADE = 0.01          # 1.0% of portfolio equity risked per trade
INITIAL_CAPITAL = 10000.0      # Starting USD balance
MAKER_FEE = 0.0010              # Maker fee on Binance (0.1%)
TAKER_FEE = 0.001              # Taker fee on Binance (0.1%)
SLIPPAGE_RATE = 0.0005          # Simulated average slippage (0.05%)
TRAILING_STOP_TRIGGER = 2.0    # ATR multiplier trigger to move stop to breakeven or trail

# Validation / Splits
TIME_SERIES_SPLITS = 5
TRAIN_TEST_RATIO = 0.8  # 80/20 train/test split for time-series evaluation
MIN_DATA_ROWS = 500     # minimum history required to form high-quality features

# XGBoost hyperparameters
XGB_PARAMS = {
    "n_estimators": 250,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1
}

# LightGBM hyperparameters
LGBM_PARAMS = {
    "n_estimators": 250,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary",
    "metric": "binary_logloss",
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1
}

# Random Forest hyperparameters
RF_PARAMS = {
    "n_estimators": 150,
    "max_depth": 7,
    "min_samples_split": 10,
    "random_state": 42,
    "n_jobs": -1
}

# Optuna configuration
OPTUNA_TRIALS = 20  # Number of trials for parameter optimization
USE_OPTUNA = False  # Set to True to enable hyperparameter search in training
