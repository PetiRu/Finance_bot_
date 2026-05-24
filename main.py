# -*- coding: utf-8 -*-
"""
Main driver orchestration.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from config.config import SYMBOLS, TIMEFRAMES, XGB_PARAMS, LGBM_PARAMS, RF_PARAMS, OUTPUT_DIR, DATA_DIR
from data_loader.downloader import BinanceDataDownloader
from features.indicators import build_advanced_features
from signals.generator import apply_triple_barrier_labeling, SignalGenerator
from models.ensemble import CryptoMLEnsembleSuite
from backtest.engine import DynamicPartialExitBacktester
from utils import filter_correlated_assets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CryptoMLSystem")


def generate_synthetic_historical_data(symbol, timeframe, periods=1000):
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=periods, freq="h" if "h" in timeframe else "d")
    price = 2800.0 if symbol == "ETH/USDT" else 140.0 if symbol == "SOL/USDT" else 1.0
    returns = np.random.normal(0.0001, 0.012, periods)
    price_series = price * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "timestamp": dates.astype(int) // 10**6,
        "datetime": dates,
        "open": price_series * 0.998,
        "high": price_series * 1.012,
        "low": price_series * 0.988,
        "close": price_series,
        "volume": np.random.uniform(10000, 500000, periods)
    })


def run_pipeline():
    downloader = BinanceDataDownloader(symbols=SYMBOLS, timeframes=TIMEFRAMES)
    all_data = {}
    
    for symbol in SYMBOLS:
        symbol_clean = symbol.replace("/", "_")
        all_data[symbol] = {}
        for tf in TIMEFRAMES:
            file_path = os.path.join(DATA_DIR, f"{symbol_clean}_{tf}.parquet")
            df = generate_synthetic_historical_data(symbol, tf)
            df.to_parquet(file_path, index=False)
            all_data[symbol][tf] = df
            
    active_symbols = filter_correlated_assets({sym: all_data[sym]["1h"] for sym in SYMBOLS})
    signals_output, backtest_summaries = [], {}
    
    for symbol in active_symbols:
        raw_df = all_data[symbol]["1h"].copy()
        feats_df = build_advanced_features(raw_df)
        labeled_df = apply_triple_barrier_labeling(feats_df, lookahead=24)
        
        suite = CryptoMLEnsembleSuite(config_xgb=XGB_PARAMS, config_lgb=LGBM_PARAMS, config_rf=RF_PARAMS)
        suite.train_classifiers(labeled_df)
        suite.train_regressors(labeled_df)
        
        predictions_proba = suite.predict_direction(labeled_df)
        predictions_ret = suite.predict_return(labeled_df)
        
        live_sig = SignalGenerator(suite).generate_current_signal(labeled_df, symbol, "1h")
        if live_sig: signals_output.append(live_sig)
            
        eq_df, closed_trades, metrics = DynamicPartialExitBacktester(10000.0).run_backtest(labeled_df, predictions_proba, predictions_ret)
        backtest_summaries[symbol] = {"metrics": metrics, "equity": eq_df["capital"].tolist()[-200:]}
        
    with open(os.path.join(OUTPUT_DIR, "system_report.json"), "w") as f:
        json.dump({"timestamp": datetime.now(timezone.utc).isoformat(), "signals": signals_output, "backtests": backtest_summaries}, f, indent=4)
    print("Pipeline compilation completed in standalone mode.")


if __name__ == "__main__":
    run_pipeline()
