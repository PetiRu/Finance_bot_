# -*- coding: utf-8 -*-
"""
Robust, production-grade Binance historical OHLCV downloader using CCXT.
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd
import ccxt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DataLoader")


class BinanceDataDownloader:
    def __init__(self, symbols=None, timeframes=None, data_dir="data"):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.symbols = symbols or [
            "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT",
            "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "MATIC/USDT"
        ]
        self.timeframes = timeframes or ["1m", "5m", "15m", "1h", "4h"]
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def timeframe_to_delta(self, timeframe):
        unit = timeframe[-1]
        amount = int(timeframe[:-1])
        if unit == 'm': return timedelta(minutes=amount)
        elif unit == 'h': return timedelta(hours=amount)
        elif unit == 'd': return timedelta(days=amount)
        raise ValueError(f"Unknown timeframe suffix: {unit}")

    def fetch_ohlcv_chunk(self, symbol, timeframe, since_ms, limit=1000):
        max_retries = 5
        backoff = 1.0
        for attempt in range(max_retries):
            try:
                return self.exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
            except Exception as e:
                logger.warning(f"Fetch failed, retrying in {backoff}s... Error: {e}")
                time.sleep(backoff)
                backoff *= 2.0
        return []

    def download_candles(self, symbol, timeframe, start_date, end_date):
        since = int(start_date.replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(end_date.replace(tzinfo=timezone.utc).timestamp() * 1000)
        all_candles = []
        delta = self.timeframe_to_delta(timeframe)
        expected_ms_increment = int(delta.total_seconds() * 1000)

        logger.info(f"Downloading {symbol} {timeframe} close...")
        last_ms = since
        stuck_counter = 0

        while last_ms < end_ms:
            candles = self.fetch_ohlcv_chunk(symbol, timeframe, last_ms, limit=1000)
            if not candles: break
            all_candles.extend(candles)
            last_candle_ms = candles[-1][0]
            if last_candle_ms <= last_ms:
                last_ms += expected_ms_increment * 1000
                stuck_counter += 1
                if stuck_counter > 5: break
            else:
                last_ms = last_candle_ms + expected_ms_increment
                stuck_counter = 0
            time.sleep(0.12)

        if not all_candles: return pd.DataFrame()
        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def sync_all(self, years=6):
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=years * 365)
        for symbol in self.symbols:
            symbol_clean = symbol.replace("/", "_")
            for tf in self.timeframes:
                file_path = os.path.join(self.data_dir, f"{symbol_clean}_{tf}.parquet")
                local_start = start_date
                existing_df = pd.DataFrame()
                if os.path.exists(file_path):
                    try:
                        existing_df = pd.read_parquet(file_path)
                        if not existing_df.empty:
                            last_t = existing_df["timestamp"].max()
                            local_start = pd.to_datetime(last_t, unit="ms", utc=True)
                    except Exception as e:
                        logger.error(f"Failed loading previous Parquet: {e}")

                new_df = self.download_candles(symbol, tf, local_start, end_date)
                if new_df.empty: continue
                combined_df = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
                combined_df = combined_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                combined_df.to_parquet(file_path, compression="snappy", index=False)
