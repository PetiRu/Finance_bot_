# -*- coding: utf-8 -*-
"""
Live Paper Trading Demo - Real-time signal generation without risking capital
"""

import os
import json
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import ccxt

from config.config import SYMBOLS, TIMEFRAMES, XGB_PARAMS, LGBM_PARAMS, RF_PARAMS, OUTPUT_DIR, DATA_DIR
from features.indicators import build_advanced_features
from signals.generator import apply_triple_barrier_labeling, SignalGenerator
from models.ensemble import CryptoMLEnsembleSuite

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("LiveTradingSystem")


class LivePaperTradingBot:
    """Real-time trading signal generator - Paper Trading (no real money)"""
    
    def __init__(self, symbols=None, timeframe="1h", check_interval=3600):
        """
        Args:
            symbols: List of trading pairs (e.g., ["ETH/USDT", "SOL/USDT"])
            timeframe: Timeframe for signals ("1h", "4h", "1d")
            check_interval: Seconds between signal checks (default 1 hour)
        """
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.symbols = symbols or SYMBOLS
        self.timeframe = timeframe
        self.check_interval = check_interval
        self.models = {}
        self.active_trades = {}
        self.trade_history = []
        
        logger.info(f"🤖 Live Paper Trading Bot Initialized for {len(self.symbols)} symbols")
        logger.info(f"⏰ Check interval: {check_interval}s ({check_interval/3600:.1f}h)")
    
    def fetch_live_ohlcv(self, symbol, timeframe, limit=100):
        """Fetch real live data from Binance"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            logger.info(f"✅ Fetched {len(df)} candles for {symbol} {timeframe}")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to fetch {symbol}: {e}")
            return pd.DataFrame()
    
    def train_models(self, symbol):
        """Train ensemble models for a symbol"""
        logger.info(f"📊 Training models for {symbol}...")
        
        # Fetch training data (past 200 candles)
        df = self.fetch_live_ohlcv(symbol, self.timeframe, limit=200)
        if df.empty:
            return False
        
        # Build features
        df_features = build_advanced_features(df)
        if len(df_features) < 50:
            logger.warning(f"⚠️ Insufficient data for {symbol}")
            return False
        
        # Label data
        df_labeled = apply_triple_barrier_labeling(df_features, lookahead=12)
        
        # Train ensemble
        suite = CryptoMLEnsembleSuite(
            config_xgb=XGB_PARAMS, 
            config_lgb=LGBM_PARAMS, 
            config_rf=RF_PARAMS
        )
        suite.train_classifiers(df_labeled)
        suite.train_regressors(df_labeled)
        
        self.models[symbol] = {"suite": suite, "last_features": df_features}
        logger.info(f"✅ Models trained for {symbol}")
        return True
    
    def generate_signal(self, symbol):
        """Generate live trading signal for a symbol"""
        if symbol not in self.models:
            return None
        
        # Fetch latest candle
        df = self.fetch_live_ohlcv(symbol, self.timeframe, limit=100)
        if df.empty or len(df) < 30:
            return None
        
        # Build features on latest data
        df_features = build_advanced_features(df)
        if df_features.empty:
            return None
        
        # Get latest row with features
        latest_row = df_features.iloc[-1:]
        suite = self.models[symbol]["suite"]
        
        # Generate signal
        signal_gen = SignalGenerator(suite)
        signal = signal_gen.generate_current_signal(latest_row, symbol, self.timeframe)
        
        if signal:
            signal["timestamp"] = datetime.now(timezone.utc).isoformat()
            signal["price"] = float(latest_row["close"].values[0])
            signal["atr"] = float(latest_row["atr_14"].values[0])
        
        return signal
    
    def execute_paper_trade(self, signal):
        """Simulate trade execution (no real money involved)"""
        symbol = signal["symbol"]
        direction = signal["direction"]
        entry = signal["entry"]
        
        if direction == "NEUTRAL":
            return None
        
        trade = {
            "id": len(self.trade_history) + 1,
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "stop_loss": signal["stop_loss"],
            "tp_levels": signal["tp_levels"],
            "confidence": signal["confidence"],
            "status": "OPEN",
            "entry_size": 1.0  # Paper trading - fixed size
        }
        
        self.active_trades[f"{symbol}_{direction}"] = trade
        self.trade_history.append(trade)
        
        logger.info(f"📈 OPENED {direction} trade on {symbol} @ {entry} | SL: {signal['stop_loss']} | TP: {signal['tp_levels'][0]}")
        return trade
    
    def check_trade_status(self):
        """Check if any open trades hit SL or TP"""
        closed_trades = []
        
        for trade_key, trade in list(self.active_trades.items()):
            symbol = trade["symbol"]
            df = self.fetch_live_ohlcv(symbol, self.timeframe, limit=5)
            
            if df.empty:
                continue
            
            current_price = float(df["close"].iloc[-1])
            entry = trade["entry_price"]
            sl = trade["stop_loss"]
            tp1 = trade["tp_levels"][0]
            
            # Check stop loss
            if (trade["direction"] == "LONG" and current_price <= sl) or \
               (trade["direction"] == "SHORT" and current_price >= sl):
                trade["status"] = "CLOSED_SL"
                trade["exit_price"] = current_price
                trade["pnl"] = (current_price - entry) * (1 if trade["direction"] == "LONG" else -1)
                logger.warning(f"⛔ STOPPED OUT {trade['symbol']} @ {current_price} | Loss: {trade['pnl']:.2f}")
                del self.active_trades[trade_key]
                closed_trades.append(trade)
            
            # Check take profit
            elif (trade["direction"] == "LONG" and current_price >= tp1) or \
                 (trade["direction"] == "SHORT" and current_price <= tp1):
                trade["status"] = "CLOSED_TP"
                trade["exit_price"] = current_price
                trade["pnl"] = (current_price - entry) * (1 if trade["direction"] == "LONG" else -1)
                logger.info(f"✅ TAKE PROFIT HIT {trade['symbol']} @ {current_price} | Gain: {trade['pnl']:.2f}")
                del self.active_trades[trade_key]
                closed_trades.append(trade)
        
        return closed_trades
    
    def generate_report(self):
        """Generate trading report"""
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_trades": len(self.active_trades),
            "closed_trades": len(self.trade_history) - len(self.active_trades),
            "active_positions": list(self.active_trades.values()),
            "trade_history": self.trade_history,
        }
        
        # Calculate stats
        closed = [t for t in self.trade_history if t["status"] != "OPEN"]
        if closed:
            wins = len([t for t in closed if t.get("pnl", 0) > 0])
            report["win_rate"] = (wins / len(closed)) * 100
            report["total_pnl"] = sum([t.get("pnl", 0) for t in closed])
            report["avg_pnl"] = report["total_pnl"] / len(closed)
        
        return report
    
    def run_live_loop(self, duration_hours=24):
        """Run continuous live trading loop"""
        logger.info(f"🚀 Starting Live Paper Trading for {duration_hours}h...")
        start_time = time.time()
        max_duration = duration_hours * 3600
        
        # Train initial models
        for symbol in self.symbols:
            self.train_models(symbol)
            time.sleep(0.5)  # Rate limiting
        
        iteration = 0
        while time.time() - start_time < max_duration:
            iteration += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"ITERATION #{iteration} - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            logger.info(f"{'='*70}")
            
            # Generate signals
            all_signals = []
            for symbol in self.symbols:
                signal = self.generate_signal(symbol)
                if signal:
                    all_signals.append(signal)
                    
                    # Execute paper trade
                    trade = self.execute_paper_trade(signal)
                
                time.sleep(0.5)  # Rate limiting
            
            # Check existing trades
            closed = self.check_trade_status()
            
            # Generate and save report
            report = self.generate_report()
            report["signals"] = all_signals
            report["closed_trades_this_check"] = closed
            
            report_path = os.path.join(OUTPUT_DIR, "live_trading_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"\n📊 Report saved to {report_path}")
            logger.info(f"📈 Active trades: {len(self.active_trades)} | Closed: {len([t for t in self.trade_history if t['status'] != 'OPEN'])}")
            
            # Wait for next check
            logger.info(f"⏰ Waiting {self.check_interval}s until next signal check...")
            time.sleep(self.check_interval)
        
        logger.info("✅ Live trading session ended")
        return report


if __name__ == "__main__":
    # Start paper trading for 24 hours, checking every 1 hour
    bot = LivePaperTradingBot(
        symbols=["ETH/USDT", "SOL/USDT", "BNB/USDT"],  # Start with 3 symbols
        timeframe="1h",
        check_interval=3600  # Check every 1 hour (change to 300 for 5 min)
    )
    
    # Run for 24 hours (set to shorter for testing)
    final_report = bot.run_live_loop(duration_hours=24)
    
    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)
    print(json.dumps(final_report, indent=2, default=str))
