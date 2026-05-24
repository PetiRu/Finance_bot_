# -*- coding: utf-8 -*-
"""
Signals calculator using Triple Barrier boundaries.
"""

import json
import numpy as np
import pandas as pd


def apply_triple_barrier_labeling(df, lookahead=24, sl_atr_mult=2.0, tp1_atr_mult=1.5):
    length = len(df)
    target_dir = np.zeros(length)
    target_ret = np.zeros(length)
    close, high, low, atr = df["close"].values, df["high"].values, df["low"].values, df["atr_14"].values
    
    for i in range(length - lookahead):
        c_price, c_atr = close[i], atr[i]
        if c_atr <= 0: c_atr = c_price * 0.01
        upper = c_price + (tp1_atr_mult * c_atr)
        lower = c_price - (sl_atr_mult * c_atr)
        
        l_hit, s_hit, f_idx = False, False, i + lookahead
        for j in range(i + 1, i + lookahead + 1):
            if high[j] >= upper: {l_hit := True, f_idx := j}; break
            if low[j] <= lower: {s_hit := True, f_idx := j}; break
                
        target_dir[i] = 1.0 if l_hit else 0.0 if s_hit else (1.0 if close[f_idx] > c_price else 0.0)
        target_ret[i] = (close[f_idx] / c_price) - 1.0
        
    df["target_dir"] = target_dir
    df["target_ret"] = target_ret
    return df


class SignalGenerator:
    def __init__(self, ensemble_suite):
        self.suite = ensemble_suite

    def generate_current_signal(self, df_with_features, symbol, timeframe):
        if df_with_features.empty: return None
        last_row = df_with_features.iloc[-1:]
        close, atr = float(last_row["close"].values[0]), float(last_row["atr_14"].values[0])
        proba = float(self.suite.predict_direction(last_row)[0])
        exp_ret = float(self.suite.predict_return(last_row)[0])
        
        direction = "LONG" if proba >= 0.52 else "SHORT" if proba <= 0.48 else "NEUTRAL"
        confidence = proba if direction == "LONG" else (1.0 - proba) if direction == "SHORT" else 0.50
        
        sw_hi = close * (1.0 + float(last_row["swing_high"].values[0]))
        sw_lo = close * (1.0 + float(last_row["swing_low"].values[0]))
        
        if direction == "LONG":
            sl = max(close - (2.0 * atr), sw_lo - (0.001 * close))
            size = close - sl
            tps = [close + (1.2 * size), close + (2.5 * size), close + (4.0 * size)]
        elif direction == "SHORT":
            sl = min(close + (2.0 * atr), sw_hi + (0.001 * close))
            size = sl - close
            tps = [close - (1.2 * size), close - (2.5 * size), close - (4.0 * size)]
        else:
            sl = close - (2.1 * atr)
            tps = [close + (1.5 * atr), close + (3.0 * atr), close + (5.0 * atr)]
            
        rr = abs(tps[1] - close) / (abs(close - sl) + 1e-10)
        
        return {
            "symbol": symbol.replace("/", ""),
            "timeframe": timeframe,
            "direction": direction,
            "entry": round(close, 2),
            "stop_loss": round(sl, 2),
            "tp_levels": [round(t, 2) for t in tps],
            "confidence": round(confidence, 4),
            "expected_return": round(exp_ret, 5),
            "risk_reward": round(rr, 2)
        }
