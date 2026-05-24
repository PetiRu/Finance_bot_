# -*- coding: utf-8 -*-
"""
Core event-driven simulation backtesting engine.
"""

import numpy as np
import pandas as pd


class DynamicPartialExitBacktester:
    def __init__(self, initial_capital=10000.0, fee_rate=0.001, slippage=0.0005, risk_pc=0.01):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.risk_pc = risk_pc

    def run_backtest(self, df, predictions, expected_returns, target_col="target_dir", enable_trailing=True, enable_breakeven=True):
        capital = self.initial_capital
        equity_curve = [capital]
        timestamps = [df.index[0]]
        open_trades, closed_trades = [], []
        
        prices_close, prices_high, prices_low = df["close"].values, df["high"].values, df["low"].values
        atr_values = df["atr_14"].values
        dt_values = df["datetime"].values if "datetime" in df.columns else df.index.values
        
        for i in range(1, len(df)):
            current_close, current_high, current_low, current_atr, current_dt = prices_close[i], prices_high[i], prices_low[i], atr_values[i], dt_values[i]
            remaining_trades = []
            
            for trade in open_trades:
                side, entry, sl, tp_stages, allocations, stage, size_units = trade["side"], trade["entry_price"], trade["stop_loss"], trade["take_profits"], trade["allocations"], trade["tp_stage"], trade["units"]
                pnl_change, hit_stop = 0.0, False
                
                if side == "LONG":
                    if current_low <= sl:
                        pnl_change += (size_units * allocations["remaining"] * (sl * (1.0 - self.slippage) - entry)) - (size_units * allocations["remaining"] * sl * self.fee_rate)
                        hit_stop = True
                    else:
                        if stage == 0 and current_high >= tp_stages[0]:
                            pnl_change += (size_units * allocations["TP1"] * (tp_stages[0] * (1.0 - self.slippage) - entry)) - (size_units * allocations["TP1"] * tp_stages[0] * self.fee_rate)
                            trade["tp_stage"] = 1
                            trade["allocations"]["remaining"] -= allocations["TP1"]
                            if enable_breakeven: sl = entry; trade["stop_loss"] = sl
                        if trade["tp_stage"] == 1 and current_high >= tp_stages[1]:
                            pnl_change += (size_units * allocations["TP2"] * (tp_stages[1] * (1.0 - self.slippage) - entry)) - (size_units * allocations["TP2"] * tp_stages[1] * self.fee_rate)
                            trade["tp_stage"] = 2
                            trade["allocations"]["remaining"] -= allocations["TP2"]
                            if enable_trailing: sl = entry + 0.5*(tp_stages[0] - entry); trade["stop_loss"] = sl
                        if trade["tp_stage"] == 2 and current_high >= tp_stages[2]:
                            pnl_change += (size_units * trade["allocations"]["remaining"] * (tp_stages[2] * (1.0 - self.slippage) - entry)) - (size_units * trade["allocations"]["remaining"] * tp_stages[2] * self.fee_rate)
                            trade["allocations"]["remaining"] = 0.0
                            hit_stop = True
                elif side == "SHORT":
                    if current_high >= sl:
                        pnl_change += (size_units * allocations["remaining"] * (entry - sl * (1.0 + self.slippage))) - (size_units * allocations["remaining"] * sl * self.fee_rate)
                        hit_stop = True
                    else:
                        if stage == 0 and current_low <= tp_stages[0]:
                            pnl_change += (size_units * allocations["TP1"] * (entry - tp_stages[0] * (1.0 + self.slippage))) - (size_units * allocations["TP1"] * tp_stages[0] * self.fee_rate)
                            trade["tp_stage"] = 1
                            trade["allocations"]["remaining"] -= allocations["TP1"]
                            if enable_breakeven: sl = entry; trade["stop_loss"] = sl
                        if trade["tp_stage"] == 1 and current_low <= tp_stages[1]:
                            pnl_change += (size_units * allocations["TP2"] * (entry - tp_stages[1] * (1.0 + self.slippage))) - (size_units * allocations["TP2"] * tp_stages[1] * self.fee_rate)
                            trade["tp_stage"] = 2
                            trade["allocations"]["remaining"] -= allocations["TP2"]
                            if enable_trailing: sl = entry - 0.5*(entry - tp_stages[0]); trade["stop_loss"] = sl
                        if trade["tp_stage"] == 2 and current_low <= tp_stages[2]:
                            pnl_change += (size_units * trade["allocations"]["remaining"] * (entry - tp_stages[2] * (1.0 + self.slippage))) - (size_units * trade["allocations"]["remaining"] * tp_stages[2] * self.fee_rate)
                            trade["allocations"]["remaining"] = 0.0
                            hit_stop = True
                
                capital += pnl_change
                trade["realized_pnl"] += pnl_change
                if hit_stop:
                    trade["exit_time"] = current_dt
                    closed_trades.append(trade)
                else: remaining_trades.append(trade)
                
            open_trades = remaining_trades
            pred_t, exp_t, atr_t, close_t = predictions[i-1], expected_returns[i-1], atr_values[i-1], prices_close[i-1]
            
            if len(open_trades) < 3 and atr_t > 0:
                is_long, is_short = (pred_t >= 0.55 and exp_t > 0.005), (pred_t <= 0.45 and exp_t < -0.005)
                if is_long or is_short:
                    side = "LONG" if is_long else "SHORT"
                    entry_p = close_t * (1.0 + self.slippage if side == "LONG" else 1.0 - self.slippage)
                    sl_dist = 2.0 * atr_t
                    sl_price = entry_p - sl_dist if side == "LONG" else entry_p + sl_dist
                    tp1 = entry_p + 1.2*sl_dist if side == "LONG" else entry_p - 1.2*sl_dist
                    tp2 = entry_p + 2.5*sl_dist if side == "LONG" else entry_p - 2.5*sl_dist
                    tp3 = entry_p + 4.0*sl_dist if side == "LONG" else entry_p - 4.0*sl_dist
                    
                    price_risk = abs(entry_p - sl_price) / entry_p
                    if price_risk > 0.001:
                        risk_usd = capital * self.risk_pc
                        pos_usd = min(risk_usd / price_risk, capital * 3.0)
                        units = pos_usd / entry_p
                        capital -= pos_usd * self.fee_rate
                        open_trades.append({
                            "side": side, "entry_time": current_dt, "entry_price": entry_p, "stop_loss": sl_price,
                            "take_profits": [tp1, tp2, tp3], "allocations": {"TP1": 0.3, "TP2": 0.3, "TP3": 0.4, "remaining": 1.0},
                            "tp_stage": 0, "units": units, "risk_usd": risk_usd, "realized_pnl": -pos_usd * self.fee_rate
                        })
            equity_curve.append(capital)
            timestamps.append(current_dt)
            
        for t in open_trades:
            pnl = (prices_close[-1] - t["entry_price"]) * t["units"] * t["allocations"]["remaining"] if t["side"] == "LONG" else (t["entry_price"] - prices_close[-1]) * t["units"] * t["allocations"]["remaining"]
            t["realized_pnl"] += pnl - (t["units"] * prices_close[-1] * t["allocations"]["remaining"] * self.fee_rate)
            closed_trades.append(t)
            
        return pd.DataFrame({"capital": equity_curve}, index=timestamps), closed_trades, self.calculate_panel_metrics(closed_trades, equity_curve)

    def calculate_panel_metrics(self, trades, eq_curve):
        if not trades: return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0, "avg_r_multiple": 0.0, "trades_count": 0}
        pnls = np.array([t["realized_pnl"] for t in trades])
        wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
        total_p = sum(wins)
        total_l = abs(sum(losses))
        win_rate = len(wins) / len(trades)
        pf = total_p / total_l if total_l > 0 else 99.9
        r_mults = [t["realized_pnl"] / t["risk_usd"] for t in trades if t["risk_usd"] > 0]
        avg_r = np.mean(r_mults) if r_mults else 0.0
        
        tot_ret = (eq_curve[-1] / eq_curve[0]) - 1.0
        chgs = pd.Series(eq_curve).pct_change().dropna()
        sharpe = (chgs.mean() / chgs.std() * np.sqrt(8760)) if len(chgs) > 2 and chgs.std() > 0 else 0.0
        
        peaks = pd.Series(eq_curve).cummax()
        max_dd = float(((pd.Series(eq_curve) - peaks) / peaks).min() * 100.0)
        
        return {"total_return": round(tot_ret * 100.0, 2), "sharpe": round(sharpe, 2), "max_drawdown": round(max_dd, 2), "win_rate": round(win_rate * 100.0, 2), "profit_factor": round(pf, 2), "expectancy": round(np.mean(pnls), 2), "avg_r_multiple": round(avg_r, 2), "trades_count": len(trades)}
