# -*- coding: utf-8 -*-
"""
Helper utilities for asset correlation filters and Telegram alerts.
"""

import logging
import pandas as pd


def filter_correlated_assets(df_collection, max_correlation=0.85):
    returns_dict = {}
    for symbol, df in df_collection.items():
        if "close" in df.columns:
            returns_dict[symbol] = df["close"].pct_change().fillna(0.0)
            
    if not returns_dict: return list(df_collection.keys())
    corr_df = pd.DataFrame(returns_dict).corr().abs()
    
    to_prune = set()
    all_syms = list(corr_df.columns)
    
    for i in range(len(all_syms)):
        for j in range(i + 1, len(all_syms)):
            s1, s2 = all_syms[i], all_syms[j]
            if corr_df.loc[s1, s2] > max_correlation:
                to_prune.add(s2)
                
    return [s for s in all_syms if s not in to_prune]
