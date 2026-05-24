# -*- coding: utf-8 -*-
"""
Classifiers and Regressors model ensemble suite.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

try: import xgboost as xgb
except ImportError: xgb = None

try: import lightgbm as lgb
except ImportError: lgb = None

logger = logging.getLogger("ModelSuite")


class CryptoMLEnsembleSuite:
    def __init__(self, config_xgb=None, config_lgb=None, config_rf=None, model_dir="saved_models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.xgb_params = config_xgb or {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05, "n_jobs": -1}
        self.lgb_params = config_lgb or {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05, "n_jobs": -1, "verbose": -1}
        self.rf_params = config_rf or {"n_estimators": 100, "max_depth": 6, "n_jobs": -1}
        self.classifiers = {}
        self.regressors = {}
        self.trained_features = []

    def prepare_xy(self, df, target_col="target_dir"):
        meta_cols = [
            "timestamp", "datetime", "open", "high", "low", "close", "volume",
            "target_dir", "target_ret", "target_sl", "target_tp1", "target_tp2", "target_tp3"
        ]
        feature_cols = [col for col in df.columns if col not in meta_cols]
        self.trained_features = feature_cols
        X = df[feature_cols].copy().ffill().bfill().fillna(0.0)
        y = df[target_col].copy()
        return X, y

    def train_classifiers(self, df, target_col="target_dir"):
        X, y = self.prepare_xy(df, target_col=target_col)
        clfs = []
        rf_clf = RandomForestClassifier(**self.rf_params)
        
        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, val_idx in tscv.split(X):
            rf_clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            
        rf_clf.fit(X, y)
        clfs.append(("rf", rf_clf))

        if xgb:
            xgb_clf = xgb.XGBClassifier(**self.xgb_params)
            xgb_clf.fit(X, y)
            clfs.append(("xgb", xgb_clf))
        if lgb:
            lgb_params_clean = self.lgb_params.copy()
            if "metric" in lgb_params_clean: lgb_params_clean.pop("metric")
            lgb_clf = lgb.LGBMClassifier(**lgb_params_clean)
            lgb_clf.fit(X, y)
            clfs.append(("lgb", lgb_clf))
            
        self.classifiers[target_col] = clfs
        return clfs

    def train_regressors(self, df, target_col="target_ret"):
        X, y = self.prepare_xy(df, target_col=target_col)
        regs = []
        rf_reg = RandomForestRegressor(**self.rf_params)
        rf_reg.fit(X, y)
        regs.append(("rf", rf_reg))

        if xgb:
            xgb_params_reg = self.xgb_params.copy()
            if "objective" in xgb_params_reg: xgb_params_reg["objective"] = "reg:squarederror"
            if "eval_metric" in xgb_params_reg: xgb_params_reg.pop("eval_metric")
            xgb_reg = xgb.XGBRegressor(**xgb_params_reg)
            xgb_reg.fit(X, y)
            regs.append(("xgb", xgb_reg))
        if lgb:
            lgb_params_reg = self.lgb_params.copy()
            if "objective" in lgb_params_reg: lgb_params_reg["objective"] = "regression"
            if "metric" in lgb_params_reg: lgb_params_reg.pop("metric")
            lgb_reg = lgb.LGBMRegressor(**lgb_params_reg)
            lgb_reg.fit(X, y)
            regs.append(("lgb", lgb_reg))
            
        self.regressors[target_col] = regs
        return regs

    def predict_direction(self, df, target_col="target_dir"):
        X, _ = self.prepare_xy(df, target_col=target_col)
        probs = [clf.predict_proba(X)[:, 1] for name, clf in self.classifiers[target_col]]
        return np.mean(probs, axis=0)

    def predict_return(self, df, target_col="target_ret"):
        X, _ = self.prepare_xy(df, target_col=target_col)
        preds = [reg.predict(X) for name, reg in self.regressors[target_col]]
        return np.mean(preds, axis=0)

    def save_suite(self, symbol, timeframe):
        symbol_clean = symbol.replace("/", "_")
        path = os.path.join(self.model_dir, f"ensemble_{symbol_clean}_{timeframe}.pkl")
        with open(path, "wb") as f:
            pickle.dump({"classifiers": self.classifiers, "regressors": self.regressors, "features": self.trained_features}, f)
