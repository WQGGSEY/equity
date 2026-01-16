# src/alpha/ops.py
import pandas as pd
import numpy as np

# --- 1. 시계열 연산 (Time-Series) ---
def ts_mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window=window).mean()

def ts_std(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window=window).std()

def ts_max(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window=window).max()

def ts_min(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window=window).min()

def ts_rank(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """과거 window 기간 중 현재 값의 순위 (0~1)"""
    return df.rolling(window=window).rank(pct=True)

def ts_delay(df: pd.DataFrame, period: int) -> pd.DataFrame:
    return df.shift(period)

def ts_delta(df: pd.DataFrame, period: int) -> pd.DataFrame:
    return df.diff(period)

# --- 2. 횡단면 연산 (Cross-Sectional) ---
def rank(df: pd.DataFrame) -> pd.DataFrame:
    """매일매일(Row) 전체 종목(Col) 중에서 순위를 매김 (0.0 ~ 1.0)"""
    return df.rank(axis=1, pct=True, method='dense')

def scale(df: pd.DataFrame) -> pd.DataFrame:
    """절대값의 합이 1이 되도록 조정 (Long/Short 비중 맞춤용)"""
    return df.div(df.abs().sum(axis=1), axis=0)

def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """횡단면 Z-Score 정규화"""
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    return df.sub(mean, axis=0).div(std, axis=0)

# --- 3. 논리/산술 연산 ---
def log(df: pd.DataFrame) -> pd.DataFrame:
    return np.log(df)

def sign(df: pd.DataFrame) -> pd.DataFrame:
    return np.sign(df)