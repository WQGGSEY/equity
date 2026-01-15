import pandas as pd
import numpy as np
from .base import BaseFeature

class MovingAverage(BaseFeature):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        window = self.params.get('window', 20)
        # 컬럼명에 파라미터 포함 (예: MA_20)
        col_name = f"MA_{window}"
        return df['Close'].rolling(window=window).mean().rename(col_name)

class FD_MovingAverage(BaseFeature):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        window = self.params.get('window', 20)
        col_name = f"FD_MA_{window}"
        return df['FD_Close'].rolling(window=window).mean().rename(col_name)

class Return(BaseFeature):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        window = self.params.get('window', 1)
        return df['Close'].pct_change(periods=window).rename(f"Return_{window}")

class FD_Return(BaseFeature):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        window = self.params.get('window', 1)
        return df['FD_Close'].pct_change(periods=window).rename(f"FD_Return_{window}")

class TrdAmount(BaseFeature):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        return ((df['Close'] + df['High'] + df['Low'] + df['Open']) / 4 * df['Volume']).rename("TrdAmount")

class FD_TrdAmount(BaseFeature):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        return ((df['FD_Close'] + df['FD_High'] + df['FD_Low'] + df['FD_Open']) / 4 * df['Volume']).rename("FD_TrdAmount")