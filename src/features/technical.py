import pandas as pd
import numpy as np
from .base import BaseFeature

class MovingAverage(BaseFeature):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        window = self.params.get('window', 20)
        # 컬럼명에 파라미터 포함 (예: MA_20)
        col_name = f"MA_{window}"
        return df['FD_Close'].rolling(window=window).mean().rename(col_name)

class DailyReturn(BaseFeature):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        return df['FD_Close'].pct_change().rename("Daily_Return")

class TradeValue(BaseFeature):
    """사용자님이 정의한 거래대금"""
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # (Open+High+Low+Close)/4 * Volume
        avg_price = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        return (avg_price * df['Volume']).rename("Trd_Amt")