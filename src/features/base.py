from abc import ABC, abstractmethod
import pandas as pd

class BaseFeature(ABC):
    """모든 피처 클래스의 부모 클래스"""
    
    def __init__(self, **kwargs):
        """config에서 넘겨준 파라미터를 받음 (예: window=20)"""
        self.params = kwargs

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input: OHLCV가 포함된 DataFrame
        Output: 피처 컬럼이 추가된 DataFrame (또는 피처 Series)
        """
        pass