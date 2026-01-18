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

class GlobalFeature(BaseFeature):
    """
    [NEW] 시장 전체 데이터를 필요로 하는 피처 (예: 섹터, 랭킹)
    PlatinumProcessor가 이 클래스를 상속받은 피처를 발견하면 
    Phase 1에서 compute_global을 실행하여 전역 데이터를 미리 계산합니다.
    """
    @abstractmethod
    def compute_global(self, prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
        """
        Input: 
            prices (Date x Ticker Matrix)
            volumes (Date x Ticker Matrix)
        Output: 
            DataFrame (Date x Ticker Matrix) - 계산된 Feature 값
        """
        pass