import numpy as np
import pandas as pd
from .base import BaseFeature

class PCAPrice(BaseFeature):
    """
    [Rolling PCA Feature]
    OHLC 데이터의 공분산 구조를 분석하여 주성분(Trend)과 잔차(Noise)를 분리합니다.
    Look-ahead Bias를 방지하기 위해 Rolling Window 방식을 사용합니다.
    """
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        params:
            window (int): PCA를 수행할 롤링 윈도우 크기 (기본값: 20)
        returns:
            PCA_Ret_{window}: PC1(주추세)의 Log Return
            PCA_Expl_{window}: PC1이 전체 변동성을 설명하는 비율 (낮으면 시장이 혼조세)
        """
        window = self.params.get('window', 20)
        cols = ['Open', 'High', 'Low', 'Close']
        
        # 1. 데이터 유효성 검사
        if not all(c in df.columns for c in cols):
            return pd.DataFrame()
            
        # 2. Log Return 계산 (가격 레벨 효과 제거 및 정규성 확보)
        # 0 division 방지를 위해 fillna 처리
        returns = np.log(df[cols] / df[cols].shift(1)).fillna(0)
        
        # 3. Rolling Covariance Matrix 계산
        # 결과 shape: (N * 4, 4) -> (Date, Feature) MultiIndex 형태가 됨
        rolling_cov = returns.rolling(window=window).cov()
        
        # 4. Numpy 기반 고속 연산 준비
        dates = df.index
        n_rows = len(df)
        
        # DataFrame -> 3D Array 변환: (N, 4, 4)
        # rolling_cov는 (N*4) rows 이므로 reshape 필요
        try:
            cov_values = rolling_cov.values.reshape(n_rows, 4, 4)
        except ValueError:
            # 데이터 개수가 맞지 않는 경우 (초기 윈도우 등으로 인한 shape mismatch 방어)
            return pd.DataFrame(index=df.index)

        ret_values = returns.values # (N, 4)
        
        # 결과 저장소
        pc1_ret_arr = np.full(n_rows, np.nan)
        explained_var_arr = np.full(n_rows, np.nan)
        
        # 5. Iterative Eigen Decomposition
        # 벡터화가 어렵기 때문에 윈도우 이후부터 루프 수행 (Numpy 연산이라 빠름)
        for i in range(window, n_rows):
            cov = cov_values[i]
            
            # 결측치(NaN)가 있는 윈도우는 스킵
            if np.isnan(cov).any():
                continue
                
            # 고유값 분해 (eigh는 대칭행렬용으로 더 빠르고 안정적)
            # vals: 고유값 (오름차순), vecs: 고유벡터
            vals, vecs = np.linalg.eigh(cov)
            
            # 내림차순 정렬 (큰 고유값이 PC1)
            idx = np.argsort(vals)[::-1]
            vals = vals[idx]
            vecs = vecs[:, idx]
            
            # PC1 벡터 (가장 강력한 추세 방향)
            pc1_vec = vecs[:, 0]
            
            # [중요] 부호 모호성(Sign Ambiguity) 해결
            # PC1이 'Close' 가격과 같은 방향을 가리키도록 강제 조정
            # Close 컬럼은 3번째 인덱스 (Open, High, Low, Close 순서일 때)
            # cols 리스트 순서에 의존하므로 cols.index('Close') 사용
            close_idx = 3 
            if pc1_vec[close_idx] < 0:
                pc1_vec = -pc1_vec
            
            # 현재 시점의 수익률을 PC1 벡터에 사영(Projection)
            # 이것이 "노이즈가 제거된 추세 수익률"이 됨
            current_ret_vector = ret_values[i]
            pc1_ret = np.dot(current_ret_vector, pc1_vec)
            
            # 설명력 비율 (Explained Variance Ratio)
            total_var = np.sum(vals)
            ratio = vals[0] / total_var if total_var > 0 else 0.0
            
            pc1_ret_arr[i] = pc1_ret
            explained_var_arr[i] = ratio
            
        # 6. 결과 DataFrame 생성
        result = pd.DataFrame(index=dates)
        result[f'PCA_Ret_{window}'] = pc1_ret_arr
        result[f'PCA_Expl_{window}'] = explained_var_arr
        
        return result