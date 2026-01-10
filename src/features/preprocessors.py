import pandas as pd
import numpy as np
from .base import BaseFeature

class DollarBarStationaryFeature(BaseFeature):
    """
    [ML Preprocessor] Final Version
    1. Data Sanitization: 0.0 및 음수 가격 제거 (Log 연산 오류 방지)
    2. Smart Dollar Bar: 종목별 거래대금 규모에 맞춰 Bar 개수 자동 확보
    3. Adaptive FracDiff: 데이터 길이에 따라 유동적인 분별 차분 적용
    4. Daily Alignment: Dollar Time -> Daily Time 복원
    5. Burn-in Trimming: 초기 연산 불가능 구간(NaN) 자동 절삭
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Config 파라미터 로드
        config_threshold = self.params.get('threshold', 500_000)
        d_val = self.params.get('d', 0.4)
        
        # [Step 0] Data Sanitization
        price_cols = ['Open', 'High', 'Low', 'Close']
        cols_to_clean = [c for c in price_cols if c in df.columns]
        
        if cols_to_clean:
            df[cols_to_clean] = df[cols_to_clean].where(df[cols_to_clean] > 0, np.nan)
            df = df.ffill().dropna()

        if df.empty or len(df) < 20: 
            return pd.DataFrame(index=df.index)

        # [Step 1] Smart Dollar Bar Generation
        dollar_df = self._to_smart_dollar_bar(df, config_threshold, min_bars=1000)
        
        if dollar_df.empty or len(dollar_df) < 50: 
            return pd.DataFrame(index=df.index)

        # [Step 2] Adaptive FracDiff
        fd_results = []
        for col in cols_to_clean:
            if col in dollar_df.columns:
                series = np.log(dollar_df[col])
                fd_series = self._adaptive_frac_diff(series, d_val)
                fd_series.name = f"FD_{col}"
                fd_results.append(fd_series)
        
        if not fd_results:
            return pd.DataFrame(index=df.index)

        dollar_fd_df = pd.concat(fd_results, axis=1)

        # [Step 3] Alignment
        aligned_df = dollar_fd_df.reindex(df.index, method='ffill')
        
        # [Step 4] Burn-in Trimming
        if not aligned_df.empty:
            first_valid = aligned_df.first_valid_index()
            if first_valid is not None:
                aligned_df = aligned_df.loc[first_valid:]
            else:
                return pd.DataFrame(index=df.index)

        return aligned_df

    def _to_smart_dollar_bar(self, df, config_threshold, min_bars=1000):
        temp_df = df.copy()
        if 'Date' not in temp_df.columns:
            temp_df = temp_df.reset_index()
            
        if 'Trd_Amt' in temp_df.columns:
            dv = temp_df['Trd_Amt']
        else:
            avg = (temp_df['Open'] + temp_df['High'] + temp_df['Low'] + temp_df['Close']) / 4
            dv = avg * temp_df['Volume']
            
        total_dv = dv.sum()
        if total_dv <= 0: return pd.DataFrame()
        
        max_allowed_threshold = total_dv / min_bars
        final_threshold = min(config_threshold, max_allowed_threshold)
        final_threshold = max(final_threshold, 1000.0)

        cum_dv = dv.cumsum()
        groups = (cum_dv // final_threshold).astype(int)
        
        agg_dict = {
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
            'Volume': 'sum'
        }
        if 'Date' in temp_df.columns:
            agg_dict['Date'] = 'last'

        dollar_df = temp_df.groupby(groups).agg(agg_dict)
        
        if 'Date' in dollar_df.columns:
            dollar_df.set_index('Date', inplace=True)
            
        return dollar_df

    def _adaptive_frac_diff(self, series, d):
        thresholds = [1e-5, 1e-4, 1e-3, 1e-2]
        for thres in thresholds:
            w = self._get_weights_ffd(d, thres, len(series))
            width = len(w) - 1
            if width < len(series):
                return self._frac_diff_ffd(series, d, thres, w)
        return series.diff()

    def _get_weights_ffd(self, d, thres, lim):
        w, k = [1.], 1
        while True:
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < thres or len(w) > lim:
                break
            w.append(w_k)
            k += 1
        return np.array(w[::-1]).reshape(-1, 1)

    def _frac_diff_ffd(self, series, d, thres, w=None):
        if w is None:
            w = self._get_weights_ffd(d, thres, len(series))
        width = len(w) - 1
        series_val = series.values
        w_val = w.flatten()
        res = np.full(len(series), np.nan)
        for i in range(width, len(series)):
            window_data = series_val[i-width : i+1] 
            res[i] = np.dot(window_data, w_val)
        return pd.Series(res, index=series.index)