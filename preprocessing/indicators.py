import pandas as pd
import numpy as np
import numba

def log_return(price_df: pd.DataFrame) -> pd.Series:
    return np.log(price_df['close'] / price_df['close'].shift(1))

def compute_sma(price_df: pd.DataFrame, window: int) -> pd.Series:
    return price_df['close'].rolling(window=window).mean()

def compute_ema(price_df: pd.DataFrame, span: int) -> pd.Series:
    return price_df['close'].ewm(span=span, adjust=False).mean()

def compute_gma(price_df: pd.DataFrame, window: int) -> pd.Series:
    # Gaussian Moving Average (GMA) 계산
    weights = np.exp(-0.5 * (np.linspace(-2, 2, window) ** 2))
    weights /= weights.sum()
    return price_df['close'].rolling(window=window).apply(lambda x: np.dot(x, weights), raw=True)

def compute_rsi(price_df: pd.DataFrame, period: int) -> pd.Series:
    delta = price_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_obv(price_df: pd.DataFrame) -> pd.Series:
    """
    Numpy/Pandas 벡터화 연산을 사용하여 OBV를 효율적으로 계산합니다.
    """
    
    price_change_sign = np.sign(price_df['close'].diff())

    signed_volume = price_df['volume'] * price_change_sign
    
    obv = signed_volume.fillna(0).cumsum()
    
    return obv

def compute_mfi(price_df: pd.DataFrame, period: int) -> pd.Series:
    typical_price = (price_df['high'] + price_df['low'] + price_df['close']) / 3
    money_flow = typical_price * price_df['volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
    
    money_flow_ratio = positive_flow / negative_flow
    mfi = 100 - (100 / (1 + money_flow_ratio))
    return mfi


@numba.jit(nopython=True) 
def _get_bayesian_slope(prices_window: np.ndarray) -> float:
    N = len(prices_window)
    if N < 2:
        return np.nan

    X = np.empty((N, 2))
    for i in range(N):
        X[i, 0] = 1.0     # 절편 (Intercept)
        X[i, 1] = float(i) # 시간 (Time: 0, 1, 2...)
    y = prices_window

    try:
        beta_ols = np.linalg.pinv(X.T @ X) @ X.T @ y
        residuals = y - X @ beta_ols
        sigma_sq = np.var(residuals)
        if sigma_sq < 1e-6:
            sigma_sq = 1e-6
            
        mu_n = beta_ols 
        return mu_n[1]
    except Exception:
        return np.nan

# --- 헬퍼 함수 2: 기울기 분산만 계산 ---
@numba.jit(nopython=True) # Numba 데코레이터 유지
def _get_bayesian_variance(prices_window: np.ndarray) -> float:
    N = len(prices_window)
    if N < 2:
        return np.nan

    # --- [수정] np.column_stack 대신 for 루프로 X 행렬 생성 ---
    X = np.empty((N, 2))
    for i in range(N):
        X[i, 0] = 1.0
        X[i, 1] = float(i)
    y = prices_window
    # --- [수정 완료] ---

    try:
        beta_ols = np.linalg.pinv(X.T @ X) @ X.T @ y
        residuals = y - X @ beta_ols
        sigma_sq = np.var(residuals)
        if sigma_sq < 1e-6:
            sigma_sq = 1e-6
            
        inv_lambda_n = (1.0 / sigma_sq) * (X.T @ X)
        lambda_n = np.linalg.inv(inv_lambda_n) 
        return lambda_n[1, 1]
    except Exception:
        return np.nan

# --- 메인 함수 ---
def compute_bayesian_trend(price_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    롤링 윈도우에서 베이지안 선형 회귀를 계산합니다.
    'b_slope' (추세 방향)와 'b_uncert' (추세 불확실성)을 반환합니다.
    
    Args:
        price_df (pd.DataFrame): 'close' 컬럼을 포함한 원본 DataFrame
        window (int): 롤링 윈도우 크기

    Returns:
        pd.DataFrame: 'b_slope_{window}'와 'b_uncert_{window}' 컬럼이 포함된 DataFrame
    """
    
    # .apply()가 단일 스칼라만 반환해야 하므로, 두 번 분리하여 호출합니다.
    
    # 1. 기울기(Slope Mean) 계산
    slope_mean = price_df['close'].rolling(window=window).apply(
        _get_bayesian_slope,
        raw=True  # Pandas Series 대신 NumPy 배열을 전달하여 속도 향상
    )
    # 2. 기울기 분산(Slope Variance) 계산
    slope_variance = price_df['close'].rolling(window=window).apply(
        _get_bayesian_variance,
        raw=True
    )
    
    # 3. 결과 DataFrame으로 결합
    df = pd.DataFrame(index=price_df.index)
    
    # 지표 1: 추세의 방향 (모멘텀)
    df[f'b_slope_{window}'] = slope_mean
    
    # 지표 2: 추세의 불확실성 (표준편차)
    # 분산(variance) 대신 표준편차(std)가 더 해석하기 좋습니다.
    df[f'b_uncert_{window}'] = np.sqrt(slope_variance)
    
    return df

def compute_atr(price_df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range (ATR) 계산"""
    high_low = price_df['high'] - price_df['low']
    high_close_prev = np.abs(price_df['high'] - price_df['close'].shift(1))
    low_close_prev = np.abs(price_df['low'] - price_df['close'].shift(1))
    
    # True Range 계산
    true_range = pd.DataFrame({
        'hl': high_low,
        'hc': high_close_prev,
        'lc': low_close_prev
    }).max(axis=1)
    
    # ATR (EMA of True Range)
    return true_range.ewm(span=period, adjust=False).mean()

def compute_bb_width(price_df: pd.DataFrame, window: int, num_std: int = 2) -> pd.Series:
    """Bollinger Band Width (BBW) 계산"""
    sma = price_df['close'].rolling(window=window).mean()
    std = price_df['close'].rolling(window=window).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    # (Upper - Lower) / Middle
    bb_width = (upper_band - lower_band) / sma
    return bb_width

def compute_skewness(log_return_series: pd.Series, window: int) -> pd.Series:
    """로그 수익률의 롤링 왜도 계산"""
    return log_return_series.rolling(window=window).skew()

def compute_kurtosis(log_return_series: pd.Series, window: int) -> pd.Series:
    """로그 수익률의 롤링 첨도 계산"""
    return log_return_series.rolling(window=window).kurt()

def compute_rolling_autocorr(log_return_series: pd.Series, window: int, lag: int = 1) -> pd.Series:
    """로그 수익률의 롤링 자기상관(lag-1) 계산"""
    # .corr()는 두 개의 Series 간의 롤링 상관관계를 벡터화하여 계산
    return log_return_series.rolling(window=window).corr(log_return_series.shift(lag))

def compute_amihud_indicator(pricde_df: pd.DataFrame, window: int) -> pd.Series:
    """Amihud Illiquidity Ratio 계산"""
    abs_log_return = log_return(pricde_df).abs()
    illiquidity = (abs_log_return / (pricde_df['volume']*pricde_df['close'])).replace([np.inf, -np.inf], np.nan)
    return illiquidity.rolling(window=window).mean()
