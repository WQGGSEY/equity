# src/alpha/ops.py
import pandas as pd
import numpy as np
import scipy
from collections import deque
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def safe_pinv(A: np.ndarray|pd.DataFrame) -> np.ndarray:
    """
    pinv 연산에서 'SVD did not converge' 에러를 방지하기 위해,
    (1) 아주 작은 ridge를 추가
    (2) 실패 시 더 큰 ridge 추가
    (3) 그래도 안 되면 0행렬 반환
    """
    eps = 1e-12
    # 정사각 행렬이면 약간의 ridge 추가
    if A.ndim == 2 and A.shape[0] == A.shape[1]:
        A_reg = A + np.eye(A.shape[0]) * eps
    else:
        A_reg = A.copy()

    try:
        # 첫 시도
        return scipy.linalg.pinv(A_reg)  # rcond 제거
    except np.linalg.LinAlgError:
        # print("[safe_pinv] 1st attempt failed, try bigger ridge")
        bigger_eps = 1e-8
        if A.ndim == 2 and A.shape[0] == A.shape[1]:
            A_reg2 = A + np.eye(A.shape[0]) * bigger_eps
        else:
            A_reg2 = A.copy()

        try:
            return scipy.linalg.pinv(A_reg2)  # rcond 제거
        except np.linalg.LinAlgError:
            # print("[safe_pinv] 2nd attempt also failed → returning zeros")
            return np.zeros_like(A)

def add(X:pd.DataFrame, Y: pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param Y: columns=Timeseries, index=Tickers Dataframe
    :return X+Y
    """
    return X+Y

def subtract(X: pd.DataFrame, Y: pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param Y: columns=Timeseries, index=Tickers Dataframe
    :return: X-Y
    """
    return X-Y

def multiply(X: pd.DataFrame, Y: pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param Y: columns=Timeseries, index=Tickers Dataframe
    :return: X*Y
    """
    return X*Y

def divide(X: pd.DataFrame, Y: pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param Y: columns=Timeseries, index=Tickers Dataframe
    :return: X/Y
    """
    return X/Y

def ts_return(X: pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    """
    return (X - ts_delay(X, d)) / ts_delay(X, d)


def df_abs(X: pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :return: Absolute value of X
    """
    return X.abs()

def equal(X:pd.DataFrame, Y:pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param Y: columns=Timeseries, index=Tickers Dataframe
    :return: 1 if X==Y, else 0
    """
    return (X==Y).astype(int)

def exp(X:pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :return: Exponential value of X
    """
    return X.apply(np.exp)


def df_max(X:pd.DataFrame, Y:pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param Y: columns=Timeseries, index=Tickers Dataframe
    :return: bigger value between X and Y
    """
    return pd.DataFrame(np.maximum(X, Y), index=X.index, columns=X.columns)


def df_min(X:pd.DataFrame, Y:pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param Y: columns=Timeseries, index=Tickers Dataframe
    :return: smaller value between X and Y
    """
    return pd.DataFrame(np.minimum(X, Y), index=X.index, columns=X.columns)


def purify(X:pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :return: Replace +inf, -inf value to NaN
    """
    return X.replace(np.inf, np.nan).replace(-np.inf, np.nan)


def replace(X: pd.DataFrame, replacement: dict):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param replacement: dictionary, the form of {Target: Dest.}
    :return: replace Target to Dest.
    """
    for target in replacement:
        dest = replacement[target]
        X = X.replace(target, dest)
    return X


def df_and(X:pd.DataFrame, Y:pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param Y: columns=Timeseries, index=Tickers Dataframe
    :return: Logical AND Operator
    """
    return pd.DataFrame(X.values & Y.values, index=X.index, columns=X.columns)


def df_or(X:pd.DataFrame, Y:pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param Y: columns=Timeseries, index=Tickers Dataframe
    :return: Logical OR Operator
    """
    return pd.DataFrame(X.values | Y.values, index=X.index, columns=X.columns)


def if_else(logic: pd.DataFrame, X: pd.DataFrame | int | float, Y: pd.DataFrame | int | float):
    """
    [Robust if_else]
    - logic이 NaN인 경우 False로 취급합니다. (데이터 부족 시 진입 방지)
    """
    # 1. logic이 DataFrame인 경우 NaN 채우기 (False 취급)
    if isinstance(logic, pd.DataFrame):
        logic = logic.fillna(False)
    
    # 2. np.where 사용
    # broadcasting을 위해 values 사용 가능여부 확인
    l_val = logic.values if isinstance(logic, pd.DataFrame) else logic
    x_val = X.values if isinstance(X, pd.DataFrame) else X
    y_val = Y.values if isinstance(Y, pd.DataFrame) else Y
    
    # DataFrame 복원 (인덱스/컬럼 보존)
    # logic의 구조를 따름
    if isinstance(logic, pd.DataFrame):
        return pd.DataFrame(np.where(l_val, x_val, y_val), index=logic.index, columns=logic.columns)
    else:
        # logic이 스칼라인 경우 (드물지만)
        return np.where(l_val, x_val, y_val)


def ts_weighted_decay(X:pd.DataFrame, k=0.5):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param k: weighting factor
    :return: k * today's value + (1-k) * yesterday's value
    """
    return k*X + (1-k)*ts_delay(X, 1)


def ts_av_diff(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: X - ts_mean(X, d)
    """
    return X - ts_mean(X, d)


def ts_mean(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: simple average of X days
    """
    return X.rolling(window=d, axis=0).mean()


def ts_co_kurtosis(X:pd.DataFrame, Y:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param Y: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: Co_kurtosis of X & Y
    """
    dev_x = ts_av_diff(X, d)
    dev_y = ts_av_diff(Y, d)
    std_x = ts_std_dev(X, d)
    std_y = ts_std_dev(Y, d)
    return (dev_x*(dev_y**3)).rolling(window=d, axis=0).mean()/(std_x*(std_y**3))


def ts_covariance(X: pd.DataFrame, Y: pd.DataFrame, d: int) -> pd.DataFrame:
    """
    메모리 효율적인 공분산 계산 (E[XY] - E[X]E[Y] 공식 사용)
    O(N*T*d) 메모리 -> O(N*T) 메모리로 감소
    """
    # 1. E[X], E[Y] 계산 (단순 롤링 평균)
    mean_x = ts_mean(X, d)
    mean_y = ts_mean(Y, d)
    
    # 2. E[XY] 계산
    mean_xy = ts_mean(X * Y, d)
    
    # 3. Cov = E[XY] - E[X]E[Y] (보정 계수 n/(n-1) 적용)
    cov = (mean_xy - mean_x * mean_y) * (d / (d - 1))
    return cov

def ts_variance(X: pd.DataFrame, d: int) -> pd.DataFrame:
    """메모리 효율적인 분산 계산"""
    mean_x = ts_mean(X, d)
    mean_x2 = ts_mean(X * X, d)
    var = (mean_x2 - mean_x * mean_x) * (d / (d - 1))
    return var

def ts_regression(Y: pd.DataFrame, X: pd.DataFrame, d: int, rettype: int = 0) -> pd.DataFrame:
    """
    Rolling Regression (Y ~ a + b*X)
    rettype: 0=residual, 1=slope(b), 2=intercept(a), 3=fitted_value
    """
    # 1. Slope (Beta) = Cov(X, Y) / Var(X)
    cov_xy = ts_covariance(X, Y, d)
    var_x = ts_variance(X, d)
    
    beta = cov_xy / (var_x + 1e-9) # 0 나누기 방지
    
    # 2. Intercept (Alpha) = E[Y] - Beta * E[X]
    mean_x = ts_mean(X, d)
    mean_y = ts_mean(Y, d)
    alpha = mean_y - beta * mean_x
    
    if rettype == 1: # Slope
        return beta
    elif rettype == 2: # Intercept
        return alpha
    elif rettype == 3: # Fitted Value (Prediction)
        return alpha + beta * X
    else: # Residual (Error) -> Y - (Alpha + Beta * X)
        # 잔차(Residual)는 평균 회귀 전략에서 가장 많이 쓰임
        return Y - (alpha + beta * X)

def ts_corr(X: pd.DataFrame, Y: pd.DataFrame, d: int) -> pd.DataFrame:
    """상관계수 (Correlation)"""
    cov_xy = ts_covariance(X, Y, d)
    std_x = ts_std_dev(X, d)
    std_y = ts_std_dev(Y, d)
    return cov_xy / (std_x * std_y + 1e-9)


def ts_ema(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: exponential moving average of X
    """
    return X.ewm(span=d, axis=0).mean()


def ts_delay(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: value before d days
    """
    return X.shift(d, axis=0).fillna(0)


def ts_delta(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: X - ts_delay(X, d)
    """
    return X - ts_delay(X, d)


def ts_ir(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: Information Ratio of X in d days, ts_mean(X, d) / ts_std_dev(X, d)
    """
    return ts_mean(X, d)/ts_std_dev(X, d)


def ts_max(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: max value in d days
    """
    return X.rolling(window=d, axis=0).max()


def ts_min(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: minimum value in d days
    """
    return X.rolling(window=d, axis=0).min()

def ts_argmax(X: pd.DataFrame, d: int) -> pd.DataFrame:
    """
    [Fixed] 윈도우 내 최대값이 위치한 '거리(Lag)'를 반환
    예: 오늘이 최대면 0, 어제가 최대면 1 ...
    """
    # rolling apply는 느리지만 argmax는 numpy 최적화가 어려움
    # (d - 1) - argmax (argmax는 윈도우 시작 기준 인덱스이므로)
    return X.rolling(d).apply(lambda x: (d - 1) - np.argmax(x), raw=True)

def ts_argmin(X: pd.DataFrame, d: int) -> pd.DataFrame:
    """
    [Fixed] 윈도우 내 최소값이 위치한 '거리(Lag)'를 반환
    """
    return X.rolling(d).apply(lambda x: (d - 1) - np.argmin(x), raw=True)


def ts_max_diff(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: X - ts_max(X, d)
    """
    return X.fillna(0) - ts_max(X, d)


def ts_min_diff(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: X - ts_min(X, d)
    """
    return X.fillna(0) - ts_min(X, d)


def ts_median(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: median value in d days
    """
    return X.rolling(window=d, axis=0).median()


def ts_std_dev(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: standard deviation of X in d days
    """
    return X.rolling(window=d, axis=0).std()

def ts_linear_decay(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: linearly decayed value of X in d days
    """
    return X.rolling(window=d, win_type='triang', axis=0).mean()


def ts_min_max_cps(X:pd.DataFrame, d:int, f=2):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :param f: subtraction factor
    :return: ts_min(X, d) + ts_max(X, d) - f*X
    """
    return ts_min(X, d) + ts_max(X, d) - f*X


def ts_min_max_diff(X:pd.DataFrame, d:int, f=0.5):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :param f: subtraction factor
    :return: X - f*(ts_min(X, d) + ts_max(X, d)
    """
    return X - f*(ts_min(X, d) + ts_max(X, d))

def ts_vector_neut(X: pd.DataFrame, Y: pd.DataFrame, d: int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param Y: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    """
    return X - ts_vector_proj(X, Y, d)

def ts_vector_proj(X: pd.DataFrame, Y: pd.DataFrame, d: int) -> pd.DataFrame:
    """
    [Optimized & Safe]
    X와 Y의 시계열(axis=0) 롤링 벡터 투영을 계산합니다.
    """
    # 1. 내적 계산 (롤링 합) - axis=0 자동 적용
    dot_xy = (X * Y).rolling(window=d, min_periods=max(1, d // 2)).sum()
    # dot_yy = (Y * Y).rolling(window=d, min_periods=max(1, d // 2)).sum()
    dot_xx = (X * X).rolling(window=d, min_periods=max(1, d // 2)).sum()
    
    # 2. 비율 계산 (0 나누기 방지)
    ratio = dot_xy / dot_xx.replace(0, np.nan)
    
    # 3. 투영 벡터의 현재 시점 값 반환
    return ratio * X


def ts_backfill(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: backfilled days
    """
    return X.fillna(method='ffill', axis=0, limit=d)

def ts_poly_regression(Y: pd.DataFrame, X: pd.DataFrame, d: int, k: int = 2) -> pd.DataFrame:
    """
    [Memory Optimized] Rolling Polynomial Regression
    Model: Y = Beta_0 + Beta_1 * X + ... + Beta_k * X^k
    
    * Sliding Window 대신 'Normal Equation(최소자승법)'의 합산 공식을 사용합니다.
    * 메모리 효율을 위해 종목(Column)별로 순회하며 계산합니다.
    * axis=0 (Time-series) 방향으로 올바르게 작동합니다.
    
    Returns:
        Residuals (Y - Fitted) at the last point of the window.
    """
    # 1. 결과 담을 그릇 (모두 NaN으로 초기화)
    residuals = pd.DataFrame(np.nan, index=Y.index, columns=Y.columns)
    
    # 2. X의 거듭제곱 미리 계산 (Powers) -> [1, X, X^2, ... X^2k]
    #    Normal Equation을 풀기 위해 필요한 모든 항을 준비합니다.
    #    메모리 절약을 위해 필요한 순간에 계산할 수도 있지만, 속도를 위해 미리 계산합니다.
    #    X가 너무 크면 overflow 위험이 있으니 주의 (보통 Returns나 Z-score 사용 권장)
    X_powers = [X.pow(i) for i in range(2 * k + 1)]
    
    # 3. Y와 X거듭제곱의 곱 미리 계산 -> [Y, Y*X, Y*X^2, ... Y*X^k]
    YX_powers = [Y * X.pow(i) for i in range(k + 1)]
    
    # 4. 종목별 순회 (Vectorized over Time, Looped over Tickers)
    #    3000개 종목을 한번에 3차원 배열로 만들면 메모리가 터지므로, 하나씩 처리합니다.
    common_cols = Y.columns.intersection(X.columns)
    
    for col in common_cols:
        try:
            # (1) 해당 종목의 데이터 추출 (Series)
            #     d일 롤링 합(Sum)을 구합니다. 이것이 곧 Matrix A와 Vector b의 성분이 됩니다.
            #     min_periods를 설정하여 데이터가 조금 부족해도 계산되게 함.
            S = [xp[col].rolling(window=d, min_periods=d//2).sum().values for xp in X_powers]
            V = [yxp[col].rolling(window=d, min_periods=d//2).sum().values for yxp in YX_powers]
            
            # (2) 각 시점(t)별 Linear System 구성 (A * beta = b)
            #     A[t] = [[Sum(1), Sum(X), ...], [Sum(X), Sum(X^2), ...], ...]
            #     b[t] = [Sum(Y), Sum(YX), ...]
            
            T_len = len(Y)
            
            # A행렬 구성 (T x k+1 x k+1)
            # 예: k=2이면 3x3 행렬. 
            # Row 0: S[0], S[1], S[2]
            # Row 1: S[1], S[2], S[3]
            # Row 2: S[2], S[3], S[4]
            A = np.zeros((T_len, k+1, k+1))
            for i in range(k + 1):
                for j in range(k + 1):
                    A[:, i, j] = S[i + j]
            
            # b벡터 구성 (T x k+1)
            b = np.zeros((T_len, k + 1))
            for i in range(k + 1):
                b[:, i] = V[i]
            
            # Ridge 추가 (Singular Matrix 방지)
            epsilon = 1e-6
            indices = np.arange(k + 1)
            A[:, indices, indices] += epsilon
            
            # (3) 행렬 풀이 (Vectorized Solver)
            #     np.linalg.solve는 (N, M, M) 형태의 input을 받아 (N, M) 해를 반환합니다.
            #     유효하지 않은 데이터(NaN)가 있으면 에러나므로, NaN 마스크 처리 필요하지만
            #     속도를 위해 try-except나 np.nan_to_num 활용
            
            # NaN이 있는 행은 계산 불가 -> 0으로 채우고 나중에 다시 NaN 처리
            valid_mask = ~np.isnan(A).any(axis=(1, 2)) & ~np.isnan(b).any(axis=1)
            
            betas = np.zeros((T_len, k + 1))
            
            if np.any(valid_mask):
                betas[valid_mask] = np.linalg.solve(A[valid_mask], b[valid_mask])
            
            # (4) 잔차 계산 (Residuals)
            #     Res = Y - (Beta0 + Beta1*X + Beta2*X^2 + ...)
            #     현재 시점의 X값(X[col])을 사용
            x_vals = X[col].values
            y_vals = Y[col].values
            y_pred = np.zeros(T_len)
            
            for i in range(k + 1):
                y_pred += betas[:, i] * (x_vals ** i)
                
            res = y_vals - y_pred
            
            # 유효하지 않았던 구간은 다시 NaN 처리
            res[~valid_mask] = np.nan
            
            residuals[col] = res
            
        except Exception:
            # Singular matrix 등 에러 발생 시 해당 종목은 Pass
            continue
            
    return residuals

def ts_product(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: timeseries production
    """
    return X.rolling(window=d, axis=0).agg(lambda x: x.prod())


def ts_returns(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: ts_delta(X, d)/X
    """
    return (ts_delta(X, d))/ts_delay(X, d)


def ts_scale(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: (X – ts_min(X, d)) / (ts_max(X, d) – ts_min(X, d))
    """
    min_value = ts_min(X, d)
    max_value = ts_max(X, d)
    return (X-min_value) / (max_value - min_value)


def ts_skewness(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: skewness of X
    """
    mean_x = ts_mean(X, d)
    std = ts_std_dev(X, d)
    return (((X-mean_x) / std)**3).rolling(window=d, axis=0).mean()


def ts_kurtosis(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: kurtosis of X
    """
    mean_x = ts_mean(X, d)
    std = ts_std_dev(X, d)
    return (((X-mean_x) / std)**4).rolling(window=d, axis=0).mean()


def ts_sum(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: sum of X in d days
    """
    X = X.fillna(0)
    return X.rolling(window=d, axis=0).sum()


def ts_rank(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: scaled ascending rank value from 0 to 1 in d days.
    """
    return X.rolling(window=d, axis=0).rank(pct=True)

def ts_zscore(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: z-score of X in d days
    """
    return (X-ts_mean(X, d))/ts_std_dev(X, d)


def vector_neut(X:pd.DataFrame, Y:pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param Y: columns=Timeseries, index=Tickers Dataframe
    :return: X - vector_proj(X, Y)
    """
    return X - vector_proj(X, Y)


def vector_proj(X:pd.DataFrame, Y:pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param Y: columns=Timeseries, index=Tickers Dataframe
    :return: project X to Y
    """
    # calculate [X1^TY1 X2^%Y2 ... XN^TYN]
    dot_xy = (X * Y).sum(axis=1)  # X dot Y
    dot_yy = (Y * Y).sum(axis=1)  # Y dot Y

    # 혹시 분모가 0인 날짜가 있을 수도 있으니 처리
    ratio = dot_xy / dot_yy
    ratio = ratio.replace([np.inf, -np.inf], 0).fillna(0)

    # ratio를 각 컬럼에 브로드캐스팅
    proj = Y.multiply(ratio, axis=0)
    return proj


def regression_neut(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """
    [Cross-sectional Regression Neutralization]
    매일매일 Y = alpha + beta * X 회귀분석을 수행하고 잔차(Residual)를 반환
    Residual = Y - (alpha + beta * X)
    """
    # 1. Beta = Cov(X, Y) / Var(X)  (Cross-sectional)
    #    Cov(X, Y) = E[XY] - E[X]E[Y]
    #    여기서 E[]는 횡단면 평균(mean(axis=1))
    
    mean_x = X.mean(axis=1)
    mean_y = Y.mean(axis=1)
    mean_xy = (X * Y).mean(axis=1)
    mean_x2 = (X * X).mean(axis=1)
    
    cov_xy = mean_xy - mean_x * mean_y
    var_x = mean_x2 - mean_x * mean_x
    
    beta = cov_xy / var_x.replace(0, np.nan)
    
    # 2. Alpha = Mean(Y) - Beta * Mean(X)
    alpha = mean_y - beta * mean_x
    
    # 3. Residual = Y - (Alpha + Beta * X)
    #    Broadcasting 주의: alpha, beta는 Series(Date)
    term1 = X.mul(beta, axis=0) # Beta * X
    term2 = term1.add(alpha, axis=0) # Alpha + Beta * X
    
    return Y.sub(term2).fillna(0.0)


def scale_down(X: pd.DataFrame) -> pd.DataFrame:
    """
    [Cross-sectional Min-Max Scale]
    (Value - DailyMin) / (DailyMax - DailyMin) -> 0~1 범위로 변환
    """
    daily_min = X.min(axis=1)
    daily_max = X.max(axis=1)
    
    denom = daily_max - daily_min
    
    # 분모 0 방지
    denom = denom.replace(0, np.nan)
    
    return X.sub(daily_min, axis=0).div(denom, axis=0).fillna(0.0)

def arc_sin(X:pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :return: arc_sin(X)
    """
    return X.apply(np.arcsin)


def arc_cos(X:pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :return: arc_cos(X)
    """
    return X.apply(np.arccos)


def arc_tan(X:pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :return: arc_tan(X)
    """
    return X.apply(np.arctan)


def sigmoid(X:pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :return: sigmoid(X)
    """
    exp = np.exp(X)
    return exp/(1+exp)


def tanh(X:pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :return: tanh(X)
    """
    return X.apply(np.tanh)


def rank(X:pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :return: scaled ascending rank value from 0 to 1 in the universe
    """
    rank = X.rank(axis=1, method='min', ascending=True)
    return scale_down(rank)


def zscore(X: pd.DataFrame) -> pd.DataFrame:
    """
    [Cross-sectional Z-Score]
    (Value - DailyMean) / DailyStd
    """
    # 1. 일별 평균과 표준편차 계산 (Series 형태: index=Date)
    daily_mean = X.mean(axis=1)
    daily_std = X.std(axis=1)
    
    # 2. 0 나누기 방지
    daily_std = daily_std.replace(0, np.nan)
    
    # 3. Broadcasting 연산 (axis=0: 날짜 인덱스끼리 매칭)
    return X.sub(daily_mean, axis=0).div(daily_std, axis=0).fillna(0.0)

# ==============================================================================
# [Fixed Group Ops]
# - 기존: Time-series Grouping (한 종목의 과거 데이터끼리 그룹화) -> WRONG ❌
# - 수정: Cross-sectional Grouping (같은 날짜의 여러 종목끼리 그룹화) -> CORRECT ✅
# ==============================================================================

def _group_operate(X: pd.DataFrame, group: pd.DataFrame, func: str) -> pd.DataFrame:
    """
    [Core Engine]
    모든 그룹 연산의 기반이 되는 함수입니다.
    Data(Date x Ticker)를 Stack하여 (Date, Ticker)로 만든 뒤,
    (Date, Group) 기준으로 Groupby 연산을 수행합니다.
    """
    # 1. MultiIndex Series로 변환 ((Date, Ticker) 형태)
    X_stacked = X.stack(dropna=False)
    g_stacked = group.stack(dropna=False)
    
    # 2. DataFrame 생성
    df = pd.DataFrame({'val': X_stacked, 'grp': g_stacked})
    
    # 3. 그룹 연산 (Level 0 = Date, grp = Group)
    #    "매 날짜마다(level=0), 각 그룹별로(grp)" func 적용
    transformed = df.groupby([df.index.get_level_values(0), 'grp'])['val'].transform(func)
    
    # 4. 원래 모양(Date x Ticker)으로 복구
    return transformed.unstack()

def group_mean(X: pd.DataFrame, group: pd.DataFrame):
    return _group_operate(X, group, 'mean')

def group_sum(X: pd.DataFrame, group: pd.DataFrame):
    return _group_operate(X, group, 'sum')

def group_max(X: pd.DataFrame, group: pd.DataFrame):
    return _group_operate(X, group, 'max')

def group_min(X: pd.DataFrame, group: pd.DataFrame):
    return _group_operate(X, group, 'min')

def group_std_dev(X: pd.DataFrame, group: pd.DataFrame):
    return _group_operate(X, group, 'std')

def group_zscore(X: pd.DataFrame, group: pd.DataFrame):
    """
    (Value - GroupMean) / GroupStd
    """
    mean = group_mean(X, group)
    std = group_std_dev(X, group)
    return (X - mean) / std.replace(0, np.nan)

def group_neutralize(X: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    """
    [Sector Neutralization]
    그룹별 평균을 0으로 만듭니다. (X - GroupMean)
    이후 날짜별 총합(Gross Exposure)이 1이 되도록 스케일링합니다.
    """
    # 1. 중립화
    X_neut = X - group_mean(X, group)
    
    # 2. 스케일링 (L1 Norm)
    # axis=1 (Cross-sectional Sum)
    abs_sum = X_neut.abs().sum(axis=1)
    
    # 0 나누기 방지
    return X_neut.div(abs_sum.replace(0, np.nan), axis=0).fillna(0.0)

def group_rank(X: pd.DataFrame, group: pd.DataFrame):
    """
    같은 날짜, 같은 그룹 내에서의 순위 (0.0 ~ 1.0)
    """
    X_stacked = X.stack(dropna=False)
    g_stacked = group.stack(dropna=False)
    df = pd.DataFrame({'val': X_stacked, 'grp': g_stacked})
    
    # Date(Level 0)와 Group 기준으로 랭킹 계산
    ranks = df.groupby([df.index.get_level_values(0), 'grp'])['val'].rank(pct=True)
    
    return ranks.unstack()

def basket(X: pd.DataFrame, tile: list[float]):
    """
    [Cross-sectional Basket]
    매일매일 종목들을 순위별로 줄 세워서 그룹(Basket)을 할당합니다.
    예: tile=[0.5] -> 상위 50%는 1, 하위 50%는 0
    """
    # 1. 날짜별(axis=1) 랭킹 계산 (0~1)
    #    method='first'로 동점자 처리
    ranks = X.rank(axis=1, pct=True, method='first')
    
    # 2. 구간 할당
    #    bins: [0, 0.3, 0.7, 1.0] -> labels: [0, 1, 2]
    #    right=True: ( ] 구간 (기본값)
    bins = [-1.0] + tile + [2.0] # 넉넉하게 잡음
    labels = list(range(len(bins) - 1))
    
    # pandas cut은 1차원만 지원하므로 stack 후 처리
    stacked_ranks = ranks.stack(dropna=False)
    baskets = pd.cut(stacked_ranks, bins=bins, labels=labels)
    
    # 복구 및 숫자형 변환
    return baskets.unstack().astype('float32')

# --- Complex Group Ops (Neutralize with Constraints 등) ---

def group_mean_masked(X: pd.DataFrame, group: pd.DataFrame, constraints: pd.DataFrame) -> pd.DataFrame:
    """
    constraints가 True인 종목들만 사용하여 그룹 평균을 구함.
    결과는 모든 종목 위치에 broadcast됨.
    """
    # 마스크 적용 (False인 곳은 NaN 처리하여 평균 계산에서 제외)
    X_masked = X.where(constraints.astype(bool), np.nan)
    return group_mean(X_masked, group)

def group_neutralize_with_constraints(X: pd.DataFrame, group: pd.DataFrame, constraints: pd.DataFrame) -> pd.DataFrame:
    """
    1. Constraints=True인 종목들만 대상으로 그룹 평균 계산
    2. 중립화 (X - Mean)
    3. Constraints=False인 종목은 0으로 만듦
    4. 전체 Gross=1 스케일링
    """
    mask = constraints.astype(bool)
    
    # 1. 마스킹된 그룹 평균
    gmean = group_mean_masked(X, group, mask)
    
    # 2. 중립화
    X_neut = X - gmean
    
    # 3. 마스크 밖 제거
    X_neut = X_neut.where(mask, 0.0)
    
    # 4. 스케일링
    abs_sum = X_neut.abs().sum(axis=1)
    return X_neut.div(abs_sum.replace(0, np.nan), axis=0).fillna(0.0)

def group_vector_proj(X: pd.DataFrame, Y: pd.DataFrame, group: pd.DataFrame):
    """
    그룹별 벡터 투영: Proj_Y(X) = (X.Y / Y.Y) * Y
    (같은 날짜, 같은 그룹 내에서 계산)
    """
    # X.Y와 Y.Y를 그룹별로 합산 (Group Sum)
    dot_xy = group_sum(X * Y, group)
    dot_yy = group_sum(Y * Y, group)
    
    ratio = dot_xy / dot_yy.replace(0, np.nan)
    return ratio * Y

def group_vector_neut(X: pd.DataFrame, Y: pd.DataFrame, group: pd.DataFrame):
    return X - group_vector_proj(X, Y, group)

def group_scale(X: pd.DataFrame, group: pd.DataFrame):
    """
    그룹별 Min-Max Scaling (0~1)
    """
    g_min = group_min(X, group)
    g_max = group_max(X, group)
    
    return (X - g_min) / (g_max - g_min).replace(0, np.nan)

def group_cartesian_product(group1: pd.DataFrame, group2: pd.DataFrame):
    """
    두 그룹의 조합을 새로운 그룹으로 생성 (예: Sector + Size -> "Tech_Large")
    문자열 결합 연산이므로 element-wise로 처리
    """
    # NaN 처리
    g1 = group1.fillna('N/A').astype(str)
    g2 = group2.fillna('N/A').astype(str)
    
    return g1 + "_" + g2

def signed_power(X: pd.DataFrame, power: int = 2):
    sign_x = np.sign(X)
    abs_x = np.abs(X)
    powered = abs_x ** power
    
    # 부호를 다시 곱해줌
    return sign_x * powered

def trade_when(in_trigger: pd.DataFrame,
               X: pd.DataFrame,
               exit_trigger: pd.DataFrame) -> pd.DataFrame:
    """
    Sequential logic:
      1) if exit_trigger[i, j] > 0, positions[i, j] = 0
      2) else if in_trigger[i, j] > 0, positions[i, j] = X[i, j]
      3) else positions[i, j] = positions[i-1, j]
    """
    # DataFrame -> Numpy array로 변환(연산속도 향상)
    arr_in  = in_trigger.values
    arr_out = exit_trigger.values
    arr_X   = X.values
    
    # 결과 저장용 배열(초기값 전부 0)
    positions_arr = np.zeros_like(arr_X)
    nrows = len(X)

    # 1) 첫 번째 행(0번째) 초기화
    #    exit_trigger > 0 -> 0
    #    in_trigger > 0   -> X
    #    else             -> 0
    positions_arr[0] = np.where(
        arr_out[0] > 0, 
        0, 
        np.where(arr_in[0] > 0, arr_X[0], 0)
    )

    # 2) 두 번째 행(1번째)부터 순차적으로
    for i in range(1, nrows):
        # exit_trigger가 1 이상이면 0
        # 아니면 in_trigger가 1 이상이면 X
        # 아니면 바로 직전 i-1 행의 positions_arr
        positions_arr[i] = np.where(
            (arr_out[i] > 0) & (arr_in[i] == 0),
            0,
            np.where(
                arr_in[i] > 0,
                arr_X[i],
                positions_arr[i-1]
            )
        )

    # Numpy array -> DataFrame
    positions = pd.DataFrame(positions_arr, 
                             index=X.index, 
                             columns=X.columns)
    return positions

def corr_vec(returns_df: pd.DataFrame, d: int, inverse=False):
    window = d
    # 롤링 윈도우 길이 (126일)
    T = returns_df.shape[1]                   # 전체 날짜 수
    num_strategies = returns_df.shape[0]      # 전략의 수

    # =============================
    # 2. 결과 배열 초기화
    # =============================
    # 날짜 길이를 보존하기 위해 최종 결과 shape는 (T, num_strategies, num_strategies)
    # 초반 window-1일은 데이터가 부족하므로 NaN으로 채움
    result_array = np.full((T, num_strategies, num_strategies), np.nan)

    # =============================================
    # 3. 각 날짜(시점)별로 최근 126일 데이터를 사용하여 계산
    # =============================================
    # t: 0 ~ T-1 에 대해, t가 window-1 보다 작으면 계산 불가, 그렇지 않으면 최근 window일 간 데이터를 사용
    for t in range(window - 1, T):
        # 최근 126일(즉, t-window+1 부터 t까지)의 데이터 선택  
        # df의 열이 날짜이므로, iloc를 이용하여 열 슬라이싱 진행
        window_df = returns_df.iloc[:, t - window + 1 : t + 1]
        
        # 행이 전략이므로, transpose 후 corr()로 전략 간 상관관계 계산  
        # 상관관계 행렬의 shape는 (num_strategies x num_strategies)
        corr_matrix = window_df.T.corr()
        if inverse:
            corr_matrix = corr_matrix.fillna(0)
            corr_matrix = safe_pinv(corr_matrix)
        
        # 날짜 t에 해당하는 결과 저장
        result_array[t] = corr_matrix
    return result_array

def corr_dot_one(returns_df: pd.DataFrame, d: int, inverse=False):
    X_3d = corr_vec(returns_df, d, inverse)
    result_list = []
    ones = np.ones((X_3d.shape[2], 1))
    for i in range(X_3d.shape[0]):
        result_list.append((X_3d[i] @ ones).flatten())
    return pd.DataFrame(np.array(result_list).T, index=returns_df.index, columns=returns_df.columns)


def hump(X: pd.DataFrame, alpha=0.5, threshold=0.002) -> pd.DataFrame:
    """
    partial + threshold 방식으로 target_raw를 점진적으로 따라가도록 하는 함수.
    
    NaN 처리:
      - w_old, w_target 각각 NaN => 0 으로 간주 (해당 종목 비중이 없다고 봄)
      - diff 계산에서도 NaN 방지
    """
    X = neutralize_and_scale(X)

    final_port = pd.DataFrame(
        0.0,
        index=X.index,
        columns=X.columns
    )

    # 날짜(열) 정렬(혹은 기존 순서)
    date_cols = X.columns.sort_values()
    if len(date_cols) == 0:
        return final_port

    # 첫 날은 목표치 그대로 (NaN->0)
    first_col = date_cols[0]
    # fillna(0) => NaN은 0으로
    final_port[first_col] = X[first_col].fillna(0)

    # 날짜 순회
    for i in range(1, len(date_cols)):
        col_prev = date_cols[i-1]
        col_now  = date_cols[i]

        # 직전 final_port
        w_old = final_port[col_prev].fillna(0)
        # 이번 날짜의 목표 비중
        w_target = X[col_now].fillna(0)

        # diff
        diff = w_target - w_old
        # 임계값 초과인 곳만 조정
        mask = diff.abs() > threshold

        # 이번 날짜 초기값: 직전 값 복사
        w_new = w_old.copy()

        # partial 매매
        w_new[mask] = w_old[mask] + alpha * diff[mask]

        # final_port에 저장
        final_port[col_now] = w_new

    return final_port

def hump_ts_rank(
    X: pd.DataFrame,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
    window: int = 60
) -> pd.DataFrame:
    """
    과거 window 동안의 'weight 변화량' 절대값을 종목별로 모아,
    오늘 변화량이 그 중 어느 정도 percentile인지 ts_rank로 계산.
    => 그 랭킹(0~1)에 따라 alpha를 alpha_min~alpha_max 사이로 선형 스케일링해서 적용.
    
    Params
    ------
    X        : [tickers x dates], 목표 비중
    alpha_min: 변화가 가장 작은 구간에서 적용할 최소 alpha
    alpha_max: 변화가 가장 큰 구간에서 적용할 최대 alpha
    window   : ts_rank를 계산할 때 쓰는 rolling window 크기
    eps      : 0 나누기 방지, rank 계싼 시 미세값
    
    Returns
    -------
    final_port : [tickers x dates], hump 조정이 적용된 최종 포트 비중
    """
    # 결과 보관용
    X = neutralize_and_scale(X)
    final_port = pd.DataFrame(0.0, index=X.index, columns=X.columns)

    date_cols = X.columns
    if len(date_cols) == 0:
        return final_port

    # 첫 날짜는 목표 비중 그대로
    first_col = date_cols[0]
    final_port[first_col] = X[first_col].fillna(0)

    # (ticker별) 과거 변화량 기록 자료구조
    # 각 ticker마다 deque나 list를 사용
    #   diff_history[ticker] = deque([float, float, ...], maxlen=window)
    # => 직전 window일치의 abs_diff 저장
    diff_history = {
        tkr: deque(maxlen=window) for tkr in X.index
    }

    # 첫 날은 "직전 포트"가 없으므로 변화량 기록 없음
    # 따라서 두 번째 날짜부터 계산
    for i in range(1, len(date_cols)):
        col_prev = date_cols[i - 1]
        col_now  = date_cols[i]

        w_old    = final_port[col_prev].fillna(0)
        w_target = X[col_now].fillna(0)

        diff     = w_target - w_old
        diff_abs = diff.abs()

        # ========== (1) 종목별 ts_rank 계산 ==========
        # 각 종목별로, 오늘의 diff_abs가 "과거 window 동안 diff_history[ticker]"에서 몇 번째로 큰지
        # rank = (등수 - 1) / (개수 - 1)
        # *개수=0이면 rank=0 (혹은 NaN) 처리
        alpha_for_today = pd.Series(index=X.index, dtype=float)

        for tkr in X.index:
            old_history = diff_history[tkr]  # deque

            # 오늘 변화량
            this_diff = diff_abs[tkr]

            # history + [this_diff]를 합쳐서 rank를 구한다면,
            # or   history만 보고 percentile을 구하고, 그보다 큰지 작은지?
            # 여기서는 "history에 대해 몇 번째 위치인지"를 직접 rank로 계산
            if len(old_history) == 0:
                # 과거 데이터가 전혀 없다면 => 랭킹 알 수 없음 => 중간값 쓰거나 alpha_min?
                alpha_for_today[tkr] = alpha_min
            else:
                # numpy array 변환
                arr = np.array(old_history)
                
                # arr 중에서 this_diff가 몇 번째 위치인지
                # (등수를 구하는 한 가지 방법)
                # rank = (arr <= this_diff).sum()  (등수)
                # scaled_rank = (rank-1)/(len(arr)-1)
                # 모두 동일값이면 (등수-1)/(개수-1) = 0,1 혼재할 수 있음
                
                # 좀 더 robust하게 'min' rank를 계산
                rank_raw = (arr < this_diff).sum() + 1  # this_diff보다 작은 것의 개수 + 1
                scaled_rank = (rank_raw - 1) / (len(arr))  # 0~1 범위
                
                # alpha = alpha_min + (alpha_max - alpha_min) * scaled_rank
                alpha_for_today[tkr] = alpha_min + (alpha_max - alpha_min) * scaled_rank
        
        # ========== (2) 포트폴리오 업데이트 ==========
        w_new = w_old + alpha_for_today * diff

        # 최종 비중 저장
        final_port[col_now] = w_new

        # ========== (3) 오늘 diff_abs를 history에 추가 ==========
        for tkr in X.index:
            diff_history[tkr].append(diff_abs[tkr])

    return final_port

def last_days_from_val(X:pd.DataFrame):
    """
    각 행에서 non-NaN 셀이 나오면 0으로 표시하고,
    그 이후 NaN 셀에 대해 순차적으로 1,2,3...을 부여합니다.
    (새로운 non-NaN이 나오면 다시 0으로 리셋)
    
    Parameters:
        df (pd.DataFrame): 행이 티커, 열이 시계열인 데이터프레임.
                           셀 값은 NaN 또는 non-NaN 값임.
                           
    Returns:
        pd.DataFrame: 같은 인덱스/컬럼 구조를 갖는 결과 데이터프레임.
    """
    # 원본 데이터를 numpy array로 변환
    arr = X.values       # shape: (n_rows, n_cols)
    n_rows, n_cols = arr.shape

    # 각 셀이 non-NaN이면 True인 마스크
    mask = ~np.isnan(arr)

    # 각 열의 인덱스를 행마다 확장 (각 행: [0,1,2,..., n_cols-1])
    col_idx = np.arange(n_cols)
    col_idx_matrix = np.broadcast_to(col_idx, (n_rows, n_cols))

    # non-NaN인 위치에서는 해당 열 인덱스, NaN인 위치는 -infinity를 넣어줍니다.
    # 이렇게 하면 누적 최대값(cumulative maximum) 연산을 통해 "마지막 리셋 위치"를 찾을 수 있음.
    resets = np.where(mask, col_idx_matrix, -np.inf)

    # 각 행마다 왼쪽부터 누적 최대값을 구합니다.
    # 즉, 현재 셀 이전(포함)에서 마지막 non-NaN 셀의 열 인덱스가 계산됨.
    last_reset = np.maximum.accumulate(resets, axis=1)

    # 현재 열 인덱스에서 마지막 리셋 인덱스를 빼면, non-NaN 이후 몇 칸 떨어져 있는지 알 수 있음.
    counts = col_idx_matrix - last_reset

    # 아직 리셋(즉, non-NaN)이 한 번도 나타나지 않은 셀은 last_reset이 -infinity이므로
    # 이 경우에는 결과를 NaN으로 남겨둡니다.
    counts[last_reset == -np.inf] = np.nan

    # 원래 셀이 non-NaN인 경우는 항상 0 (리셋 지점)이 되어야 하므로 0으로 강제.
    counts[mask] = 0

    # DataFrame의 인덱스와 컬럼을 복원하여 반환합니다.
    return pd.DataFrame(counts, index=X.index, columns=X.columns)

def ts_profile(X: pd.DataFrame, d: int, bins: int = 10) -> pd.DataFrame:
    """
    (행=티커, 열=시계열) 구조의 DataFrame X에 대해,
    각 날짜 t에서 '직전 d일' (t-d+1 ~ t)에 걸친 모든 티커×d일의 값 중
    min~max를 bins등분하고,
    X[i,t]가 그 중 몇 번째 bin에 해당하는지 정수(0..bins-1)로 할당한다.

    - 만약 window 전체가 NaN이거나, X[i,t]가 NaN이면 -> 결과 NaN
    - 만약 min==max(값이 전부 동일)라면 -> 유효 값들은 bin=0 할당
    - 반환 DF는 X와 동일한 shape이며, 각 셀에는 bin 인덱스(또는 NaN)이 들어있다.

    Parameters
    ----------
    X    : pd.DataFrame
        index=Tickers, columns=Dates
        예) X.loc["005930", "2021-01-03"] = 123.45
    d    : int
        Rolling window 크기(며칠)
    bins : int
        등분할 bin 개수 (기본 10)

    Returns
    -------
    bin_df : pd.DataFrame
        X와 같은 인덱스/컬럼 구조.
        각 셀은 0..(bins-1) 정수 or NaN (pandas의 float NaN)
    """
    # (N,T) = (종목수, 날짜수)
    N, T = X.shape

    # 결과 저장용 (N x T) 배열(초기 NaN)
    bin_array = np.full((N, T), np.nan, dtype=float)

    # 원본을 numpy array로 추출
    x_arr = X.values  # shape=(N, T)
    dates = X.columns

    for t in range(T):
        # 윈도우 구간 [start ~ t]
        start = max(0, t - d + 1)
        sub_block = x_arr[:, start : t+1]  # shape=(N, d')  (d' <= d)

        # 윈도우 전체(티커×d'일)에서 유효한 값(na 제외)
        valid_mask = ~np.isnan(sub_block)
        if not np.any(valid_mask):
            # 이 윈도우가 전부 NaN -> 오늘은 bin 할당 불가
            continue

        # min, max
        block_min = sub_block[valid_mask].min()
        block_max = sub_block[valid_mask].max()

        if block_min == block_max:
            # window 내 유효 값이 모두 동일
            # => 그중 "오늘" 값이 NaN이 아니면 bin=0, 그렇지 않으면 NaN
            #    즉, 오늘 X[i,t]가 NaN이 아니면 bin=0
            today_col = x_arr[:, t]
            today_mask = ~np.isnan(today_col)
            bin_array[today_mask, t] = 0
            continue

        # 일반 케이스
        # 오늘 열(=day t)의 (N,) array
        today_col = x_arr[:, t]
        # 유효 값인 종목만 bin 할당
        valid_today = ~np.isnan(today_col)
        if not np.any(valid_today):
            # 당일 모든 종목이 NaN이라면 skip
            continue

        # 스케일링
        span = block_max - block_min
        # bin index = floor( (x - block_min)/span * (bins - 1) )
        scaled = (today_col[valid_today] - block_min) / span * (bins - 1)
        # clip [0, bins-1]
        scaled = np.clip(scaled, 0, bins - 1)
        # 정수화
        bin_idx = np.floor(scaled)

        # 결과에 기록
        bin_array[valid_today, t] = bin_idx

    # DataFrame으로 변환
    bin_df = pd.DataFrame(bin_array, index=X.index, columns=dates)
    # 필요하다면 "Int64"(nullable int)로 변환 가능
    # bin_df = bin_df.astype("Int64")

    return bin_df

def neutralize_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 열에 대해, nan이 아닌 값들만 대상으로
      1) 평균(= net 포지션)을 빼서 중립화하고, 
      2) 절대값 합이 1이 되도록 스케일링
    
    단, 해당 열에 유효한(즉, nan이 아닌) 셀이 2개 미만이면 
    조정하지 않고 그대로 두거나 0으로 세팅
    """
    df_out = df.copy()
    for col in df.columns:
        valid = df[col].notna()  # nan이 아닌 셀 마스크
        n_valid = valid.sum()
        if n_valid < 2:
            # 유효값이 너무 적으면 (예: 0 또는 1개) 강제로 조정하기 어려움
            # 여기서는 해당 셀들을 0으로 처리합니다.
            df_out.loc[valid, col] = 0.0
            continue

        # nan이 아닌 값들만 선택
        x = df.loc[valid, col]
        
        # 1) 중립화: 평균을 빼서 합을 0으로 만듦
        x_neut = x - x.mean()
        
        # 2) 스케일링: 절대값 합이 1이 되도록 함
        gross = x_neut.abs().sum()
        if gross != 0:
            x_scaled = x_neut / gross
        else:
            # 모든 값이 0이면 그대로 둠 (또는 0으로 세팅)
            x_scaled = x_neut
        df_out.loc[valid, col] = x_scaled

        # 확인: (디버깅용)
        # print(f"{col}: net sum = {df_out.loc[valid, col].sum()}, gross = {df_out.loc[valid, col].abs().sum()}")
        
    return df_out

def prev_value(X: pd.DataFrame):
    X_filled = X.ffill(axis=1)
    
    # (2) 한 칸 오른쪽으로 shift하여 "직전 값"을 위치시키기
    X_prev = X.shift(1, axis=1)
    
    # (3) 발표가 있는 곳(df.notna())에만 df_prev 값을 넣음
    #     나머지는 NaN
    mask = X.notna()
    out_arr = np.where(mask, X_prev, np.nan)
    
    # 결과 DataFrame 생성
    out_df = pd.DataFrame(out_arr, index=X.index, columns=X.columns)
    return out_df



def ts_frac_diff(df: pd.DataFrame, d: float, window: int = 20) -> pd.DataFrame:
    """
    Fractional Differentiation Operator
    :param df: Time-Series DataFrame (Index=Date, Col=Ticker)
    :param d: 차분 계수 (0.0 ~ 1.0 사이, 보통 0.3~0.7 사용)
    :param window: 계산에 사용할 과거 기간 (너무 크면 느려짐, 20~50 권장)
    :return: 분수 차분된 데이터프레임
    """
    # 1. 가중치 미리 계산 (모든 종목에 동일 적용)
    def _get_weights(d: float, size: int) -> np.ndarray:
        """
        분수 차분을 위한 가중치 계산 (이항 전개)
        :param d: 차분 차수 (예: 0.4, 0.6 등)
        :param size: 윈도우 크기
        """
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] * (d - k + 1) / k
            w.append(w_k)
        # 최근 데이터가 가장 뒤에 오도록 뒤집음 (Convolution용)
        return np.array(w[::-1])

    weights = _get_weights(d, window)
    
    # 2. 롤링 윈도우에 가중치 적용 (Dot Product)
    # raw=True를 써야 numpy 연산으로 빠르게 처리됨
    # (메모리 최적화를 위해 fillna(0) 처리 후 수행)
    return df.fillna(method='ffill').rolling(window=window).apply(
        lambda x: np.dot(x, weights), 
        raw=True
    )


def __raise_no_file_error__(contents, others=''):
    raise FileNotFoundError(f'Download Data {contents} -> Update.py {others}')
