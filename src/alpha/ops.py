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


def if_else(logic:pd.DataFrame, X:pd.DataFrame|int|float, Y:pd.DataFrame|int|float):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param Y: columns=Timeseries, index=Tickers Dataframe
    :return: if logic is true, return X, otherwise return Y
    """
    return pd.DataFrame(np.where(logic, X, Y), index = logic.index, columns=logic.columns)


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
    return X.rolling(window=d, axis=1).mean()


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
    return (dev_x*(dev_y**3)).rolling(window=d, axis=1).mean()/(std_x*(std_y**3))


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
    return X.ewm(span=d, axis=1).mean()


def ts_delay(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: value before d days
    """
    return X.shift(d, axis=1).fillna(0)


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
    return X.rolling(window=d, axis=1).max()


def ts_min(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: minimum value in d days
    """
    return X.rolling(window=d, axis=1).min()


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
    return X.rolling(window=d, axis=1).median()


def ts_std_dev(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: standard deviation of X in d days
    """
    return X.rolling(window=d, axis=1).std()

def ts_linear_decay(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: linearly decayed value of X in d days
    """
    return X.rolling(window=d, win_type='triang', axis=1).mean()


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
    Rolling(창길이 = d) 기반의 '벡터 투영' 함수를 날짜별로 계산하여 반환.
    
    각 (티커 i, 날짜 t)에 대해:
      1) 최근 d일치 (t-d+1 ~ t)의 X[i], Y[i]를 각각 추출
      2) dot_xy = Σ(X[i,τ]*Y[i,τ]) ,  dot_yy = Σ(Y[i,τ]^2)
      3) ratio = dot_xy / dot_yy  (단, dot_yy=0이면 ratio=0)
      4) 최종 ts_vector_proj(i,t) = ratio * Y[i, t]
    
    반환 DataFrame은 X와 동일한 인덱스/컬럼 크기를 가지며,
    첫 (d-1)일은 계산 불가하므로 NaN이 됩니다.
    
    Parameters
    ----------
    X : pd.DataFrame
        shape = [tickers x dates]
    Y : pd.DataFrame
        shape = [tickers x dates]
    d : int
        Rolling 윈도우 길이 (일 수)
    
    Returns
    -------
    proj_df : pd.DataFrame
        shape = [tickers x dates], 첫 (d-1) 열은 NaN, 
        그 뒤는 위 벡터투영 계산 결과.
    """
    # 0) 기본 정보
    tickers = X.index
    dates   = X.columns
    n_tickers, n_dates = X.shape
    
    # (에러 처리) d가 전체 날짜 수보다 클 경우
    if d > n_dates:
        raise ValueError(f"Window size d={d} is larger than number of columns={n_dates}.")

    # 1) values로 변환
    X_val = X.values  # shape=(n_tickers, n_dates)
    Y_val = Y.values

    # 2) rolling window를 위한 sliding_window_view
    #    X_rolled: shape=(n_tickers, n_windows, d)
    #    n_windows = n_dates - d + 1
    X_rolled = np.lib.stride_tricks.sliding_window_view(
        X_val, window_shape=d, axis=1
    )
    Y_rolled = np.lib.stride_tricks.sliding_window_view(
        Y_val, window_shape=d, axis=1
    )
    # n_windows = n_dates - d + 1
    _, n_windows, _ = X_rolled.shape

    # 3) 각 윈도우에 대해 dot_xy, dot_yy 계산
    #    dot_xy[i, w] = Σ_{k=0..d-1} X_rolled[i,w,k] * Y_rolled[i,w,k]
    dot_xy = (X_rolled * Y_rolled).sum(axis=2)  # shape=(n_tickers, n_windows)
    dot_yy = (Y_rolled ** 2).sum(axis=2)

    # ratio = dot_xy / dot_yy (단, dot_yy=0이면 0 처리)
    ratio = np.divide(
        dot_xy, dot_yy, out=np.zeros_like(dot_xy), where=(dot_yy != 0)
    )

    # 4) 윈도우의 마지막 날짜(= t)에서 Y[i,t] * ratio
    #    Y_rolled[:, w, -1] = 윈도우 마지막 칸(=t일)에서의 Y
    y_last = Y_rolled[:, :, -1]         # shape=(n_tickers, n_windows)
    proj_rolled = ratio * y_last       # shape=(n_tickers, n_windows)

    # 5) proj_rolled를 (n_tickers x n_windows) -> (n_tickers x n_dates)로 매핑
    #    윈도우 끝나는 날짜 인덱스: range(d-1, n_dates)
    window_end_idx = np.arange(d-1, n_dates)
    
    # 일단 partial_df를 만든 뒤, 나머지는 NaN
    partial_df = pd.DataFrame(proj_rolled, index=tickers, columns=dates[window_end_idx])
    proj_df = pd.DataFrame(np.nan, index=tickers, columns=dates)
    
    # (d-1)번째 열부터 partial_df 값을 대입
    proj_df.iloc[:, d-1:] = partial_df.values

    return proj_df


def ts_backfill(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: backfilled days
    """
    return X.fillna(method='ffill', axis=1, limit=d)

def ts_poly_regression(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    d: int,
    k: int
) -> pd.DataFrame:
    """
    Rolling Window 다항(Polynomial) 회귀 (차수=k, Window 크기=d) 후,
    해당 윈도우의 '마지막 시점'에서의 잔차(Residual)를 반환.

    [수정사항]
    - 결과 DataFrame의 columns를 원본 X와 동일하게 유지하되,
      실제 계산 구간만 (d-1)번째 열부터 값을 넣고 나머지는 NaN 처리.
    """

    X_values = X.values
    Y_values = Y.values
    n_tickers, n_dates = X_values.shape

    # 윈도우 생성
    X_windows = np.lib.stride_tricks.sliding_window_view(
        X_values, window_shape=d, axis=1
    )
    Y_windows = np.lib.stride_tricks.sliding_window_view(
        Y_values, window_shape=d, axis=1
    )
    _, n_windows, _ = X_windows.shape
    batch_size = n_tickers * n_windows

    # 2D로 펼쳐서 (batch_size, d)
    X_windows_flat = X_windows.reshape(batch_size, d)
    Y_windows_flat = Y_windows.reshape(batch_size, d)

    # 차수(k)에 따른 basis matrix
    powers = np.arange(k + 1)  # 0,1,2,...k
    X_design_flat = X_windows_flat[:, :, None] ** powers  # (batch_size, d, k+1)

    # (X^T X), (X^T Y) 계산
    XtX = np.einsum('bwd,bwe->bde', X_design_flat, X_design_flat)
    Xty = np.einsum('bwd,bw->bd', X_design_flat, Y_windows_flat)

    # (XtX)^+ * (X^T Y)
    XtX_inv = safe_pinv(XtX)
    Beta = np.einsum('bde,bd->be', XtX_inv, Xty)  # (batch_size, k+1)

    # 예측값
    Beta_expanded = Beta[:, None, :]
    Y_pred_flat = np.einsum('bwd,bnd->bw', X_design_flat, Beta_expanded)
    Residuals_flat = Y_windows_flat - Y_pred_flat
    # shape 복원
    Residuals = Residuals_flat.reshape(n_tickers, n_windows, d)
    # 각 윈도우 마지막 시점의 Residual
    Residuals_last = Residuals[:, :, -1]

    # 롤링 윈도우 끝나는 날짜 인덱스
    window_end_indices = np.arange(d - 1, n_dates)
    partial_df = pd.DataFrame(
        Residuals_last,
        index=X.index,
        columns=X.columns[window_end_indices]
    )

    # 전체 컬럼(NaN) df 생성 후, 해당 구간만 채움
    full_df = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
    full_df.iloc[:, d - 1:] = partial_df.values

    return full_df

def ts_product(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: timeseries production
    """
    return X.rolling(window=d, axis=1).agg(lambda x: x.prod())

def ts_regression(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    d: int,
    rettype: int = 0
) -> pd.DataFrame:
    """
    Rolling Window 단순 선형회귀 (Window 크기 = d) 후, 결과(Residual/Alpha/Beta/Y_hat/R^2)를 반환.

    [수정사항]
    - 결과 DataFrame의 columns를 원본 X와 동일하게 유지하되,
      실제로 값이 계산되는 구간만 (d-1)번째 열부터 채움.
      나머지는 NaN으로 유지.
    """

    X_values = X.values  # shape: (n_tickers, n_dates)
    Y_values = Y.values
    X = X.astype('float64')
    Y = Y.astype('float64')
    n_tickers, n_dates = X_values.shape
    # np.lib.stride_tricks.sliding_window_view 를 통해
    # 각 ticker별 (n_windows x d) 윈도우 추출
    X_windows = np.lib.stride_tricks.sliding_window_view(
        X_values, window_shape=d, axis=1
    )
    Y_windows = np.lib.stride_tricks.sliding_window_view(
        Y_values, window_shape=d, axis=1
    )

    # n_windows = n_dates - d + 1
    _, n_windows, _ = X_windows.shape
    # Window별 합(ΣX, ΣY), 제곱합(ΣX^2), 곱합(ΣX·Y) 계산
    sum_x = X_windows.sum(axis=2)
    sum_y = Y_windows.sum(axis=2)
    sum_xx = (X_windows ** 2).sum(axis=2)
    sum_xy = (X_windows * Y_windows).sum(axis=2)
    # Beta, Alpha 계산
    # Beta = [d*Σ(XY) - ΣX·ΣY] / [d*Σ(X^2) - (ΣX)^2]
    # Alpha = (ΣY - Beta·ΣX) / d
    beta_numer = d * sum_xy - (sum_x * sum_y)
    beta_denom = d * sum_xx - (sum_x ** 2)
    beta_denom = np.where(beta_denom == 0, np.nan, beta_denom)  # 분모=0 방지
    beta = beta_numer / beta_denom
    alpha = (sum_y - beta * sum_x) / d
    # 각 윈도우의 마지막 X, Y (즉 윈도우 끝날 때 기준값)
    X_last = X_windows[:, :, -1]  # shape: (n_tickers, n_windows)
    Y_last = Y_windows[:, :, -1]
    Y_pred_last = alpha + beta * X_last
    residuals_last = Y_last - Y_pred_last

    # R^2 등 계산
    if rettype == 4:
        # R^2 = 1 - (SS_res / SS_tot)
        mean_y = sum_y / d
        SS_res = ((Y_windows - (alpha[:, :, None] + beta[:, :, None] * X_windows)) ** 2).sum(axis=2)
        SS_tot = ((Y_windows - mean_y[:, :, None]) ** 2).sum(axis=2)
        R2 = 1 - SS_res / SS_tot
        data = R2
    else:
        if rettype == 0:
            data = residuals_last
        elif rettype == 1:
            data = alpha
        elif rettype == 2:
            data = beta
        elif rettype == 3:
            data = Y_pred_last
        else:
            raise ValueError("Invalid rettype. Must be 0, 1, 2, 3, or 4.")

    # 결과를 (tickers x n_windows) 형태로 만든 뒤,
    # 원래 (tickers x n_dates) 크기 DataFrame에 매핑
    window_end_indices = np.arange(d - 1, n_dates)  # 롤링 윈도우 끝나는 지점들
    partial_df = pd.DataFrame(data, index=X.index, columns=X.columns[window_end_indices])
    # 전체 컬럼 유지용 NaN DataFrame 생성
    full_df = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
    # (d-1)번째 컬럼부터 값 대입
    full_df.iloc[:, d - 1:] = partial_df.values
    return full_df


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
    return (((X-mean_x) / std)**3).rolling(window=d, axis=1).mean()


def ts_kurtosis(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: kurtosis of X
    """
    mean_x = ts_mean(X, d)
    std = ts_std_dev(X, d)
    return (((X-mean_x) / std)**4).rolling(window=d, axis=1).mean()


def ts_sum(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: sum of X in d days
    """
    X = X.fillna(0)
    return X.rolling(window=d, axis=1).sum()


def ts_rank(X:pd.DataFrame, d:int):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param d: lookback days
    :return: scaled ascending rank value from 0 to 1 in d days.
    """
    n_tickers, n_dates = X.shape
    if d > n_dates:
        raise ValueError("Window size d cannot be larger than the number of dates in X.")

    X_values = X.values  # Shape: (n_tickers, n_dates)
    n_windows = n_dates - d + 1

    # Generate rolling windows over the columns (dates) for X
    # X_windows shape: (n_tickers, n_windows, window)
    X_windows = np.lib.stride_tricks.sliding_window_view(X_values, window_shape=d, axis=1)

    # Reshape to (batch_size, window) for vectorized operations
    batch_size = n_tickers * n_windows
    X_windows_flat = X_windows.reshape(batch_size, d)

    # Use pandas DataFrame for vectorized ranking with 'min' method
    df_windows = pd.DataFrame(X_windows_flat)
    # Compute ranks along the columns (axis=1)
    df_ranks = df_windows.rank(axis=1, method='min', ascending=True)

    # Extract the rank of the last value in each window
    last_value_rank = df_ranks.iloc[:, -1].values  # Shape: (batch_size,)

    # Scale ranks between 0 and 1, ensuring rational increments
    scaled_ranks = (last_value_rank - 1) / (d - 1)

    # Reshape to (n_tickers, n_windows)
    scaled_ranks = scaled_ranks.reshape(n_tickers, n_windows)

    # Prepare the columns corresponding to the end dates of each window
    window_end_indices = np.arange(d - 1, n_dates)
    columns = X.columns[window_end_indices]

    # Create the result DataFrame with the calculated ranks
    result_df = pd.DataFrame(scaled_ranks, index=X.index, columns=columns)

    # Create a full DataFrame with NaN for initial periods where the window is incomplete
    full_result = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
    full_result.iloc[:, d - 1:] = result_df.values

    return full_result

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
    dot_xy = (X * Y).sum(axis=0)  # X dot Y
    dot_yy = (Y * Y).sum(axis=0)  # Y dot Y

    # 혹시 분모가 0인 날짜가 있을 수도 있으니 처리
    ratio = dot_xy / dot_yy
    ratio = ratio.replace([np.inf, -np.inf], 0).fillna(0)

    # ratio를 각 컬럼에 브로드캐스팅
    proj = Y.multiply(ratio, axis=1)
    return proj


def regression_neut(X:pd.DataFrame, Y:pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :param Y: columns=Timeseries, index=Tickers Dataframe
    :return: Y - (a+bX)
    """
    X, Y = X.fillna(0), Y.fillna(0)
    X, Y = X.replace([np.inf, -np.inf], 0), Y.replace([np.inf, -np.inf], 0)

    inv = safe_pinv(np.dot(X.T, X))
    projection_matrix = np.dot(inv, X.T)
    projection_matrix = np.dot(X, projection_matrix)
    pred_y = np.dot(projection_matrix, Y)
    return pd.DataFrame(Y - pred_y, index=X.index, columns=X.columns)


def scale_down(X:pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :return: scale down X from 0 to 1 for each day
    """
    min_x = X.min()
    max_x = X.max()
    df = (X-min_x) / (max_x - min_x)
    df = df.fillna(0)
    return df

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
    rank = X.rank(axis=0, method='min', ascending=True)
    return scale_down(rank)


def zscore(X:pd.DataFrame):
    """
    :param X: columns=Timeseries, index=Tickers Dataframe
    :return: zscore in the universe
    """
    col_mean = X.fillna(0).mean(axis=0)
    col_std = X.fillna(0).std(axis=0)

    # 2) 분모=0 or NaN 방지
    col_std = col_std.replace(0, np.nan)
    col_std = col_std.fillna(1e-9)

    # 3) Z-score
    Z = (X - col_mean) / col_std

    # 4) 최종 NaN은 0 등으로 대체 가능
    Z = Z.fillna(0)

    return Z

def vec_sum(X:dict[str, pd.DataFrame]):
    """
    :param X: vector type data
    :return: sum of each cell
    """
    # --- 합(sum)과 개수(count)를 이용한 평균 구하기 ---
    sum_df = None

    for df in X.values():
        # 아직 초기값이 없으면 현재 DataFrame을 기준으로 생성
        if sum_df is None:
            sum_df = df.fillna(0).copy()  # NaN은 0으로 대체하여 합산
        else:
            sum_df += df.fillna(0)
    return sum_df


def vec_avg(X:dict[str, pd.DataFrame]):
    """
    :param X: vector type data
    :return: average of each cell
    """
    return vec_sum(X)/vec_count(X)


def vec_min(X:dict[str, pd.DataFrame]):
    """
    :param X: vector type data
    :return: minimum of each cell
    """
    min_df = None
    for df in X.values():
        # NaN은 비교에 영향을 주지 않도록 +∞로 치환
        filled_df = df.fillna(np.inf)
        if min_df is None:
            min_df = filled_df.copy()
        else:
            # 셀 단위로 최소값
            min_df = np.minimum(min_df, filled_df)
    # 다시 +∞를 NaN으로 복원
    min_df = min_df.replace(np.inf, np.nan)
    return min_df

def vec_max(X:dict[str, pd.DataFrame]):
    """
    :param X: vector type data
    :return: maximum of each cell
    """
    max_df = None
    for df in X.values():
        # NaN은 비교에 영향을 주지 않도록 -∞로 치환
        filled_df = df.fillna(-np.inf)
        if max_df is None:
            max_df = filled_df.copy()
        else:
            # 셀 단위로 최소값
            max_df = np.maximum(max_df, filled_df)
    # 다시 -∞를 NaN으로 복원
    min_df = max_df.replace(-np.inf, np.nan)
    return min_df

def vec_count(X: dict[str, pd.DataFrame]):
    """
    :param X: vector type data
    :return: number of matrix in each cell
    """
    count_df = None
    for df in X.values():
        # 아직 초기값이 없으면 현재 DataFrame을 기준으로 생성
        if count_df is None:
            count_df = df.notna().astype(int)  # NaN이 아닌 셀만 1로 카운트
        else:
            count_df += df.notna().astype(int)

    # 유효값이 전혀 없는 셀(= count_df == 0)은 0으로 나누어지지 않도록 NaN으로 처리
    count_df = count_df.replace(0, np.nan)

    return count_df


def vec_std_dev(X:dict[str, pd.DataFrame]):
    """
    :param X: vector type data
    :return: std dev of each cell
    """
    sum_df = None  # 각 셀의 x_i를 모두 더한 누적 합 (sum of x)
    sum_sq_df = None  # 각 셀의 x_i^2를 모두 더한 누적 합 (sum of x^2)
    count_df = None  # 각 셀의 유효값 개수 (n)

    for df in X.values():
        # 현재 df에서 NaN이 아닌 값만 반영하기 위해 fillna(0)를 사용
        filled = df.fillna(0)
        # 유효값(= NaN이 아님) 마스크
        valid_mask = df.notna().astype(int)

        if sum_df is None:
            # 처음 한 번만 초기화
            sum_df = filled.copy()
            sum_sq_df = (df ** 2).fillna(0).copy()  # x^2의 합
            count_df = valid_mask.copy()
        else:
            sum_df += filled
            sum_sq_df += (df ** 2).fillna(0)
            count_df += valid_mask

    # 이제 각 셀에 대해 sum(x), sum(x^2), n(유효값 개수)가 구해졌음
    # 표본 표준편차 계산: sqrt( ( sum(x^2) - (sum(x)^2 / n ) ) / (n - 1) )

    # (n-1) 이 0 이하인 곳은 계산 불가 => NaN 처리
    # 먼저 (n < 2) 인 곳은 NaN으로 만들어 두자
    invalid_mask = (count_df < 2)
    count_df2 = count_df.copy()
    count_df2[invalid_mask] = np.nan  # 0이나 1인 경우 -> NaN으로 만들어 버림

    # 편차 제곱합
    numerator = sum_sq_df - (sum_df ** 2) / count_df2

    # 표본 분산
    var_df = numerator / (count_df2 - 1)

    # 분산이 음수가 되는 경우(부동소수점 오차 등)는 0으로 클리핑
    # (수학적으로 음수가 될 수 없으므로, 아주 미세한 오차일 가능성이 큼)
    var_df = var_df.clip(lower=0)

    # 표준편차
    std_df = np.sqrt(var_df)
    # n < 2인 곳은 최종적으로 NaN 처리
    std_df[invalid_mask] = np.nan

    return std_df

def vec_backfill(X: dict[str, pd.DataFrame], d: int):
    """
    :param X: vector type data
    :param d: lookback
    :return: backfilled vector
    """
    new_dict = {}
    for key in X.keys():
        new_dict[key] = ts_backfill(X[key], 252)
    return new_dict

def _groupby_transform_ignorena(
    s: pd.Series, group_labels: pd.Series, func='mean'
) -> pd.Series:
    """
    s (한 칼럼의 값), group_labels(동일 index, 각 row가 속한 그룹 레이블)
    => group_labels가 NaN인 행은 무시
    => s가 NaN인 행은 무시
    => 해당 group에 속한 유효값들만 모아서 func 적용
    => transform 결과를 같은 index로 돌려줌
    """
    df_temp = pd.DataFrame({'val': s, 'grp': group_labels})
    # NaN인 행(그룹 레이블 없거나 값이 없거나)은 제외
    df_temp = df_temp.dropna(subset=['val', 'grp'])

    # groupby.transform(func)를 통해 그룹별 동일한 값(평균, 합 등)으로 broadcast
    transformed = df_temp.groupby('grp')['val'].transform(func)

    # 원래 index 형태로 맞추기
    out = pd.Series(index=s.index, dtype=float)
    out.loc[transformed.index] = transformed
    return out


def group_mean_masked(
    X: pd.DataFrame, group: pd.DataFrame, constraints: pd.DataFrame
) -> pd.DataFrame:
    """
    constraints=True인 (X, group) 값들만 사용해서,
    '각 날짜/그룹별 평균'을 계산하고, 그 값을 동일한 shape로 리턴.
    (constraints=False인 위치는 NaN으로 둔다)
    """
    def aggregator_for_mean(s: pd.Series) -> pd.Series:
        # s.name = 현재 '열(날짜)' 이름
        col_group = group[s.name]           # 이 날짜에서 각 종목이 속한 그룹
        col_mask  = constraints[s.name]     # 이 날짜에서 True/False 마스크
        # => True인 셀만 대상으로 그룹 평균
        #    X값도 마찬가지로 False는 NaN 처리(무시)
        s_masked = s.where(col_mask, np.nan)

        # groupby transform으로 그룹 평균
        means = _groupby_transform_ignorena(s_masked, col_group, func='mean')
        # means는 True 위치에만 그룹평균 값이 있고, 나머지는 NaN
        return means

    # 각 날짜(열)에 대해 aggregator_for_mean을 적용
    return X.apply(aggregator_for_mean, axis=0)


def group_neutralize_with_constraints(
    X: pd.DataFrame, group: pd.DataFrame, constraints: pd.DataFrame
) -> pd.DataFrame:
    """
    (1) constraints=False인 셀은 0으로 처리
    (2) constraints=True인 셀만 그룹평균 계산 후 중립화 진행
    (3) 각 날짜별로 constraints=True인 셀들의 절대값 합이 1이 되도록 정규화
    (4) 재정규화 전후에 제약 마스크를 다시 곱해, 허용되지 않은 셀에 값이 들어가지 않도록 함
    최종 결과: False인 셀은 0, True인 셀은 '그룹합=0' & '날짜별 sum(|X|)=1'
    """

    # 0) constraints를 불리언 마스크로 변환 (확실히 True/False가 되도록)
    constraints_bool = constraints.astype(bool)

    # 1) 제약조건이 False인 셀은 0으로 처리
    X_masked = X.where(constraints_bool, 0.0)

    # 2) 그룹별 평균 계산 (오직 constraints=True인 셀만 대상으로)
    group_means = group_mean_masked(X_masked, group, constraints_bool)
    X_neut = X_masked - group_means

    # 3) 초기 정규화: 각 날짜별로, constraints=True인 셀들의 abs합으로 나누어 스케일링
    sumabs = X_neut.abs().where(constraints_bool, 0).sum(axis=0)
    X_scaled = X_neut.div(sumabs, axis=1).fillna(0)

    # 4) 다시 제약 마스크를 곱해 미세 오차 등으로 인해 허용되지 않은 셀에 값이 남는 경우를 제거
    X_scaled = X_scaled.where(constraints_bool, 0.0)

    # 5) 재정규화: 제약조건(True)인 셀들의 절대합을 다시 1로 맞추기
    sumabs_allowed = X_scaled.abs().where(constraints_bool, 0).sum(axis=0)
    X_final = X_scaled.div(sumabs_allowed, axis=1).fillna(0)

    return X_final


def group_neutralize(X: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    """
    (1) group_mean(X, group)를 이용해 '그룹별 평균'을 빼서, 각 날짜/그룹 합=0 달성
    (2) 각 날짜별 sum(abs(.))=1 이 되도록 스케일링

    :param X:      [tickers x dates], 예: 팩터 값 or 포트폴리오 비중
    :param group:  [tickers x dates], 그룹 라벨(섹터명 등)
                -> 각 날짜별로 종목이 속한 그룹을 나타내는 문자열(또는 코드)
    :return:       [tickers x dates], 중립화 & gross=1 스케일링된 결과
    """

    # 1) 그룹 중립화: X - group_mean(X, group)
    gmean = group_mean(X, group)       # (tickers x dates), 각 날짜/그룹 평균
    X_neut = X - gmean                 # 그룹별 합=0 달성

    # 2) 각 날짜별 절대값 합 계산
    abs_sum = X_neut.abs().sum(axis=0) # shape: (n_dates,)
    
    # 3) 각 날짜별로 abs_sum으로 나누어 => sum(|X|)=1
    #    abs_sum=0 인 날이 있으면 0 으로 채움
    #    => (모두 0이라면 그대로 0)
    result = X_neut.div(abs_sum, axis=1).fillna(0)

    return result

def group_rank(X: pd.DataFrame, group: pd.DataFrame):
    """
    :param X:      columns=Timeseries, index=Tickers DataFrame
    :param group:  columns=Timeseries, index=Tickers Group Dataframe
    :return: rank value in the same group
    """
     # 1) stack
    #  - X_stacked, group_stacked은 각각 (티커, 열) MultiIndex에 대한 1차원 시리즈
    X_stacked = X.stack(dropna=False)
    group_stacked = group.stack(dropna=False)

    # 2) 하나의 DataFrame으로 합침
    df_stacked = pd.DataFrame({
        'value': X_stacked,
        'grp': group_stacked
    })
    # df_stacked.index: (티커, 컬럼)
    # df_stacked.columns: ['value', 'grp']

    # 3) [컬럼, grp] 기준으로 groupby → rank
    #  - ascending=True 이므로 큰 값일수록 rank가 큼
    df_stacked['rank'] = df_stacked.groupby(
        [df_stacked.index.get_level_values(1), 'grp']
    )['value'].rank(method='min', ascending=True)

    # 4) rank_min, rank_max를 그룹별로 구한 뒤 transform('min'), transform('max')
    df_stacked['rank_min'] = df_stacked.groupby(
        [df_stacked.index.get_level_values(1), 'grp']
    )['rank'].transform('min')

    df_stacked['rank_max'] = df_stacked.groupby(
        [df_stacked.index.get_level_values(1), 'grp']
    )['rank'].transform('max')

    # 5) 스케일링: (rank - rank_min) / (rank_max - rank_min)
    #    분모가 0인 경우(그룹 내 종목이 1개뿐 등)는 0으로 처리
    eps = 1e-9
    denom = (df_stacked['rank_max'] - df_stacked['rank_min']).replace(0, np.nan)
    df_stacked['scaled'] = (df_stacked['rank'] - df_stacked['rank_min']) / denom
    df_stacked['scaled'] = df_stacked['scaled'].fillna(0)

    # 6) 다시 2차원 형태로 복원 (unstack)
    #    - level=1이 날짜(열)이므로, unstack(level=1)을 하면 열 축으로 복원
    result = df_stacked['scaled'].unstack(level=1)

    # 7) 원래 순서대로 reindex
    result = result.reindex(index=X.index, columns=X.columns)

    return result

def group_zscore(X: pd.DataFrame, group: pd.DataFrame):
    return (X - group_mean(X, group))/group_std_dev(X, group)

def group_sum(X: pd.DataFrame, group: pd.DataFrame):
    def aggregator_for_sum(s: pd.Series) -> pd.Series:
        col_group = group[s.name]
        return _groupby_transform_ignorena(s, col_group, 'sum')
    return X.apply(aggregator_for_sum)

def group_mean(X: pd.DataFrame, group: pd.DataFrame):
    def aggregator_for_mean(s: pd.Series) -> pd.Series:
        col_group = group[s.name]
        return _groupby_transform_ignorena(s, col_group, 'mean')
    return X.apply(aggregator_for_mean)

def group_max(X: pd.DataFrame, group: pd.DataFrame):
    def aggregator_for_max(s: pd.Series) -> pd.Series:
        col_group = group[s.name]
        return _groupby_transform_ignorena(s, col_group, 'max')
    return X.apply(aggregator_for_max)

def group_min(X: pd.DataFrame, group: pd.DataFrame):
    def aggregator_for_min(s: pd.Series) -> pd.Series:
        col_group = group[s.name]
        return _groupby_transform_ignorena(s, col_group, 'min')
    return X.apply(aggregator_for_min)

def group_std_dev(X: pd.DataFrame, group: pd.DataFrame):
    def aggregator_for_std(s: pd.Series) -> pd.Series:
        col_group = group[s.name]
        return _groupby_transform_ignorena(s, col_group, 'std')
    return X.apply(aggregator_for_std)

def _groupby_transform_ignorena(s: pd.Series, g: pd.Series, func: str) -> pd.Series:
    """
    s : X의 '한 컬럼' (tickers x 1개 date)
    g : group의 '한 컬럼' (tickers x 1개 date)
    func : 'sum', 'mean', 'max', 'min', 'std' 등 집계 함수
    """
    # (1) s와 g를 합쳐서 DataFrame을 만들고,
    #     group 컬럼이 NaN인 행은 drop
    df = pd.DataFrame({'val': s, 'grp': g})
    df_drop = df.dropna(subset=['grp'])  # grp=NaN인 행 제외

    # (2) groupby-transform
    #     => 예: df_drop.groupby('grp')['val'].transform('sum')
    df_drop['agg'] = df_drop.groupby('grp')['val'].transform(func)

    # (3) 원래 인덱스 순서를 유지하기 위해 reindex
    #     group=NaN이었던 행은 집계결과가 없으므로 NaN 유지
    df_result = df_drop.reindex(df.index)

    return df_result['agg']

def group_vector_proj(X: pd.DataFrame, Y: pd.DataFrame, group: pd.DataFrame):
    """
    (각 날짜, 그룹별로)
      1) X, Y를 같은 그룹인 종목끼리 묶어
      2) '벡터 투영' 계산: 
         Proj_X_on_Y = (X dot Y / Y dot Y) * Y
      3) 최종적으로 X와 동일 shape의 DataFrame 리턴
    """

    # 1) stack하여 한 번에 처리 (날짜, 그룹별 연산)
    #    index: (티커, date), columns: [X, Y, group]
    #    -> 그러나 여기서는 X, Y, group이 모두 (행=종목, 열=날짜)이므로
    #       각 열별로 groupby를 해야 함
    #    아래는 apply(lambda col: ...) 방식으로 열을 순회
    def _proj_one_column(col_X, col_Y, col_group):
        # col_X, col_Y는 Series(index=티커), col_group도 Series(index=티커)
        # 그룹별로 벡터 투영 계산
        df = pd.DataFrame({'x': col_X, 'y': col_Y, 'grp': col_group})
        # grp 기준으로 나눠서 (x,y) 벡터의 dot prod 계산
        def _compute_proj(g):
            # g[['x','y']] -> 각각 종목별 값
            x_vec = g['x']
            y_vec = g['y']
            # dot_xy, dot_yy
            dot_xy = (x_vec * y_vec).sum()
            dot_yy = (y_vec * y_vec).sum()
            if dot_yy == 0:
                # 모든 y가 0이거나 nan이면, 투영 불가 -> 0 처리
                return pd.Series([0]*len(x_vec), index=x_vec.index)
            ratio = dot_xy / dot_yy
            proj_vec = y_vec * ratio
            return proj_vec

        # 그룹별 apply
        df['proj'] = df.groupby('grp').apply(_compute_proj).reset_index(level=0, drop=True)
        return df['proj']
    result = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for c in X.columns:
        result[c] = _proj_one_column(X[c], Y[c], group[c])
    return result

def group_scale(X: pd.DataFrame, Y: pd.DataFrame, group: pd.DataFrame):
    """
    (각 날짜, 그룹별로)
      - X를 Min-Max 정규화: (X - min) / (max - min)
      - 이때 Y는 굳이 안 쓰거나, 다른 파라미터로 확장 가능
        (현재 예시에서는 Y 사용 X)
    """
    def _scale_one_column(col_X, col_group):
        df = pd.DataFrame({'x': col_X, 'grp': col_group})
        def _compute_scale(g):
            x_vec = g['x']
            mn = x_vec.min()
            mx = x_vec.max()
            if mx == mn:
                return pd.Series([0]*len(g), index=g.index)
            return (x_vec - mn) / (mx - mn)
        df['scaled'] = df.groupby('grp').apply(_compute_scale).reset_index(level=0, drop=True)
        return df['scaled']
    result = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for c in X.columns:
        result[c] = _scale_one_column(X[c], group[c])
    return result

def group_vector_neut(X: pd.DataFrame, Y: pd.DataFrame, group: pd.DataFrame):
    return X - group_vector_proj(X, Y, group)

def group_cartesian_product(group1: pd.DataFrame, group2: pd.DataFrame):
    result = pd.DataFrame(index=group1.index, columns=group1.columns, dtype=object)
    for c in group1.columns:
        # group1[c], group2[c] 각각 ticker별 라벨
        if group1[c].dtype == 'object':
            s1 = group1[c].fillna('Unknown')
        else:
            s1 = group1[c].fillna(0)

        if group2[c].dtype == 'object':
            s2 = group2[c].fillna('Unknown')
        else:
            s2 = group2[c].fillna(0)
        combined = s1.astype(str) + '|' + s2.astype(str)
        result[c] = combined
    return result

def basket(X: pd.DataFrame, tile: list[float]):
    """
    generate group datafield based on X and tile
    EX) X = get_data('MarketCap'), tile = [0.33, 0.66]
    Then, set 0~0.33 rank of X be 0, 0.33~0.66 rank of X be 1, and 0.66~1 rank of X be 2.
    """
    baskets = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    
    # 각 날짜(열)마다 반복
    for col in X.columns:
        col_series = X[col]
        
        # 1) NaN을 제외한 값에 대해 rank() 수행
        #    - method='dense': 순위 간격을 1씩 부여, 예) [10,10,15] -> rank=[1,1,2]
        #    - ascending=True : 값이 작은게 rank가 작음
        ranks = col_series.rank(method='dense', ascending=True)
        
        # 2) rank를 [0,1] 구간으로 스케일링
        #    - (rank - min) / (max - min)
        rank_min = ranks.min()
        rank_max = ranks.max()
        
        # 만약 모든 값이 NaN이거나 동일하다면(분모=0) → 해당 열 전부 NaN 처리
        if pd.isna(rank_min) or pd.isna(rank_max) or rank_min == rank_max:
            baskets[col] = np.nan
            continue
        
        scaled_ranks = (ranks - rank_min) / (rank_max - rank_min)
        
        # 3) tile 경계값을 이용해 pd.cut() 실행
        #    - bins 예시: [0, 0.33, 0.66, 1.0000001]
        #    - right=False → 구간 [이상, 미만) 으로 처리
        bins = [0] + tile + [1.0000001]  # upper bound를 약간 더 높게
        labels = list(range(len(bins) - 1))  # 0, 1, 2, ...
        
        # cut으로 구간 분할
        col_baskets = pd.cut(
            scaled_ranks, 
            bins=bins, 
            labels=labels, 
            right=False
        )
        
        baskets[col] = col_baskets
    
    # 그룹 레이블을 int로 맞춤
    return baskets.astype('Int64')  # or just int, but allow NA → 'Int64' for nullable int

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
