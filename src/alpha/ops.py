# src/alpha/ops.py
import pandas as pd
import numpy as np
import scipy
from collections import deque
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# [0] Core Utils & Safety
# ==============================================================================

def safe_pinv(A: np.ndarray) -> np.ndarray:
    """Singular Matrix 에러 방지용 Pseudo-inverse"""
    try:
        return scipy.linalg.pinv(A)
    except np.linalg.LinAlgError:
        eps = 1e-8
        A_reg = A + np.eye(A.shape[0]) * eps
        try:
            return scipy.linalg.pinv(A_reg)
        except:
            return np.zeros_like(A)

def _roll(X: pd.DataFrame, d: int):
    """
    [Standard Rolling]
    axis=0 (Time-series Direction)
    min_periods=d//2 (데이터 절반만 있어도 계산)
    """
    return X.rolling(window=d, min_periods=max(1, d // 2), axis=0)

# ==============================================================================
# [1] Time-Series Operators (시계열 연산)
# - 데이터 형상: (Index=Date, Columns=Ticker)
# - 연산 방향: axis=0 (시간축)
# ==============================================================================

def ts_delay(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return X.shift(d, axis=0).fillna(0)

def ts_delta(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return X - ts_delay(X, d)

def ts_return(X: pd.DataFrame, d: int) -> pd.DataFrame:
    denom = ts_delay(X, d)
    return (X - denom) / denom.replace(0, np.nan)

def ts_returns(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return ts_return(X, d)

def ts_mean(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return _roll(X, d).mean()

def ts_std_dev(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return _roll(X, d).std()

def ts_max(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return _roll(X, d).max()

def ts_min(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return _roll(X, d).min()

def ts_sum(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return _roll(X, d).sum()

def ts_product(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return _roll(X, d).apply(np.prod, raw=True)

def ts_median(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return _roll(X, d).median()

def ts_argmax(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return _roll(X, d).apply(lambda x: (len(x) - 1) - np.argmax(x), raw=True)

def ts_argmin(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return _roll(X, d).apply(lambda x: (len(x) - 1) - np.argmin(x), raw=True)

def ts_rank(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return _roll(X, d).rank(pct=True)

def ts_backfill(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return X.fillna(method='ffill', axis=0, limit=d)

def ts_ema(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return X.ewm(span=d, axis=0, adjust=False).mean()

def ts_weighted_decay(X: pd.DataFrame, k: float = 0.5) -> pd.DataFrame:
    return k * X + (1 - k) * ts_delay(X, 1)

def ts_linear_decay(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return X.rolling(window=d, win_type='triang', min_periods=d//2, axis=0).mean()

def ts_av_diff(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return X - ts_mean(X, d)

def ts_max_diff(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return X - ts_max(X, d)

def ts_min_diff(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return X - ts_min(X, d)

def ts_min_max_cps(X: pd.DataFrame, d: int, f: float = 2.0) -> pd.DataFrame:
    """Min + Max - f * X"""
    return ts_min(X, d) + ts_max(X, d) - f * X

def ts_min_max_diff(X: pd.DataFrame, d: int, f: float = 0.5) -> pd.DataFrame:
    """X - f * (Min + Max)"""
    return X - f * (ts_min(X, d) + ts_max(X, d))

def ts_scale(X: pd.DataFrame, d: int) -> pd.DataFrame:
    mn = ts_min(X, d)
    mx = ts_max(X, d)
    return (X - mn) / (mx - mn).replace(0, np.nan)

# --- Advanced Time-Series Statistics ---

def ts_skewness(X: pd.DataFrame, d: int) -> pd.DataFrame:
    mean = ts_mean(X, d)
    std = ts_std_dev(X, d)
    z = (X - mean) / std.replace(0, np.nan)
    return z.pow(3).rolling(d, min_periods=d//2, axis=0).mean()

def ts_kurtosis(X: pd.DataFrame, d: int) -> pd.DataFrame:
    mean = ts_mean(X, d)
    std = ts_std_dev(X, d)
    z = (X - mean) / std.replace(0, np.nan)
    return z.pow(4).rolling(d, min_periods=d//2, axis=0).mean()

def ts_co_kurtosis(X: pd.DataFrame, Y: pd.DataFrame, d: int) -> pd.DataFrame:
    dev_x = ts_av_diff(X, d)
    dev_y = ts_av_diff(Y, d)
    std_x = ts_std_dev(X, d)
    std_y = ts_std_dev(Y, d)
    num = (dev_x * (dev_y ** 3)).rolling(d, min_periods=d//2, axis=0).mean()
    den = std_x * (std_y ** 3)
    return num / den.replace(0, np.nan)

def ts_vector_proj(X: pd.DataFrame, Y: pd.DataFrame, d: int) -> pd.DataFrame:
    dot_xy = (X * Y).rolling(d, min_periods=d//2, axis=0).sum()
    dot_yy = (Y * Y).rolling(d, min_periods=d//2, axis=0).sum()
    ratio = dot_xy / dot_yy.replace(0, np.nan)
    return ratio * Y

def ts_vector_neut(X: pd.DataFrame, Y: pd.DataFrame, d: int) -> pd.DataFrame:
    return X - ts_vector_proj(X, Y, d)

def ts_covariance(X: pd.DataFrame, Y: pd.DataFrame, d: int) -> pd.DataFrame:
    mean_x = ts_mean(X, d)
    mean_y = ts_mean(Y, d)
    mean_xy = ts_mean(X * Y, d)
    return (mean_xy - mean_x * mean_y) * (d / (d - 1))

def ts_variance(X: pd.DataFrame, d: int) -> pd.DataFrame:
    mean_x = ts_mean(X, d)
    mean_x2 = ts_mean(X * X, d)
    return (mean_x2 - mean_x ** 2) * (d / (d - 1))

def ts_corr(X: pd.DataFrame, Y: pd.DataFrame, d: int) -> pd.DataFrame:
    cov = ts_covariance(X, Y, d)
    std_x = ts_std_dev(X, d)
    std_y = ts_std_dev(Y, d)
    return cov / (std_x * std_y).replace(0, np.nan)

def ts_zscore(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return (X - ts_mean(X, d)) / ts_std_dev(X, d).replace(0, np.nan)

def ts_ir(X: pd.DataFrame, d: int) -> pd.DataFrame:
    return ts_mean(X, d) / ts_std_dev(X, d).replace(0, np.nan)

def ts_regression(Y: pd.DataFrame, X: pd.DataFrame, d: int, rettype: int = 0) -> pd.DataFrame:
    cov = ts_covariance(X, Y, d)
    var = ts_variance(X, d)
    beta = cov / var.replace(0, np.nan)
    mean_x = ts_mean(X, d)
    mean_y = ts_mean(Y, d)
    alpha = mean_y - beta * mean_x
    
    if rettype == 1: return beta
    if rettype == 2: return alpha
    if rettype == 3: return alpha + beta * X
    return Y - (alpha + beta * X)

def ts_poly_regression(Y: pd.DataFrame, X: pd.DataFrame, d: int, k: int = 2) -> pd.DataFrame:
    """Memory Optimized Polynomial Regression"""
    residuals = pd.DataFrame(np.nan, index=Y.index, columns=Y.columns)
    X_powers = [X.pow(i) for i in range(2 * k + 1)]
    YX_powers = [Y * X.pow(i) for i in range(k + 1)]
    common_cols = Y.columns.intersection(X.columns)
    
    for col in common_cols:
        try:
            S = [xp[col].rolling(d, min_periods=d//2).sum().values for xp in X_powers]
            V = [yxp[col].rolling(d, min_periods=d//2).sum().values for yxp in YX_powers]
            T_len = len(Y)
            A = np.zeros((T_len, k+1, k+1))
            b = np.zeros((T_len, k+1))
            
            for i in range(k+1):
                b[:, i] = V[i]
                for j in range(k+1):
                    A[:, i, j] = S[i+j]
            
            idx = np.arange(k+1)
            A[:, idx, idx] += 1e-6 
            
            valid_mask = ~np.isnan(A).any(axis=(1,2)) & ~np.isnan(b).any(axis=1)
            betas = np.zeros((T_len, k+1))
            if np.any(valid_mask):
                betas[valid_mask] = np.linalg.solve(A[valid_mask], b[valid_mask])
            
            x_vals = X[col].values
            y_pred = np.zeros(T_len)
            for i in range(k+1):
                y_pred += betas[:, i] * (x_vals ** i)
            
            res = Y[col].values - y_pred
            res[~valid_mask] = np.nan
            residuals[col] = res
        except:
            continue
    return residuals

def ts_frac_diff(df: pd.DataFrame, d: float, window: int = 20) -> pd.DataFrame:
    def _get_weights(d, size):
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] * (d - k + 1) / k
            w.append(w_k)
        return np.array(w[::-1])

    weights = _get_weights(d, window)
    return df.fillna(method='ffill', axis=0).rolling(window=window, axis=0).apply(
        lambda x: np.dot(x, weights), raw=True
    )

def ts_profile(X: pd.DataFrame, d: int, bins: int = 10) -> pd.DataFrame:
    """Time-Series Profiling (Binning)"""
    T, N = X.shape
    bin_array = np.full((T, N), np.nan)
    x_arr = X.values
    
    for t in range(T):
        start = max(0, t - d + 1)
        block = x_arr[start:t+1, :] 
        valid_mask = ~np.isnan(block).all(axis=0)
        if not np.any(valid_mask): continue
        
        b_min = np.nanmin(block, axis=0)
        b_max = np.nanmax(block, axis=0)
        today = x_arr[t, :]
        span = b_max - b_min
        span[span == 0] = np.nan
        
        scaled = (today - b_min) / span * (bins - 1)
        scaled = np.clip(scaled, 0, bins - 1)
        bin_array[t, :] = np.floor(scaled)
        
    return pd.DataFrame(bin_array, index=X.index, columns=X.columns)

# ==============================================================================
# [2] Cross-Sectional Operators (횡단면 연산)
# - 연산 방향: axis=1 (종목축)
# ==============================================================================

def rank(X: pd.DataFrame) -> pd.DataFrame:
    return X.rank(axis=1, pct=True, method='min')

def scale_down(X: pd.DataFrame) -> pd.DataFrame:
    d_min = X.min(axis=1)
    d_max = X.max(axis=1)
    return X.sub(d_min, axis=0).div((d_max - d_min).replace(0, np.nan), axis=0).fillna(0)

def zscore(X: pd.DataFrame) -> pd.DataFrame:
    d_mean = X.mean(axis=1)
    d_std = X.std(axis=1)
    return X.sub(d_mean, axis=0).div(d_std.replace(0, np.nan), axis=0).fillna(0)

def vector_proj(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    dot_xy = (X * Y).sum(axis=1)
    dot_yy = (Y * Y).sum(axis=1)
    ratio = dot_xy / dot_yy.replace(0, np.nan)
    return Y.mul(ratio, axis=0).fillna(0)

def vector_neut(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    return X - vector_proj(X, Y)

def regression_neut(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    mean_x = X.mean(axis=1)
    mean_y = Y.mean(axis=1)
    mean_xy = (X * Y).mean(axis=1)
    mean_xx = (X * X).mean(axis=1)
    
    cov_xy = mean_xy - mean_x * mean_y
    var_x = mean_xx - mean_x ** 2
    
    beta = cov_xy / var_x.replace(0, np.nan)
    alpha = mean_y - beta * mean_x
    
    fitted = X.mul(beta, axis=0).add(alpha, axis=0)
    return Y.sub(fitted).fillna(0)

def neutralize_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    means = df.mean(axis=1)
    df_neut = df.sub(means, axis=0)
    gross = df_neut.abs().sum(axis=1)
    return df_neut.div(gross.replace(0, np.nan), axis=0).fillna(0)

def basket(X: pd.DataFrame, tile: list[float]):
    """Cross-sectional Basket"""
    ranks = X.rank(axis=1, pct=True, method='dense') # 사용자 원본: dense
    bins = [-1.0] + tile + [2.0]
    labels = list(range(len(bins) - 1))
    stacked = ranks.stack(dropna=False)
    baskets = pd.cut(stacked, bins=bins, labels=labels)
    return baskets.unstack().astype('float32')

# ==============================================================================
# [3] Group Operators (그룹 연산)
# - 연산 방향: 같은 날짜(Row) 내에서의 그룹핑
# ==============================================================================

def _group_operate(X: pd.DataFrame, group: pd.DataFrame, func: str) -> pd.DataFrame:
    X_st = X.stack(dropna=False)
    g_st = group.stack(dropna=False)
    df = pd.DataFrame({'val': X_st, 'grp': g_st})
    res = df.groupby([df.index.get_level_values(0), 'grp'])['val'].transform(func)
    return res.unstack()

def group_mean(X: pd.DataFrame, group: pd.DataFrame): return _group_operate(X, group, 'mean')
def group_sum(X: pd.DataFrame, group: pd.DataFrame): return _group_operate(X, group, 'sum')
def group_max(X: pd.DataFrame, group: pd.DataFrame): return _group_operate(X, group, 'max')
def group_min(X: pd.DataFrame, group: pd.DataFrame): return _group_operate(X, group, 'min')
def group_std_dev(X: pd.DataFrame, group: pd.DataFrame): return _group_operate(X, group, 'std')

def group_zscore(X: pd.DataFrame, group: pd.DataFrame):
    mean = group_mean(X, group)
    std = group_std_dev(X, group)
    return (X - mean) / std.replace(0, np.nan)

def group_neutralize(X: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    neut = X - group_mean(X, group)
    abs_sum = neut.abs().sum(axis=1)
    return neut.div(abs_sum.replace(0, np.nan), axis=0).fillna(0)

def group_rank(X: pd.DataFrame, group: pd.DataFrame):
    X_st = X.stack(dropna=False)
    g_st = group.stack(dropna=False)
    df = pd.DataFrame({'val': X_st, 'grp': g_st})
    ranks = df.groupby([df.index.get_level_values(0), 'grp'])['val'].rank(pct=True)
    return ranks.unstack()

def group_mean_masked(X: pd.DataFrame, group: pd.DataFrame, constraints: pd.DataFrame) -> pd.DataFrame:
    X_masked = X.where(constraints.astype(bool), np.nan)
    return group_mean(X_masked, group)

def group_neutralize_with_constraints(X: pd.DataFrame, group: pd.DataFrame, constraints: pd.DataFrame) -> pd.DataFrame:
    mask = constraints.astype(bool)
    gmean = group_mean_masked(X, group, mask)
    X_neut = X - gmean
    X_neut = X_neut.where(mask, 0.0)
    abs_sum = X_neut.abs().sum(axis=1)
    return X_neut.div(abs_sum.replace(0, np.nan), axis=0).fillna(0.0)

def group_scale(X: pd.DataFrame, group: pd.DataFrame):
    g_min = group_min(X, group)
    g_max = group_max(X, group)
    return (X - g_min) / (g_max - g_min).replace(0, np.nan)

def group_vector_proj(X: pd.DataFrame, Y: pd.DataFrame, group: pd.DataFrame):
    dot_xy = group_sum(X * Y, group)
    dot_yy = group_sum(Y * Y, group)
    ratio = dot_xy / dot_yy.replace(0, np.nan)
    return ratio * Y

def group_vector_neut(X: pd.DataFrame, Y: pd.DataFrame, group: pd.DataFrame):
    return X - group_vector_proj(X, Y, group)

def group_cartesian_product(group1: pd.DataFrame, group2: pd.DataFrame):
    g1 = group1.fillna('N/A').astype(str)
    g2 = group2.fillna('N/A').astype(str)
    return g1 + "_" + g2

# ==============================================================================
# [4] Logic & Execution Strategies
# ==============================================================================

def if_else(logic: pd.DataFrame, X: pd.DataFrame | float, Y: pd.DataFrame | float):
    if isinstance(logic, pd.DataFrame): logic = logic.fillna(False)
    l_val = logic.values if isinstance(logic, pd.DataFrame) else logic
    x_val = X.values if isinstance(X, pd.DataFrame) else X
    y_val = Y.values if isinstance(Y, pd.DataFrame) else Y
    res = np.where(l_val, x_val, y_val)
    if isinstance(logic, pd.DataFrame):
        return pd.DataFrame(res, index=logic.index, columns=logic.columns)
    return res

def trade_when(in_trigger: pd.DataFrame, X: pd.DataFrame, exit_trigger: pd.DataFrame) -> pd.DataFrame:
    """Stateful Trade Logic"""
    arr_in = in_trigger.fillna(False).values
    arr_out = exit_trigger.fillna(False).values
    arr_X = X.fillna(0).values
    pos = np.zeros_like(arr_X)
    
    pos[0] = np.where(arr_out[0], 0, np.where(arr_in[0], arr_X[0], 0))
    for i in range(1, len(X)):
        # i번째 Row (Time)
        pos[i] = np.where(arr_out[i], 0, np.where(arr_in[i], arr_X[i], pos[i-1]))
    return pd.DataFrame(pos, index=X.index, columns=X.columns)

def hump(X: pd.DataFrame, alpha=0.5, threshold=0.002) -> pd.DataFrame:
    """Execution Hump (Date-wise iteration)"""
    X = neutralize_and_scale(X)
    final_port = pd.DataFrame(0.0, index=X.index, columns=X.columns)
    
    final_port.iloc[0] = X.iloc[0].fillna(0)
    for i in range(1, len(X)):
        w_old = final_port.iloc[i-1]
        w_target = X.iloc[i].fillna(0)
        diff = w_target - w_old
        mask = diff.abs() > threshold
        w_new = w_old.copy()
        w_new[mask] = w_old[mask] + alpha * diff[mask]
        final_port.iloc[i] = w_new
    return final_port

def hump_ts_rank(X: pd.DataFrame, alpha_min=0.0, alpha_max=1.0, window=60) -> pd.DataFrame:
    """Adaptive Hump based on Diff Rank"""
    X = neutralize_and_scale(X)
    final_port = pd.DataFrame(0.0, index=X.index, columns=X.columns)
    final_port.iloc[0] = X.iloc[0].fillna(0)
    
    diff_history = {tkr: deque(maxlen=window) for tkr in X.columns}
    
    for i in range(1, len(X)):
        w_old = final_port.iloc[i-1]
        w_target = X.iloc[i].fillna(0)
        diff = w_target - w_old
        diff_abs = diff.abs()
        
        alpha_series = pd.Series(index=X.columns, dtype=float)
        
        for tkr in X.columns:
            hist = diff_history[tkr]
            this_diff = diff_abs[tkr]
            
            if len(hist) == 0:
                alpha_series[tkr] = alpha_min
            else:
                arr = np.array(hist)
                rank_raw = (arr < this_diff).sum() + 1
                scaled_rank = (rank_raw - 1) / len(arr)
                alpha_series[tkr] = alpha_min + (alpha_max - alpha_min) * scaled_rank
            
            hist.append(this_diff)
            
        final_port.iloc[i] = w_old + alpha_series * diff
        
    return final_port

def last_days_from_val(X: pd.DataFrame) -> pd.DataFrame:
    """Count days since last valid value"""
    arr = X.values
    mask = ~np.isnan(arr)
    # axis=0 (Time) Index Broadcast
    idx_mat = np.broadcast_to(np.arange(len(X))[:, None], arr.shape)
    resets = np.where(mask, idx_mat, -np.inf)
    last_reset = np.maximum.accumulate(resets, axis=0)
    counts = idx_mat - last_reset
    counts[last_reset == -np.inf] = np.nan
    counts[mask] = 0
    return pd.DataFrame(counts, index=X.index, columns=X.columns)

def prev_value(X: pd.DataFrame) -> pd.DataFrame:
    return X.fillna(method='ffill', axis=0).shift(1, axis=0)

def corr_vec(returns_df: pd.DataFrame, d: int, inverse=False):
    """Returns Rolling Correlation (T, N, N)"""
    T = len(returns_df)
    N = len(returns_df.columns)
    res = np.full((T, N, N), np.nan)
    for t in range(d-1, T):
        win = returns_df.iloc[t-d+1 : t+1, :]
        corr = win.corr().values
        if inverse:
            corr = np.nan_to_num(corr)
            corr = safe_pinv(corr)
        res[t] = corr
    return res

def corr_dot_one(returns_df: pd.DataFrame, d: int, inverse=False):
    corrs = corr_vec(returns_df, d, inverse)
    ones = np.ones((returns_df.shape[1], 1))
    res = []
    for t in range(len(corrs)):
        if np.isnan(corrs[t]).all():
            res.append(np.full(returns_df.shape[1], np.nan))
        else:
            res.append((corrs[t] @ ones).flatten())
    return pd.DataFrame(res, index=returns_df.index, columns=returns_df.columns)

# ==============================================================================
# [5] Vector Utils (Dict of DataFrames)
# ==============================================================================

def vec_sum(X: dict) -> pd.DataFrame:
    s = None
    for df in X.values():
        filled = df.fillna(0)
        s = filled if s is None else s + filled
    return s

def vec_count(X: dict) -> pd.DataFrame:
    c = None
    for df in X.values():
        valid = df.notna().astype(int)
        c = valid if c is None else c + valid
    return c.replace(0, np.nan)

def vec_avg(X: dict) -> pd.DataFrame:
    return vec_sum(X) / vec_count(X)

def vec_min(X: dict) -> pd.DataFrame:
    m = None
    for df in X.values():
        filled = df.fillna(np.inf)
        m = filled if m is None else np.minimum(m, filled)
    return pd.DataFrame(m, index=list(X.values())[0].index, columns=list(X.values())[0].columns).replace(np.inf, np.nan)

def vec_max(X: dict) -> pd.DataFrame:
    m = None
    for df in X.values():
        filled = df.fillna(-np.inf)
        m = filled if m is None else np.maximum(m, filled)
    return pd.DataFrame(m, index=list(X.values())[0].index, columns=list(X.values())[0].columns).replace(-np.inf, np.nan)

def vec_backfill(X: dict, d: int) -> dict:
    return {k: ts_backfill(v, d) for k, v in X.items()}

def vec_std_dev(X: dict) -> pd.DataFrame:
    sum_df = None
    sum_sq_df = None
    count_df = None
    for df in X.values():
        filled = df.fillna(0)
        valid = df.notna().astype(int)
        if sum_df is None:
            sum_df = filled.copy()
            sum_sq_df = (filled ** 2).copy()
            count_df = valid.copy()
        else:
            sum_df += filled
            sum_sq_df += (filled ** 2)
            count_df += valid
            
    invalid = count_df < 2
    count_df[invalid] = np.nan
    
    num = sum_sq_df - (sum_df ** 2) / count_df
    var = (num / (count_df - 1)).clip(lower=0)
    return np.sqrt(var)

# ==============================================================================
# [6] Basic Maths & Aliases
# ==============================================================================

def add(X, Y): return X + Y
def subtract(X, Y): return X - Y
def multiply(X, Y): return X * Y
def divide(X, Y): return X / Y

def df_abs(X): return X.abs()
def log(X): return np.log(X)
def sign(X): return np.sign(X)
def signed_power(X, p=2): return np.sign(X) * (X.abs() ** p)

def sigmoid(X): return 1 / (1 + np.exp(-X))
def tanh(X): return np.tanh(X)
def arc_sin(X): return np.arcsin(X)
def arc_cos(X): return np.arccos(X)
def arc_tan(X): return np.arctan(X)

def purify(X): return X.replace([np.inf, -np.inf], np.nan)
def replace(X, d): return X.replace(d)
def df_and(X, Y): return (X.astype(bool) & Y.astype(bool)).astype(int)
def df_or(X, Y): return (X.astype(bool) | Y.astype(bool)).astype(int)
def df_max(X, Y): return np.maximum(X, Y)
def df_min(X, Y): return np.minimum(X, Y)