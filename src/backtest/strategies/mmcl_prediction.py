import numpy as np
import pandas as pd
from .base import Strategy

class MMCL_Prediction_V1(Strategy):
    def __init__(self, z_threshold=3.0, max_pos=5, hold_days=5, train_window=100, alpha=1.0):
        super().__init__(name="MMCL_Prediction_V1")
        self.z_threshold = z_threshold
        self.max_pos = max_pos
        self.hold_days = hold_days
        self.train_window = train_window
        self.alpha = alpha  # Ridge alpha
        
        # 내부 상태
        self.entry_info = {} 
        self.last_train_idx = -1
        
        # [Optimization] Numpy Tensors
        self.X_tensor = None  # (Time, Ticker, Feature)
        self.Y_matrix = None  # (Time, Ticker) - Next Day Return
        self.P_matrix = None  # (Time, Ticker) - Price
        
        # Mappings
        self.date_to_idx = {}
        self.ticker_to_idx = {}
        self.idx_to_ticker = {}
        
        # Pre-allocated memory for Ridge weights
        self.weights = None # Current Model Weights

    def initialize(self, market_data):
        print("⚡ [Fast Strategy] Converting Data to Numpy Tensors...")
        self.md = market_data
        
        # 1. Index Mappings
        self.date_to_idx = {d: i for i, d in enumerate(self.md.dates)}
        self.ticker_to_idx = {t: i for i, t in enumerate(self.md.tickers)}
        self.idx_to_ticker = {i: t for t, i in self.ticker_to_idx.items()}
        
        n_dates = len(self.md.dates)
        n_tickers = len(self.md.tickers)
        
        # 2. Feature Selection & Tensor Construction
        feat_keys = [k for k in self.md.features.keys() if 'ts2vec_manifold' in k]
        n_feats = len(feat_keys)
        
        # (Time, Ticker, Feat) - 3D Array
        # 메모리 효율을 위해 float32 사용
        self.X_tensor = np.zeros((n_dates, n_tickers, n_feats), dtype=np.float32)
        
        for f_i, key in enumerate(feat_keys):
            # DataFrame -> Numpy (자동으로 Index/Column 정렬 안됨에 주의)
            # 여기서는 MarketData가 이미 reindex(dates) 되어있다고 가정
            # 컬럼(Ticker) 순서를 맞춰야 함
            df = self.md.features[key][self.md.tickers] # Ticker 순서 강제 정렬
            self.X_tensor[:, :, f_i] = df.values.astype(np.float32)
            
        # 3. Price & Target Matrix
        # Price: (Time, Ticker)
        self.P_matrix = self.md.prices['Close'][self.md.tickers].values.astype(np.float32)
        
        # Target: Next Day Return (Shifted -1)
        # today's features -> tomorrow's return
        returns = self.md.prices['Close'][self.md.tickers].pct_change().shift(-1).fillna(0)
        self.Y_matrix = returns.values.astype(np.float32)
        
        print(f"   -> Tensor Shape: {self.X_tensor.shape} (Dates, Tickers, Feats)")

    def on_bar(self, date, universe_tickers, portfolio):
        orders = []
        
        # 1. Fast Index Lookup (Pandas 검색 제거)
        curr_t = self.date_to_idx.get(date)
        if curr_t is None: return []
        
        # 2. Daily Retraining (Pure Numpy Ridge)
        # 매일 수행하지만 Numpy 연산이라 빠름
        self._fit_fast_ridge(curr_t)
        
        if self.weights is None: return []

        # 3. Fast Prediction (Dot Product)
        # X_today: (N_tickers, N_feats)
        X_today = self.X_tensor[curr_t] # Slicing (No Copy)
        
        # Raw Scores: (N_tickers,)
        raw_scores = X_today @ self.weights
        
        # 4. Filter Universe & Compute Z-Score
        # 유니버스 마스킹을 인덱스로 변환
        univ_indices = [self.ticker_to_idx[t] for t in universe_tickers if t in self.ticker_to_idx]
        if not univ_indices: return []
        
        # 유니버스 내에서의 통계 계산
        univ_scores = raw_scores[univ_indices]
        
        # Z-Score Normalization
        mu = np.mean(univ_scores)
        sigma = np.std(univ_scores) + 1e-9
        
        # 전체 종목에 대한 Z-Score (Vectorized)
        all_z_scores = (raw_scores - mu) / sigma
        
        # 5. Logic Execution (기존 로직과 동일하지만 Numpy 값 사용)
        # ... (이하 로직은 Python dict 조작이므로 병목 아님, 그대로 둠)
        
        # [Optimized Entry/Exit Loop]
        # 필요한 종목만 순회하기 위해 Z-Score가 높은 인덱스 찾기
        
        # A. Exit Logic
        # (기존 코드의 portfolio.holdings loop 유지)
        for ticker in list(portfolio.holdings.keys()):
            should_sell = False
            t_idx = self.ticker_to_idx.get(ticker)
            if t_idx is None: continue
            
            # Time Stop
            info = self.entry_info.get(ticker)
            if info:
                days_held = (date - info['date']).days
                if days_held >= self.hold_days:
                    should_sell = True
            
            # Signal Stop
            if not should_sell:
                z = all_z_scores[t_idx]
                if z < 1.0: should_sell = True
            
            if should_sell:
                qty = portfolio.holdings[ticker]
                orders.append({'ticker': ticker, 'action': 'SELL', 'quantity': qty})
                if ticker in self.entry_info: del self.entry_info[ticker]
                
        # B. Entry Logic
        # Numpy argsort로 상위 N개 추출 (빠름)
        # threshold 넘는 것만 고려
        candidate_indices = np.where(all_z_scores > self.z_threshold)[0]
        
        if len(candidate_indices) > 0:
            # 그 중에서 점수 높은 순 정렬
            # (candidate_indices 내에서 정렬)
            sorted_local_idx = np.argsort(all_z_scores[candidate_indices])[::-1][:self.max_pos]
            target_indices = candidate_indices[sorted_local_idx]
            
            real_targets = []
            for t_idx in target_indices:
                ticker = self.idx_to_ticker[t_idx]
                price = self.P_matrix[curr_t, t_idx]
                
                if ticker not in portfolio.holdings and price > 0:
                    real_targets.append((ticker, price, all_z_scores[t_idx]))
            
            if real_targets:
                target_weight = 1.0 / self.max_pos
                equity = portfolio.equity(self.md.prices['Close'].loc[date]) # 여긴 Pandas 써도 됨 (가끔 호출)
                target_amt = equity * target_weight
                
                for ticker, price, z_val in real_targets:
                    if portfolio.cash < target_amt: break
                    qty = int(target_amt / price)
                    if qty > 0:
                        orders.append({'ticker': ticker, 'action': 'BUY', 'quantity': qty})
                        self.entry_info[ticker] = {'date': date}

        return orders

    def _fit_fast_ridge(self, curr_t):
        """
        [Ultra Fast Ridge Training using Numpy]
        Solves: w = (X^T X + alpha I)^-1 X^T y
        """
        # Window Slicing (No Copy, just View)
        start_t = max(0, curr_t - self.train_window)
        end_t = curr_t
        
        if start_t >= end_t: return

        # 1. Flatten Data (Batch Training)
        # X_window: (Window, N_tickers, N_feats) -> (N_samples, N_feats)
        # Y_window: (Window, N_tickers) -> (N_samples,)
        
        X_slice = self.X_tensor[start_t:end_t]
        Y_slice = self.Y_matrix[start_t:end_t]
        
        # Reshape (여기서 Copy 발생하지만 훨씬 가벼움)
        X_train = X_slice.reshape(-1, X_slice.shape[-1])
        y_train = Y_slice.flatten()
        
        # 2. NaN Removal (Boolean Masking is fast in Numpy)
        # np.isnan은 float32에서 매우 빠름
        mask = ~np.isnan(y_train) & ~np.isnan(X_train).any(axis=1)
        X_train = X_train[mask]
        y_train = y_train[mask]
        
        # Outlier Clipping
        y_train = np.clip(y_train, -0.3, 0.3)
        
        if len(y_train) < 100: return

        # 3. Closed-form Ridge Solution (Cholesky Solve)
        # sklearn.Ridge보다 훨씬 빠름
        n_features = X_train.shape[1]
        
        # A = X^T X + alpha * I
        XT = X_train.T
        A = XT @ X_train
        A.flat[::n_features + 1] += self.alpha # Add diagonal (alpha)
        
        # b = X^T y
        b = XT @ y_train
        
        # Solve Ax = b
        try:
            # linalg.solve가 inv보다 빠르고 안정적임
            self.weights = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.weights = None