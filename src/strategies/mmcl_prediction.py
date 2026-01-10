import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from .base import Strategy

class MMCL_Prediction(Strategy):
    """
    [N-Body Sniper Strategy]
    Added Feature: Stop Loss Logic
    """
    def __init__(self, z_threshold=3.0, max_pos=5, hold_days=5, stop_loss=0.1, train_window=250):
        super().__init__(name="MMCL_Prediction")
        self.z_threshold = z_threshold
        self.max_pos = max_pos
        self.hold_days = hold_days
        self.stop_loss = stop_loss      # [NEW] 손절 기준 (0.1 = -10%)
        self.train_window = train_window
        
        # {ticker: {'date': date, 'entry_z': z, 'entry_price': price}}
        self.entry_info = {} 
        self.model = Ridge(alpha=1.0)
        self.last_train_month = -1

    def on_bar(self, date, universe_tickers, portfolio):
        orders = []
        current_prices = self.md.prices['Close'].loc[date]
        
        # --- 1. Train Model (Monthly Update) ---
        if date.month != self.last_train_month:
            self._train_model(date)
            self.last_train_month = date.month
            
        # --- 2. Calculate Scores & Z-Scores ---
        candidates = list(set(universe_tickers) | set(portfolio.holdings.keys()))
        features_df = self._get_features(date, candidates)
        
        z_scores = pd.Series(dtype=float)
        
        if not features_df.empty:
            X_pred = features_df.astype(np.float32).fillna(0).values
            raw_scores = self.model.predict(X_pred)
            
            # Z-Score Normalization
            mu = raw_scores.mean()
            sigma = raw_scores.std() + 1e-9
            z_values = (raw_scores - mu) / sigma
            z_scores = pd.Series(z_values, index=features_df.index)
        
        # --- 3. Exit Logic (Time / Signal / Stop Loss) ---
        for ticker in list(portfolio.holdings.keys()):
            should_sell = False
            curr_price = current_prices.get(ticker, np.nan)
            
            info = self.entry_info.get(ticker)
            if info:
                # A. Time Stop
                days_held = (date - info['date']).days
                if days_held >= self.hold_days:
                    should_sell = True
                
                # B. Stop Loss [NEW]
                # 현재 가격이 있고, 진입 가격 정보가 있을 때만 계산
                if not should_sell and not np.isnan(curr_price):
                    entry_price = info.get('entry_price')
                    if entry_price and entry_price > 0:
                        ret = (curr_price / entry_price) - 1
                        if ret < -self.stop_loss:
                            should_sell = True
                            # (선택) 로그 찍기
                            # print(f"  📉 Stop Loss Triggered for {ticker}: {ret*100:.2f}%")

            # C. Signal Weakness Cut
            if not should_sell:
                curr_z = z_scores.get(ticker, -999)
                if curr_z < 1.0:
                    should_sell = True
            
            # Execute Sell
            if should_sell:
                qty = portfolio.holdings[ticker]
                orders.append({'ticker': ticker, 'action': 'SELL', 'quantity': qty})
                if ticker in self.entry_info: del self.entry_info[ticker]

        # --- 4. Sniper Entry Logic ---
        if not z_scores.empty:
            strong_signals = z_scores[z_scores > self.z_threshold]
            targets = strong_signals.nlargest(self.max_pos).index.tolist()
            
            real_targets = []
            for t in targets:
                p = current_prices.get(t, np.nan)
                if t not in portfolio.holdings and not np.isnan(p) and p > 0:
                    real_targets.append(t)
            
            if real_targets:
                target_weight = 1.0 / self.max_pos
                target_amt = portfolio.equity(current_prices) * target_weight
                
                for t in real_targets:
                    if portfolio.cash < target_amt: break 
                    
                    price = current_prices[t]
                    qty = int(target_amt / price)
                    if qty > 0:
                        orders.append({'ticker': t, 'action': 'BUY', 'quantity': qty})
                        
                        # [NEW] 진입 가격(entry_price) 저장
                        self.entry_info[t] = {
                            'date': date, 
                            'entry_z': z_scores[t],
                            'entry_price': price 
                        }

        return orders

    def _get_features(self, date, tickers):
        data = {}
        feat_keys = [k for k in self.md.features.keys() if 'ts2vec_manifold' in k]
        for k in feat_keys:
            try:
                data[k] = self.md.features[k].loc[date].reindex(tickers)
            except:
                pass
        return pd.DataFrame(data)

    def _train_model(self, current_date):
        idx = self.md.dates.index(current_date)
        start_idx = max(0, idx - self.train_window - 1)
        train_dates = self.md.dates[start_idx : idx]
        
        feat_keys = [k for k in self.md.features.keys() if 'ts2vec_manifold' in k]
        X_list, y_list = [], []
        
        for d in train_dates:
            try:
                next_d_idx = self.md.dates.index(d) + 1
                if next_d_idx >= len(self.md.dates): continue
                next_d = self.md.dates[next_d_idx]
                
                p_curr = self.md.prices['Close'].loc[d]
                p_next = self.md.prices['Close'].loc[next_d]
                ret = (p_next / p_curr) - 1
                
                x_data = pd.DataFrame({k: self.md.features[k].loc[d] for k in feat_keys})
                
                valid_mask = ~np.isnan(ret) & ~x_data.isna().any(axis=1)
                
                if valid_mask.sum() > 0:
                    ret_clip = np.clip(ret[valid_mask].values, -0.3, 0.3)
                    X_list.append(x_data[valid_mask].values)
                    y_list.append(ret_clip)
            except:
                continue
                
        if X_list:
            X_train = np.concatenate(X_list, axis=0)
            y_train = np.concatenate(y_list, axis=0)
            self.model.fit(X_train, y_train)