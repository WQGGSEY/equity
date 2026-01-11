import pandas as pd
import numpy as np
try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    raise ImportError("❌ scikit-learn이 설치되지 않았습니다. 'pip install scikit-learn'을 실행하세요.")

# [수정 1] 올바른 상속
from .base import Strategy

class LongOnlySniperStrategy(Strategy):
    """
    [Long Only Sniper Strategy]
    - Entry: 과거 N일(lookback) 데이터를 학습해 기대 수익률 상위 종목 매수
    - Exit: 설정한 보유 기간(hold_period)이 지나면 전량 매도 (Time-Cut)
    """
    def __init__(self, 
                 feature_col='ts2vec_manifold_0', 
                 target_col='FD_Close', 
                 lookback_window=20, 
                 top_n=10, 
                 hold_period=10): # [복구] 사용자님이 지정한 보유 기간
        
        super().__init__(name=f"LongOnlySniper_Hold{hold_period}")
        
        self.feature_col = feature_col
        self.target_col = target_col
        self.lookback_window = lookback_window
        self.top_n = top_n
        self.hold_period = hold_period
        
        # 모델 객체 (매일 재학습)
        self.model = LinearRegression()
        
        # [복구] 진입 시점 기록용 딕셔너리 {ticker: entry_date}
        self.entry_dates = {}
        
        # 데이터 캐싱
        self.feat_data = None
        self.target_data = None

    def initialize(self, market_data):
        self.md = market_data
        
        # Feature 존재 확인
        if self.feature_col not in self.md.features:
            raise ValueError(f"❌ Feature '{self.feature_col}' not found!")
        if self.target_col not in self.md.features:
            raise ValueError(f"❌ Target '{self.target_col}' not found!")
            
        # 데이터 로딩
        self.feat_data = self.md.features[self.feature_col]
        self.target_data = self.md.features[self.target_col]
        
        print(f"✅ Strategy Initialized: Window={self.lookback_window}, Hold={self.hold_period}")

    def on_bar(self, date, universe_tickers, portfolio):
        orders = []
        
        # -----------------------------
        # 1. 매도 로직 (Exit Logic) - [복구됨]
        # -----------------------------
        # 현재 보유 중인 종목들을 검사하여 보유 기간이 만료되었는지 확인
        current_holdings = list(portfolio.holdings.keys())
        
        for ticker in current_holdings:
            # 진입 날짜 확인
            entry_date = self.entry_dates.get(ticker)
            
            # 진입 기록이 없으면(예외 상황) 그냥 둠, 기록이 있으면 기간 계산
            if entry_date is not None:
                # 경과 일수 계산 (거래일 기준이 아니라 실제 날짜 차이로 계산됨. 필요시 인덱스 차이로 변경 가능)
                # 여기서는 간단히 pd.Timestamp 간의 차이(.days)를 사용
                days_held = (date - entry_date).days
                
                if days_held >= self.hold_period:
                    # 보유 기간 만료 -> 매도 주문
                    qty = portfolio.holdings[ticker]
                    orders.append({'ticker': ticker, 'action': 'SELL', 'quantity': qty})
                    
                    # 기록 삭제
                    del self.entry_dates[ticker]

        # -----------------------------
        # 2. 매수 로직 (Entry Logic)
        # -----------------------------
        # 데이터 인덱싱
        try:
            date_idx = self.md.dates.get_loc(date)
            if date_idx < self.lookback_window:
                return orders # 데이터 부족
                
            start_idx = date_idx - self.lookback_window
            end_idx = date_idx
        except KeyError:
            return orders

        predictions = []

        # 유니버스 종목 순회
        for ticker in universe_tickers:
            # 이미 보유 중이면 추가 매수 안 함 (Sniper 스타일)
            if ticker in portfolio.holdings:
                continue

            try:
                # 데이터 추출
                X_series = self.feat_data[ticker].iloc[start_idx:end_idx].values
                y_series = self.target_data[ticker].iloc[start_idx:end_idx].values
                X_today = self.feat_data[ticker].iloc[date_idx]
                
                # 결측치 체크
                if np.isnan(X_today) or np.isnan(X_series).any() or np.isnan(y_series).any():
                    continue
                
                # 학습 데이터 구성 (Shift 적용: t-1 Feature -> t Target)
                X_train = X_series[:-1].reshape(-1, 1)
                y_train = y_series[1:] 
                
                if len(y_train) < 10: continue 

                # 모델 학습 및 예측
                self.model.fit(X_train, y_train)
                pred = self.model.predict([[X_today]])[0]
                
                # 예측 수익률이 양수일 때만 후보 등록
                if pred > 0:
                    predictions.append((ticker, pred))
                    
            except Exception:
                continue

        # Top N 선정
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_picks = predictions[:self.top_n]
        
        if top_picks:
            # 자금 배분: 현재 현금 / (타겟 종목 수 - 현재 보유 종목 수)? 
            # 혹은 단순하게 (현재 현금 / 선택된 종목 수)로 배분
            if portfolio.cash > 0:
                target_amt = portfolio.cash / len(top_picks)
                current_prices = self.md.prices['Close'].loc[date]
                
                for ticker, score in top_picks:
                    price = current_prices.get(ticker, 0)
                    if price > 0:
                        qty = int(target_amt / price)
                        if qty > 0:
                            orders.append({'ticker': ticker, 'action': 'BUY', 'quantity': qty})
                            # [중요] 진입 시점 기록
                            self.entry_dates[ticker] = date
                        
        return orders