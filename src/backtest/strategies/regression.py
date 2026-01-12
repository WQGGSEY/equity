import pandas as pd
import numpy as np
from .base import Strategy

class LongOnlySniperStrategy(Strategy):
    """
    [Fast Vectorized Sniper Strategy]
    - 속도 개선: 매일 반복하던 회귀분석을 initialize()에서 Pandas Rolling 연산으로 한 번에 처리
    - 로직 동일: 과거 N일간의 (Feature t-1 -> Return t) 관계를 학습해, 오늘 Feature t로 내일 Return t+1 예측
    """
    def __init__(self, 
                 feature_col='ts2vec_manifold_0', 
                 target_col='FD_Close', 
                 lookback_window=20, 
                 top_n=10, 
                 hold_period=10):
        
        super().__init__(name=f"FastSniper_Hold{hold_period}")
        
        self.feature_col = feature_col
        self.target_col = target_col
        self.lookback_window = lookback_window
        self.top_n = top_n
        self.hold_period = hold_period
        
        self.entry_dates = {}
        self.predictions = None # 미리 계산된 예측값 행렬

    def initialize(self, market_data):
        self.md = market_data
        
        # 1. 데이터 검증
        if self.feature_col not in self.md.features:
            raise ValueError(f"❌ Feature '{self.feature_col}' not found!")
        if self.target_col not in self.md.features:
            raise ValueError(f"❌ Target '{self.target_col}' not found!")
            
        print("⚡ [Vectorization] Pre-computing all regression signals... (Please wait)")

        # 2. 전체 데이터 가져오기
        # X: 오늘의 Feature
        features = self.md.features[self.feature_col]
        # Y: 내일의 수익률 (Target)
        # 회귀분석 학습 데이터는 (X_t-1, Y_t) 쌍이므로,
        # 편의상 '어제 Feature'와 '오늘 Target'의 관계를 구하는 것과 수학적으로 동일함.
        
        # Y (Target): 수익률
        targets = self.md.features[self.target_col]
        
        # X_lag (Predictor for training): 한 박자 전의 Feature
        features_lag = features.shift(1)
        
        # 3. Rolling Regression (베타와 알파 계산)
        # Beta = Cov(X, Y) / Var(X)
        # Alpha = Mean(Y) - Beta * Mean(X)
        
        # 공분산 (Window 기간 동안 X_lag와 Y의 관계)
        cov = targets.rolling(window=self.lookback_window).cov(features_lag)
        # 분산 (Window 기간 동안 X_lag의 움직임)
        var = features_lag.rolling(window=self.lookback_window).var()
        
        # 기울기(Beta) 계산 (분모가 0이거나 NaN이면 결과도 NaN)
        beta = cov / var
        
        # 평균값 (Rolling Mean)
        mean_y = targets.rolling(window=self.lookback_window).mean()
        mean_x = features_lag.rolling(window=self.lookback_window).mean()
        
        # 절편(Alpha) 계산
        alpha = mean_y - (beta * mean_x)
        
        # 4. 최종 예측 (Prediction)
        # 예측값(Expected Return) = Alpha + Beta * (오늘의 Feature)
        # 여기서 '오늘의 Feature'는 shift하지 않은 원본 features 사용
        raw_predictions = alpha + (beta * features)
        self.predictions = raw_predictions.astype(float)
        
        print("   -> ✅ Pre-computation Complete!")

    def on_bar(self, date, universe_tickers, portfolio):
        orders = []
        
        # -----------------------------
        # 1. 매도 로직 (Exit) - 기존과 동일
        # -----------------------------
        current_holdings = list(portfolio.holdings.keys())
        for ticker in current_holdings:
            entry_date = self.entry_dates.get(ticker)
            if entry_date is not None:
                days_held = (date - entry_date).days
                if days_held >= self.hold_period:
                    orders.append({'ticker': ticker, 'action': 'SELL', 'quantity': portfolio.holdings[ticker]})
                    del self.entry_dates[ticker]

        # -----------------------------
        # 2. 매수 로직 (Entry) - 초고속 조회 방식
        # -----------------------------
        # 이미 initialize에서 계산한 값에서 '오늘 날짜' 행만 쏙 빼옴
        try:
            # (Vectorized) 오늘 날짜의 모든 종목 예측값 가져오기
            daily_preds = self.predictions.loc[date] 
        except KeyError:
            return orders

        # 유니버스에 포함되고, 데이터가 유효한 종목만 필터링
        candidates = []
        
        # Pandas Series 연산이므로 매우 빠름
        # 1) 유니버스 필터
        valid_preds = daily_preds.reindex(universe_tickers)
        # 2) NaN 제거 및 양수 조건
        valid_preds = valid_preds[valid_preds > 0].dropna()
        
        # 후보가 없다면 종료
        if valid_preds.empty:
            return orders
            
        # 3) 상위 N개 추출 (Top N)
        top_picks = valid_preds.nlargest(self.top_n).index.tolist()
        
        # 4) 주문 생성
        if top_picks and portfolio.cash > 0:
            target_amt = portfolio.cash / len(top_picks)
            current_prices = self.md.prices['Close'].loc[date]
            
            for ticker in top_picks:
                if ticker in portfolio.holdings: continue # 이미 보유중이면 패스
                
                price = current_prices.get(ticker, 0)
                if price > 0:
                    qty = int(target_amt / price)
                    if qty > 0:
                        orders.append({'ticker': ticker, 'action': 'BUY', 'quantity': qty})
                        self.entry_dates[ticker] = date
                        
        return orders