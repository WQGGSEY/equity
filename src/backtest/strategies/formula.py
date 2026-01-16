import pandas as pd
from .base import Strategy
from src.alpha.parser import AlphaParser

class FormulaStrategy(Strategy):
    def __init__(self, expressions, top_n=20, **kwargs): # expressions 타입 힌트 제거
        super().__init__(**kwargs)
        
        # [수정] 만약 문자열 하나만 들어오면 리스트로 감싸줌 (방어 코드)
        if isinstance(expressions, str):
            self.expressions = [expressions]
        else:
            self.expressions = expressions
            
        self.top_n = top_n
        self.parser = AlphaParser()
        self.signal_matrix = None

    def initialize(self, market_data):
        print("🧪 [Alpha Engine] Initializing Data Context...")
        
        # 1. 기본 가격 데이터 (Standard Context)
        # 키를 소문자로도 접근 가능하게 설정 (Close -> close)
        data_context = {
            'Open': market_data.prices['Open'],
            'High': market_data.prices['High'],
            'Low': market_data.prices['Low'],
            'Close': market_data.prices['Close'],
            'Volume': market_data.prices['Volume'],
            'Amount': market_data.prices['Amount'],
        }
        
        # 2. [핵심] 파생 피처 동적 주입 (Dynamic Injection)
        # Loader가 읽어온 모든 features(RSI, MA_20, ts2vec 등)를 변수로 등록
        if hasattr(market_data, 'features'):
            for feat_name, feat_df in market_data.features.items():
                data_context[feat_name] = feat_df
                # 편의를 위해 소문자 이름도 허용 (예: 'RSI_14' -> 'rsi_14')
                # (단, 이름 충돌 주의)
                if feat_name.lower() not in data_context:
                    data_context[feat_name.lower()] = feat_df

        # 3. 소문자/대문자 호환성 (기본 가격)
        # 이미 위에서 넣었지만 확실하게 처리
        basic_keys = list(data_context.keys())
        for k in basic_keys:
            if k.lower() not in data_context:
                data_context[k.lower()] = data_context[k]

        # 디버깅: 사용 가능한 변수 목록 출력
        available_vars = sorted(list(data_context.keys()))
        print(f"   -> Available Variables: {available_vars[:10]} ... (Total {len(available_vars)})")

        # 4. 수식 계산
        final_signal = pd.DataFrame(0.0, index=market_data.dates, columns=market_data.tickers)
        
        for expr in self.expressions:
            print(f"   -> Calculating: {expr}")
            try:
                # 이제 여기서 'RSI_14', 'ts2vec_0' 등을 바로 쓸 수 있음!
                alpha_val = self.parser.parse(expr, data_context)
                
                # 결과 누적 (단, NaN은 0으로 처리하거나 전략에 따라 다름)
                final_signal = final_signal.add(alpha_val, fill_value=0)
            except Exception as e:
                print(f"   🚨 Error in Expression: {expr}")
                raise e
            
        self.signal_matrix = final_signal
        print("✅ Signal Matrix Computed.")

    def on_bar(self, date, valid_tickers, portfolio):
        # (기존과 동일)
        if date not in self.signal_matrix.index:
            return []
            
        daily_scores = self.signal_matrix.loc[date]
        valid_scores = daily_scores[valid_tickers].dropna()
        
        if valid_scores.empty: return []

        top_stocks = valid_scores.nlargest(self.top_n).index.tolist()
        target_weight = 1.0 / len(top_stocks) if top_stocks else 0
        orders = []
        
        for ticker in top_stocks:
            price = self.get_price(date, ticker)
            if price <= 0: continue
            
            target_val = portfolio.equity() * target_weight
            current_qty = portfolio.holdings.get(ticker, 0)
            target_qty = int(target_val / price)
            
            diff = target_qty - current_qty
            if diff != 0:
                action = 'BUY' if diff > 0 else 'SELL'
                orders.append({'ticker': ticker, 'action': action, 'quantity': abs(diff)})
                
        return orders