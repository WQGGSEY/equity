# 📈 Equity Quantitative Research Platform

이 프로젝트는 대량의 미국 주식 데이터를 수집, 정제하여 금융 특화 Feature(Dollar Bar, FracDiff, Embeddings)를 생성하고, 이를 기반으로 다양한 퀀트 전략을 검증(Backtest)할 수 있는 통합 연구 플랫폼입니다.

---

## 📂 프로젝트 구조 (Directory Structure)

```bash
equity/
├── data/                  # 데이터 저장소 (Bronze -> Silver -> Gold -> Platinum)
├── configs/               # 백테스트 및 전략 설정 파일 (.yaml)
├── scripts/               # 실행 가능한 파이프라인 스크립트 (00~07)
├── src/
│   ├── pipeline/          # 데이터 ETL 파이프라인 로직
│   ├── features/          # Feature Engineering (Dollar Bar, Tech indicators 등)
│   ├── models/            # 딥러닝 모델 아키텍처 (TS2Vec 등)
│   ├── backtest/          # 백테스트 엔진 및 전략 클래스
│   └── utils/             # 유틸리티 함수
└── requirements.txt       # 의존성 패키지 목록

```

---

## 🚀 1. 데이터 파이프라인 (Data Pipeline)

Raw Data(Yahoo Finance)에서 학습 가능한 Platinum 데이터까지 이어지는 6단계 ETL 프로세스입니다.

| 단계 | 스크립트 | 설명 | 관련 코드 |
| --- | --- | --- | --- |
| **00** | `scripts/00_train_global_model.py` | **Global Model 학습**: 전체 데이터를 이용하여 Contrastive Learning 모델(TS2Vec)을 학습하고 가중치(.pth)를 저장합니다. |  |
| **01** | `scripts/01_define_universe.py` | **유니버스 정의**: 로컬 파일, SEC, NASDAQ 등에서 티커를 수집하여 `master_ticker_list.csv`를 생성/갱신합니다. |  |
| **02** | `scripts/02_data_download...py` | **Bronze 생성**: Master List를 기반으로 Yahoo Finance에서 일별 주가 데이터를 수집합니다. (Raw Data) |  |
| **03** | `scripts/03_create_silver.py` | **Silver 생성**: Outlier 제거, 결측치 처리 등 기본적인 정제를 수행합니다. |  |
| **04** | `scripts/04_create_gold.py` | **Gold 생성**: 티커 변경/합병 이슈 처리, 유사 종목 병합(Deduplication), 동전주 필터링을 수행합니다. |  |
| **05** | `scripts/05_quarantine_gold.py` | **검증(Quarantine)**: Gold 데이터의 무결성(데이터 길이, 가격 0 존재 여부 등)을 최종 확인하고 불량 데이터를 격리합니다. |  |
| **06** | `scripts/06_create_platinum.py` | **Platinum 생성**: 최종 학습용 Feature(Dollar Bar, FracDiff, Embeddings 등)를 생성합니다. |  |

### 🛠 Feature Engineering

`src/features/` 내부의 클래스들을 통해 Platinum 데이터가 생성됩니다.

* **Preprocessing**: `DollarBarStationaryFeature`를 통해 시간 기준이 아닌 거래대금 기준 바(Dollar Bar)를 생성하고, 분수 차분(FracDiff)을 적용하여 정상성(Stationarity)을 확보합니다.
* **Contrastive Learning**: `Contrastive_OC_HL` 클래스는 00번 스크립트에서 학습된 모델을 로드하여 시장의 내재적 표현(Embedding)을 추출합니다.

---

## 📊 2. 백테스트 구조 (Backtest Framework)

백테스트는 `src/backtest/` 모듈에 의해 구동되며 설정 파일(`yaml`)을 통해 제어됩니다.

### 핵심 컴포넌트

1. **Engine (`src/backtest/engine.py`)**:
* 전략 실행, 매매 체결(Match trades), PnL 계산, 결과 집계를 담당하는 핵심 엔진입니다.


2. **Loader (`src/backtest/loader.py`)**:
* `configs/base.yaml` 등에 정의된 시나리오에 따라 Platinum 데이터를 로드하고 Train/Test 기간을 분할합니다.


3. **Strategy (`src/backtest/strategies/`)**:
* 모든 전략은 `BaseStrategy`를 상속받아야 하며, 매수/매도 신호를 생성하는 로직을 담습니다.



### 실행 방법

```bash
# 특정 전략 설정 파일을 지정하여 백테스트 실행
python scripts/07_backtest.py --config configs/strategies/golden_cross_v1.yaml

```

---

## 💡 3. 신규 전략 개발 가이드 (How to Create a New Strategy)

새로운 전략(예: RSI 전략)을 추가하는 방법은 다음과 같습니다.

### Step 1: 전략 클래스 구현

`src/backtest/strategies/market_buy.py` 파일을 생성하고 코드를 작성합니다.

```python
class FDRebalanceStrategy(Strategy):
    """
    [FD Based Daily Rebalancing Strategy] (Fixed Version)
    """
    def __init__(self, top_n=10, ascending=False, feature_name='FD_TrdAmount'):
        super().__init__(name=f"FD_Rebalance_Top{top_n}")
        self.top_n = top_n
        self.ascending = ascending
        self.feature_name = feature_name
        self.md = None

    def initialize(self, market_data):
        self.md = market_data
        if self.feature_name not in self.md.features:
            available = list(self.md.features.keys())
            raise ValueError(f"❌ Feature '{self.feature_name}' not found in MarketData! Available: {available}")
        print(f"⚖️ [FD Rebalance] initialized. Target Feature: {self.feature_name}, Top: {self.top_n}")

    def on_bar(self, date, universe_tickers, portfolio):
        orders = []
        current_prices = self.md.prices['Close'].loc[date]
        
        # 1. Feature 데이터 가져오기
        try:
            feature_vals = self.md.features[self.feature_name].loc[date]
        except KeyError:
            return []

        # 2. 유효 종목 필터링 및 랭킹 산출
        valid_candidates = []
        for t in universe_tickers:
            val = feature_vals.get(t, np.nan)
            price = current_prices.get(t, np.nan)
            
            # 가격과 피처 값이 모두 유효한 경우만 후보 등록
            if not np.isnan(val) and not np.isnan(price) and price > 0:
                valid_candidates.append((t, val))
        
        if not valid_candidates:
            return []

        # 정렬
        valid_candidates.sort(key=lambda x: x[1], reverse=not self.ascending)
        top_picks = [x[0] for x in valid_candidates[:self.top_n]]
        
        # 3. 목표 수량 계산 (Total Equity 기준 1/N)
        total_equity = portfolio.cash
        for t, qty in portfolio.holdings.items():
            price = current_prices.get(t, np.nan)
            # [수정] 보유 종목의 가격이 NaN이면 0으로 처리하여 전체 자산 가치 오염 방지
            if pd.isna(price) or price <= 0:
                price = 0
            total_equity += qty * price
            
        target_amt_per_stock = total_equity / len(top_picks) if top_picks else 0
        
        # [안전장치] 만약 자산 계산이 잘못되어 NaN이나 음수가 나오면 매매 중단
        if pd.isna(target_amt_per_stock) or target_amt_per_stock <= 0:
            return []
        
        # 4. 주문 생성
        
        # (A) 매도 주문
        current_holdings = list(portfolio.holdings.keys())
        for t in current_holdings:
            qty = portfolio.holdings[t]
            price = current_prices.get(t, np.nan)
            
            # 가격 정보를 알 수 없으면 일단 매도 보류 (또는 시장가 강제 매도 고려 가능)
            if pd.isna(price) or price <= 0:
                continue

            if t not in top_picks:
                orders.append({'ticker': t, 'action': 'SELL', 'quantity': qty})
            else:
                # 리밸런싱 (비중 축소)
                target_qty = int(target_amt_per_stock / price)
                diff = target_qty - qty
                if diff < 0:
                    orders.append({'ticker': t, 'action': 'SELL', 'quantity': abs(diff)})
        
        # (B) 매수 주문
        for t in top_picks:
            price = current_prices.get(t, np.nan)
            
            # [수정] 가격 안전장치
            if pd.isna(price) or price <= 0:
                continue
                
            target_qty = int(target_amt_per_stock / price)
            current_qty = portfolio.holdings.get(t, 0)
            diff = target_qty - current_qty
            
            if diff > 0:
                orders.append({'ticker': t, 'action': 'BUY', 'quantity': diff})
                
        return orders

```

### Step 2: 설정 파일 생성

`configs/strategies/market_buy_v1.yaml` 파일을 생성합니다.

```yaml
# configs/strategies/rsi_v1.yaml
base_config: "configs/base.yaml"

experiment_name: "FD_TrdAmount_Rebalance_Base"
strategy:
  module: "src.backtest.strategies.market_buy"
  class: "FDRebalanceStrategy"
  params:
    top_n: 10
    ascending: false        # False: FD_TrdAmount가 큰 순서대로 (True면 작은 순서대로)
    feature_name: "TrdAmount"
```

### Step 3: 실행

```bash
python scripts/07_backtest.py --config configs/strategies/market_buy_v1.yaml

```

---

## 🔧 환경 설정 (Setup)

```bash
# 1. 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows

# 2. 패키지 설치
pip install -r requirements.txt

```

---

## 📋 4. 데이터 사전 (Data Dictionary)

`scripts/06_create_platinum.py` 실행 후 생성되는 `data/platinum/features/{ticker}.parquet` 파일의 컬럼 명세입니다. 이 파일 하나에 OHLCV, 전처리된 피처, 그룹 정보, 모델 임베딩이 모두 포함되어 있습니다.

| Column Name | Source Module | Defined Class | Description |
| --- | --- | --- | --- |
| **Open, High, Low, Close** | Raw Data | - | Yahoo Finance에서 수집한 시가, 고가, 저가, 종가 (Adjusted Price 반영) |
| **Volume** | Raw Data | - | 거래량 |
| **FD_Open, FD_High, FD_Low, FD_Close** | `src/features/preprocessors.py` | `DollarBarStationaryFeature` | 시간 기준이 아닌 **Dollar Bar(거래대금)** 기준으로 샘플링한 후, **분별 차분(FracDiff)**을 적용하여 정상성(Stationarity)을 확보한 가격 데이터 |
| **grp_sector** | `src/features/groups.py` | `SectorGroup` | 주요 섹터 ETF(XLK, XLF 등)와의 수익률 상관계수를 기반으로 매일 동적으로 할당된 **섹터 그룹 ID** (0~10) |
| **grp_liquidity** | `src/features/groups.py` | `LiquidityGroup` | 전체 시장 내 거래대금(Dollar Volume) 순위를 기준으로 나눈 **유동성 등급** (0: 하위 ~ 9: 상위) |
| **ts2vec_manifold_0, ts2vec_manifold_1, ts2vec_manifold_2** | `src/features/contrastive.py` | `Contrastive_OC_HL` | **TS2Vec** 모델이 학습한 고차원 시장 내재 표현을 Micro-Autoencoder를 통해 압축한 **저차원 Manifold 좌표**. (유사한 가격 패턴을 가진 종목은 이 좌표상에서 가깝게 위치함) |