현재 코드베이스(Data Pipeline, Feature Engineering, Backtesting Framework)를 기반으로 작성된 상세한 `README.md`입니다.

---

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

`src/backtest/strategies/rsi_strategy.py` 파일을 생성하고 코드를 작성합니다.

```python
import pandas as pd
import numpy as np
from src.backtest.strategies.base import BaseStrategy

class RSIStrategy(BaseStrategy):
    """
    RSI 기반 역추세 전략 예시
    """
    def __init__(self, rsi_period=14, buy_threshold=30, sell_threshold=70):
        super().__init__()
        self.rsi_period = rsi_period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Platinum 데이터(df)를 받아 'signal' 컬럼(1: 매수, -1: 매도, 0: 관망)을 반환
        """
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        
        # 종가 기준 RSI 계산 (예시 로직)
        close = df['Close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 신호 생성
        signals.loc[rsi < self.buy_threshold, 'signal'] = 1  # 과매도 구간 매수
        signals.loc[rsi > self.sell_threshold, 'signal'] = -1 # 과매수 구간 매도
        
        return signals

```

### Step 2: 설정 파일 생성

`configs/strategies/rsi_v1.yaml` 파일을 생성합니다.

```yaml
# configs/strategies/rsi_v1.yaml
defaults:
  - base  # configs/base.yaml 상속

strategy:
  name: "RSI_Reversal_V1"
  class: "src.backtest.strategies.rsi_strategy.RSIStrategy"  # 클래스 경로 지정
  params:
    rsi_period: 14
    buy_threshold: 30
    sell_threshold: 70

backtest:
  start_date: "2020-01-01"
  end_date: "2023-12-31"
  initial_capital: 10000.0

```

### Step 3: 실행

```bash
python scripts/07_backtest.py --config configs/strategies/rsi_v1.yaml

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