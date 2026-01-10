import sys
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import PLATINUM_FEATURES_DIR
from src.backtest.loader import MarketData
from src.backtest.engine import BacktestEngine
from src.strategies.mmcl_prediction import MMCL_Prediction

def main():
    # 1. Load Data
    md = MarketData(PLATINUM_FEATURES_DIR)
    md.load_all()
    
    # 2. Setup Engine
    engine = BacktestEngine(md, start_date="2010-01-01", end_date="2015-01-01")
    
    # 3. Setup Sniper Strategy
    # - z_threshold=3.0: 3시그마(상위 0.13%) 이상의 초강력 시그널일 때만 매수
    # - hold_days=10: 최대 10일 보유 (Hit-and-Run)
    # - max_pos=5: 최대 5종목 분산
    strategy = MMCL_Prediction(
        z_threshold=4.0, 
        max_pos=2, 
        hold_days=5, 
        train_window=250,
        stop_loss=0.3
    )
    
    print(f"\n🔫 Activating Sniper Mode (Threshold: {strategy.z_threshold} sigma)...")
    result = engine.run(strategy)
    
    # 4. Results
    final_eq = result['equity'].iloc[-1]
    ret = (final_eq / 100_000_000 - 1) * 100
    print(f"\n💰 Final Equity: {int(final_eq):,} KRW ({ret:.2f}%)")
    print(f"📉 Max Drawdown: {((result['equity'] / result['equity'].cummax() - 1).min() * 100):.2f}%")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(result.index, result['equity'], label='Sniper Equity')
    plt.title(f'N-Body Sniper Strategy (Z>{strategy.z_threshold})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()