import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import yaml
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import PLATINUM_FEATURES_DIR
from src.backtest.loader import MarketData
from src.backtest.engine import BacktestEngine
from src.utils.config_loader import load_config, get_strategy_class

def calculate_metrics(df, initial_cash):
    """
    백테스트 결과 DataFrame을 받아 핵심 성과 지표를 계산합니다.
    """
    # 1. Basic Returns
    final_equity = df['equity'].iloc[-1]
    total_return = (final_equity / initial_cash) - 1
    
    # 2. Annual Metrics (CAGR)
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.0
    cagr = (final_equity / initial_cash) ** (1 / years) - 1 if years > 0 else 0
    
    # 3. MDD (Max Drawdown)
    # cummax: 지금까지의 최고점
    peak = df['equity'].cummax()
    drawdown = (df['equity'] - peak) / peak
    mdd = drawdown.min()
    
    # 4. Sharpe Ratio (Rf=0)
    # 일별 수익률
    daily_ret = df['equity'].pct_change().fillna(0)
    # 연율화 (Daily Mean * 252 / Daily Std * sqrt(252))
    # Sharpe = Mean / Std * sqrt(252)
    mean_ret = daily_ret.mean()
    std_ret = daily_ret.std()
    
    if std_ret == 0:
        sharpe = 0
    else:
        sharpe = (mean_ret / std_ret) * np.sqrt(252)
        
    # 5. Turnover Ratio (Daily Average)
    # Turnover Ratio = Daily Traded Value / Daily Equity
    daily_turnover_ratio = df['daily_turnover'] / df['equity']
    avg_turnover = daily_turnover_ratio.mean()
    
    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'MDD': mdd,
        'Sharpe Ratio': sharpe,
        'Avg Daily Turnover': avg_turnover,
        'Final Equity': final_equity,
        'Trading Days': len(df)
    }

# (앞부분 Import 및 calculate_metrics는 기존과 동일) ...

def save_report(result_df, metrics, config, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 텍스트 리포트
    with open(output_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("="*40 + "\n")
        f.write(f" Backtest Report\n")
        f.write("="*40 + "\n")
        f.write(f"Experiment : {config.get('experiment_name', 'Unnamed')}\n")
        f.write(f"Date Range : {result_df.index[0].date()} ~ {result_df.index[-1].date()}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Return : {metrics['Total Return']*100:6.2f} %\n")
        f.write(f"CAGR         : {metrics['CAGR']*100:6.2f} %\n")
        f.write(f"MDD          : {metrics['MDD']*100:6.2f} %\n")
        f.write(f"Sharpe Ratio : {metrics['Sharpe Ratio']:6.4f}\n")
        f.write(f"Avg Turnover : {metrics['Avg Daily Turnover']*100:6.2f} %\n")
        f.write(f"Final Equity : {int(metrics['Final Equity']):,} KRW\n")
        f.write("="*40 + "\n")
    print(f"📄 Report saved to {output_dir / 'report.txt'}")
    
    # 2. Config 스냅샷
    with open(output_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)
        
    # 3. 데이터 저장
    # (A) 요약본: positions 컬럼은 CSV로 저장하면 지저분하므로 제외하고 저장
    summary_cols = [c for c in result_df.columns if c != 'positions']
    result_df[summary_cols].to_csv(output_dir / "daily_summary.csv")
    
    # (B) [NEW] 상세 내역 (Daily Positions) 저장
    # DataFrame의 positions 컬럼(리스트)을 풀어서 별도 CSV 생성
    pos_data = []
    for date, row in result_df.iterrows():
        # 현금 비중 추가
        equity = row['equity']
        cash = row['cash']
        cash_weight = cash / equity if equity > 0 else 0
        
        # 현금 Row 추가 (선택사항)
        pos_data.append({
            'Date': date,
            'Ticker': 'CASH',
            'Price': 1,
            'Qty': int(cash),
            'Value': int(cash),
            'Weight': cash_weight
        })
        
        # 보유 주식 Row 추가
        if isinstance(row['positions'], list):
            for p in row['positions']:
                pos_data.append({
                    'Date': date,
                    'Ticker': p['ticker'],
                    'Price': p['price'],
                    'Qty': p['qty'],
                    'Value': p['value'],
                    'Weight': p['weight']
                })
                
    pos_df = pd.DataFrame(pos_data)
    if not pos_df.empty:
        # 보기 좋게 포맷팅 (날짜, 비중 등)
        pos_df['Weight_Pct'] = (pos_df['Weight'] * 100).round(2)
        # CSV 저장
        pos_df.to_csv(output_dir / "daily_positions.csv", index=False)
        print(f"📄 Positions saved to {output_dir / 'daily_positions.csv'}")

    # 4. 차트 (Log Scale)
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(result_df.index, result_df['equity'], label='Equity', color='blue')
    plt.yscale('log')
    plt.title(f"Equity Curve (Log): {config.get('experiment_name')}")
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    peak = result_df['equity'].cummax()
    drawdown = (result_df['equity'] - peak) / peak
    plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
    plt.title(f"MDD: {metrics['MDD']*100:.2f}%")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "performance_chart.png")

# (main 함수는 기존과 동일)

def main(config_path):
    print(f"📂 Loading Config: {config_path}")
    cfg = load_config(config_path)
    
    # 1. Setup
    md = MarketData(PLATINUM_FEATURES_DIR)
    md.load_all()
    
    bt_cfg = cfg['backtest']
    initial_cash = bt_cfg.get('initial_cash', 100_000_000)
    engine = BacktestEngine(md, start_date=bt_cfg.get('start_date'), end_date=bt_cfg.get('end_date'))
    
    # 2. Strategy Logic
    strat_cfg = cfg['strategy']
    StrategyClass = get_strategy_class(strat_cfg['module'], strat_cfg['class'])
    strategy = StrategyClass(**strat_cfg['params'])
    
    # 3. Run
    print(f"▶️ Start Simulation: {cfg.get('experiment_name')}")
    result = engine.run(strategy, initial_cash=initial_cash)
    
    # 4. Analyze
    metrics = calculate_metrics(result, initial_cash)
    
    # Console Output
    print("\n" + "="*30)
    print(f" [Backtest Result]")
    print(f" Return : {metrics['Total Return']*100:.2f}%")
    print(f" MDD    : {metrics['MDD']*100:.2f}%")
    print(f" Sharpe : {metrics['Sharpe Ratio']:.4f}")
    print("="*30 + "\n")
    
    # 5. Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = cfg.get('experiment_name', 'default').replace(" ", "_")
    output_dir = PROJECT_ROOT / "results" / f"{timestamp}_{exp_name}"
    
    save_report(result, metrics, cfg, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/strategies/sniper_v1.yaml")
    args = parser.parse_args()
    
    main(args.config)