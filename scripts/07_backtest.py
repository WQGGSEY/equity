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
from src.alpha.parser import AlphaParser

def calculate_metrics(df, initial_cash):
    # 빈 데이터프레임 방어 로직
    if df.empty:
        return {
            'Total Return': 0.0, 'CAGR': 0.0, 'MDD': 0.0, 
            'Sharpe Ratio': 0.0, 'Avg Daily Turnover': 0.0, 
            'Final Equity': initial_cash, 'Trading Days': 0
        }

    final_equity = df['equity'].iloc[-1]
    total_return = (final_equity / initial_cash) - 1
    
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.0
    cagr = (final_equity / initial_cash) ** (1 / years) - 1 if years > 0 else 0
    
    peak = df['equity'].cummax()
    drawdown = (df['equity'] - peak) / peak
    mdd = drawdown.min()
    
    daily_ret = df['equity'].pct_change().fillna(0)
    mean_ret = daily_ret.mean()
    std_ret = daily_ret.std()
    
    if std_ret == 0:
        sharpe = 0
    else:
        sharpe = (mean_ret / std_ret) * np.sqrt(252)
        
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

def save_report(result_df, metrics, config, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 텍스트 리포트 저장
    with open(output_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("="*40 + "\n")
        f.write(f" Backtest Report (US Market)\n") 
        f.write("="*40 + "\n")
        f.write(f"Experiment : {config.get('experiment_name', 'Unnamed')}\n")
        if not result_df.empty:
            f.write(f"Date Range : {result_df.index[0].date()} ~ {result_df.index[-1].date()}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Return : {metrics['Total Return']*100:6.2f} %\n")
        f.write(f"CAGR         : {metrics['CAGR']*100:6.2f} %\n")
        f.write(f"MDD          : {metrics['MDD']*100:6.2f} %\n")
        f.write(f"Sharpe Ratio : {metrics['Sharpe Ratio']:6.4f}\n")
        f.write(f"Avg Turnover : {metrics['Avg Daily Turnover']*100:6.2f} %\n")
        f.write(f"Final Equity : $ {metrics['Final Equity']:,.2f} USD\n")
        f.write("="*40 + "\n")
        
    print(f"📄 Report saved to {output_dir / 'report.txt'}")
    
    # 2. Config 스냅샷 저장
    with open(output_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)
        
    if result_df.empty:
        return

    # 3. CSV 저장
    # (A) 요약본
    summary_cols = [c for c in result_df.columns if c != 'positions']
    result_df[summary_cols].to_csv(output_dir / "daily_summary.csv")
    
    # (B) 상세 내역 (Daily Positions)
    pos_data = []
    for date, row in result_df.iterrows():
        equity = row['equity']
        cash = row['cash']
        cash_weight = cash / equity if equity > 0 else 0
        
        pos_data.append({
            'Date': date,
            'Ticker': 'CASH',
            'Price': 1.0,
            'Qty': cash, 
            'Value': cash,
            'Weight': cash_weight
        })
        
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
        pos_df['Weight_Pct'] = (pos_df['Weight'] * 100).round(2)
        pos_df.to_csv(output_dir / "daily_positions.csv", index=False)
        print(f"📄 Positions saved to {output_dir / 'daily_positions.csv'}")

    # 4. 차트 그리기
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(result_df.index, result_df['equity'], label='Equity', color='blue')
    plt.yscale('log')
    plt.title(f"Equity Curve (Log Scale): {config.get('experiment_name')}")
    plt.ylabel("Equity (USD)") 
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    peak = result_df['equity'].cummax()
    drawdown = (result_df['equity'] - peak) / peak
    plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
    plt.title(f"Drawdown (MDD: {metrics['MDD']*100:.2f}%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "performance_chart.png")

def main(config_path):
    print(f"📂 Loading Config: {config_path}")
    cfg = load_config(config_path)
    
    # -----------------------------------------------------------
    # [핵심 수정] 전략에 필요한 피처를 미리 분석하여 로더에 전달
    # -----------------------------------------------------------
    strat_cfg = cfg['strategy']
    required_features = []
    
    # 1. 'expressions' 파라미터가 있다면 (FormulaStrategy 등) 파싱
    if 'params' in strat_cfg and 'expressions' in strat_cfg['params']:
        exprs = strat_cfg['params']['expressions']
        if isinstance(exprs, str): exprs = [exprs]
        
        print(f"🔍 Analyzing Strategy Requirements...")
        parser = AlphaParser()
        try:
            # 수식에서 변수명(피처) 추출
            extracted = parser.extract_needed_features(exprs)
            required_features.extend(extracted)
        except Exception as e:
            print(f"  ⚠️ Failed to parse expressions: {e}")

    # 2. 명시적인 'required_features' 설정이 있다면 추가 (GoldenCross 등)
    if 'required_features' in strat_cfg:
        required_features.extend(strat_cfg['required_features'])
        
    # 중복 제거
    required_features = sorted(list(set(required_features)))
    print(f"🤖 [Auto-Detect] Loading Features: {required_features}")
    
    # -----------------------------------------------------------
    # 1. Setup & Load
    # -----------------------------------------------------------
    md = MarketData(PLATINUM_FEATURES_DIR)
    
    # [핵심] 필요한 것만 골라서 로드 (8GB 램 생존 & 데이터 정렬 보장)
    md.load_all(required_features=required_features)
    
    bt_cfg = cfg['backtest']
    initial_cash = bt_cfg.get('initial_cash', 100_000)
    
    fee_rate = bt_cfg.get('fee_rate', 0.0)
    slippage = bt_cfg.get('slippage', 0.0)
    total_cost_rate = fee_rate + slippage
    universe_size = bt_cfg.get('universe_size', 3000)
    
    print(f"⚙️  Engine Settings: Fee={fee_rate*100:.3f}%, Slippage={slippage*100:.3f}%")
    
    engine = BacktestEngine(
        md, 
        start_date=bt_cfg.get('start_date'), 
        end_date=bt_cfg.get('end_date'),
        fee_rate=total_cost_rate,
        universe_size=universe_size
    )
    
    # 2. Strategy Logic
    StrategyClass = get_strategy_class(strat_cfg['module'], strat_cfg['class'])
    strategy = StrategyClass(**strat_cfg['params'])
    
    # 3. Run
    print(f"▶️ Start Simulation: {cfg.get('experiment_name')}")
    result = engine.run(strategy, initial_cash=initial_cash)
    
    # 4. Analyze
    metrics = calculate_metrics(result, initial_cash)
    
    # Console Output
    print("\n" + "="*30)
    print(f" [Backtest Result (US Market)]")
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