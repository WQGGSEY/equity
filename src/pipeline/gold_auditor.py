import pandas as pd
from tqdm import tqdm
from src.config import GOLD_DIR

def run_audit():
    print(">>> [Audit] Gold Layer Final Check")
    if not GOLD_DIR.exists(): return

    files = list(GOLD_DIR.glob("*.parquet"))
    issues = []
    
    for f in tqdm(files, desc="Auditing"):
        try:
            df = pd.read_parquet(f)
            ticker = f.stem
            
            if df.empty:
                issues.append({'ticker': ticker, 'error': 'Empty'})
                continue
            if (df[['Open','Close']] < 0).any().any():
                issues.append({'ticker': ticker, 'error': 'Negative Price'})
            if not df.index.is_monotonic_increasing:
                issues.append({'ticker': ticker, 'error': 'Unsorted Index'})
            
            pct = df['Close'].pct_change()
            if ((pct > 3.0) | (pct < -0.9)).any():
                issues.append({'ticker': ticker, 'error': 'Extreme Volatility'})
        except Exception as e:
            issues.append({'ticker': f.stem, 'error': str(e)})

    print("-" * 40)
    if issues:
        path = GOLD_DIR.parent / "audit_report.csv"
        pd.DataFrame(issues).to_csv(path, index=False)
        print(f"  ⚠️ {len(issues)}개 문제 발견 (audit_report.csv 확인)")
    else:
        print("  ✨ Perfect Integrity: 결함 없음.")
    print("-" * 40)

if __name__ == "__main__":
    run_audit()