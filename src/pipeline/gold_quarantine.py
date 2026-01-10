import pandas as pd
import shutil
from pathlib import Path
from src.config import GOLD_DIR, MASTER_PATH

def run_quarantine():
    print(">>> [Pipeline 05] Gold Data 격리 조치 (Quarantine)")
    
    # 1. 감사 리포트 확인
    # gold_auditor가 저장한 리포트 경로
    report_path = GOLD_DIR.parent / "audit_report.csv"
    
    if not report_path.exists():
        print("  ✅ 격리 대상 없음 (리포트 파일 미발견)")
        return

    try:
        df_error = pd.read_csv(report_path)
    except Exception:
        print("  ⚠️ 리포트 파일 읽기 실패. 격리를 건너뜁니다.")
        return
        
    if df_error.empty:
        print("  ✅ 격리 대상 없음 (리포트 깨끗함)")
        return

    # 2. 격리 폴더 준비
    quarantine_dir = GOLD_DIR.parent / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  🗑️ 격리 대상: {len(df_error)} 개 종목")
    print(f"  📂 이동 경로: {quarantine_dir}")

    # 3. 장부 로드 (상태 업데이트용)
    df_master = None
    if MASTER_PATH.exists():
        df_master = pd.read_csv(MASTER_PATH)

    moved_cnt = 0
    
    # 4. 파일 이동 및 장부 갱신
    for ticker in df_error['ticker']:
        src_path = GOLD_DIR / f"{ticker}.parquet"
        dst_path = quarantine_dir / f"{ticker}.parquet"
        
        # 파일이 실제로 존재하면 이동
        if src_path.exists():
            try:
                shutil.move(str(src_path), str(dst_path))
                moved_cnt += 1
                
                # 장부 업데이트: is_active -> False
                if df_master is not None:
                    mask = df_master['ticker'] == ticker
                    df_master.loc[mask, 'is_active'] = False
                    df_master.loc[mask, 'note'] = 'Quarantined: Integrity Fail'
            except Exception as e:
                print(f"    ⚠️ 이동 실패 ({ticker}): {e}")
    
    # 5. 장부 저장
    if df_master is not None and moved_cnt > 0:
        df_master.to_csv(MASTER_PATH, index=False)
        print("  📝 Master List 업데이트 완료 (격리 종목 비활성화)")

    print(f"  ✅ 격리 조치 완료: {moved_cnt} 개 파일 이동됨.")

if __name__ == "__main__":
    run_quarantine()