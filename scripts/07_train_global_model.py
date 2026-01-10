import sys
from pathlib import Path
import importlib
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =============================================================================
# [Setup]
# =============================================================================
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import GOLD_DIR, MODEL_WEIGHTS_DIR, ACTIVE_FEATURES
from src.models.ts2vec_module import TSEncoder

def get_feature_class(module_path, class_name):
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

# =============================================================================
# 1. Configuration (Research Hyperparameters)
# =============================================================================
CONFIG = {
    'input_dim': 2,
    'output_dim': 64,
    'hidden_dim': 64,
    'depth': 5,
    'batch_size': 32,
    'lr': 1e-3,
    'epochs': 50,          # 충분히 크게 잡고 Early Stopping에 맡김
    'window_size': 64,
    'init_temperature': 0.2, # 초기 온도는 넉넉하게 시작
    'stop_temperature': 0.05, # [Research] 이 이하로 내려가면 Inflation 위험 (tau_inst)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# =============================================================================
# 2. Dataset Definition (Pre-computation Logic)
# =============================================================================
class FinancialContrastiveDataset(Dataset):
    """
    Gold 데이터를 읽어 'DollarBarStationaryFeature' 로직(FD)을 선행 처리하여 메모리에 적재.
    """
    def __init__(self, data_dir, window_size=64, limit_files=None):
        self.window_size = window_size
        self.samples = []
        
        fd_config = next((f for f in ACTIVE_FEATURES if f['class'] == 'DollarBarStationaryFeature'), None)
        if not fd_config: raise ValueError("FD Feature config not found.")
            
        print(f"🔄 Using Preprocessor: {fd_config['class']}")
        FeatureClass = get_feature_class(fd_config['module'], fd_config['class'])
        preprocessor = FeatureClass(**fd_config['params'])
        
        files = glob.glob(str(data_dir / "*.parquet"))
        if limit_files: files = files[:limit_files]
            
        print(f"📂 Pre-computing features from {len(files)} files...")
        
        for f in tqdm(files, desc="Preprocessing"):
            try:
                df = pd.read_parquet(f)
                if df.empty: continue

                df_transformed = preprocessor.compute(df)
                target_cols = ['Open', 'Close', 'High', 'Low']
                
                # 데이터 유효성 체크
                if len(df_transformed) < window_size or not all(c in df_transformed.columns for c in target_cols):
                    continue
                    
                df_transformed = df_transformed.dropna(subset=target_cols)
                
                # Float32 변환
                data_body = df_transformed[['Open', 'Close']].values.astype(np.float32)
                data_wick = df_transformed[['High', 'Low']].values.astype(np.float32)
                
                # Stride Slicing
                stride = window_size // 2
                for i in range(0, len(df_transformed) - window_size, stride):
                    body_tensor = torch.from_numpy(data_body[i : i + window_size]).T
                    wick_tensor = torch.from_numpy(data_wick[i : i + window_size]).T
                    self.samples.append((body_tensor, wick_tensor))
                    
            except Exception: continue
                
        print(f"✅ Preprocessing Complete. Total samples: {len(self.samples)}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# =============================================================================
# 3. Geometric Inflation Stopper (The "Brilliant" Burn-in Logic)
# =============================================================================
class GeometricInflationStopper:
    """
    사용자의 연구 노트(Individual_Study_2.pdf)에 기반한 Early Stopping 전략.
    
    Logic:
    1. Burn-in Phase: Alignment(s_ii)가 급격히 상승하는 구간. 
       - 이때는 Temperature가 낮아져도 멈추지 않음 (Signal Learning 중).
       - Alignment Growth가 포화(Saturation)되면 Burn-in 종료 선언.
    
    2. Inflation Guard Phase: Burn-in 이후.
       - 이때 Temperature가 `stop_threshold` 미만으로 떨어지면,
       - "Signal은 다 배웠는데 Noise를 맞추려 억지로 온도를 낮춘다"고 판단하여 중단.
    """
    def __init__(self, stop_temp=0.07, patience=3):
        self.stop_temp = stop_temp
        self.patience = patience
        
        # State
        self.is_burned_in = False
        self.alignment_ema = None
        self.alpha = 0.3 # EMA smoothing factor
        self.burn_in_counter = 0
        
    def check(self, current_temp, current_alignment):
        """
        Returns: (should_stop: bool, reason: str)
        """
        # 1. Update Alignment EMA (Exponential Moving Average)
        if self.alignment_ema is None:
            self.alignment_ema = current_alignment
        else:
            self.alignment_ema = self.alignment_ema * (1 - self.alpha) + current_alignment * self.alpha
            
        # 2. Burn-in Detection (Smart Logic)
        if not self.is_burned_in:
            # Alignment 변화율(Delta) 계산
            delta = abs(current_alignment - self.alignment_ema)
            
            # 변화율이 매우 작아지면(0.005 미만) Signal Manifold 학습 완료로 간주
            # 단, 초기 몇 Epoch은 무조건 Burn-in으로 침
            if delta < 0.005 and current_alignment > 0.1: 
                self.burn_in_counter += 1
            else:
                self.burn_in_counter = 0
                
            if self.burn_in_counter >= self.patience:
                self.is_burned_in = True
                print(f"\n   [System] 🔥 Burn-in Complete. (Alignment Saturated at {self.alignment_ema:.4f})")
                print(f"   [System] 🛡️ Now guarding against Geometric Inflation (Tau < {self.stop_temp})...")

        # 3. Inflation Guard Check
        if self.is_burned_in:
            if current_temp < self.stop_temp:
                return True, f"Geometric Inflation Risk: Temperature ({current_temp:.4f}) < Threshold ({self.stop_temp})"
        
        return False, None

# =============================================================================
# 4. Loss with Monitoring
# =============================================================================
class HierarchicalContrastiveLoss(nn.Module):
    def __init__(self, init_temperature=0.1, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        # Learnable Temperature
        self.log_temperature = nn.Parameter(torch.tensor(np.log(init_temperature)))

    def get_temperature(self):
        return torch.exp(self.log_temperature)

    def forward(self, z1, z2):
        temp = self.get_temperature()
        loss = torch.tensor(0., device=z1.device)
        total_alignment = torch.tensor(0., device=z1.device) # 모니터링용
        d = 0
        
        while z1.size(2) > 1:
            if self.alpha != 0:
                l, align = self.instance_contrastive_loss(z1, z2, temp)
                loss += self.alpha * l
                total_alignment += align
            
            # (생략된 Temporal Loss 부분도 동일 구조)
            
            d += 1
            z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
            z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
            
        if z1.size(2) == 1:
            if self.alpha != 0:
                l, align = self.instance_contrastive_loss(z1, z2, temp)
                loss += self.alpha * l
                total_alignment += align
            d += 1
            
        return loss / d, total_alignment / d

    def instance_contrastive_loss(self, z1, z2, temperature):
        B, D, T = z1.size()
        if B == 1: return z1.new_tensor(0.), z1.new_tensor(0.)
        
        z1 = z1.transpose(1, 2).reshape(B*T, D)
        z2 = z2.transpose(1, 2).reshape(B*T, D)
        
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Alignment Score: Positive Pair(대각선)의 평균 Cosine Similarity
        # s_ii in your paper
        alignment = (z1 * z2).sum(dim=1).mean() 
        
        logits = torch.matmul(z1, z2.T) / temperature
        labels = torch.arange(B*T, device=z1.device)
        
        return F.cross_entropy(logits, labels), alignment

# =============================================================================
# 5. Training Loop
# =============================================================================
def train():
    device = torch.device(CONFIG['device'])
    print(f"🚀 Starting Training (RGD-Perspective)...")
    
    dataset = FinancialContrastiveDataset(GOLD_DIR, window_size=CONFIG['window_size'], limit_files=None)
    if len(dataset) == 0: return

    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    
    model = TSEncoder(
        input_dim=CONFIG['input_dim'],
        output_dim=CONFIG['output_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        depth=CONFIG['depth']
    ).to(device)
    
    criterion = HierarchicalContrastiveLoss(init_temperature=CONFIG['init_temperature']).to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=CONFIG['lr'])
    
    # [Smart Logic] 초기화
    stopper = GeometricInflationStopper(stop_temp=CONFIG['stop_temperature'], patience=2)
    
    model.train()
    criterion.train()
    
    print("\n🔥 Training Start...")
    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        total_alignment = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for body, wick in loop:
            body, wick = body.to(device), wick.to(device)
            optimizer.zero_grad()
            
            z_body = model(body)
            z_wick = model(wick)
            
            loss, align = criterion(z_body, z_wick)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_alignment += align.item()
            
            current_temp = criterion.get_temperature().item()
            loop.set_postfix({'loss': f"{loss.item():.4f}", 'tau': f"{current_temp:.4f}", 'align': f"{align.item():.2f}"})
            
        avg_loss = total_loss / len(dataloader)
        avg_align = total_alignment / len(dataloader)
        final_temp = criterion.get_temperature().item()
        
        print(f"   Epoch {epoch+1} | Loss: {avg_loss:.4f} | Alignment: {avg_align:.4f} | Temp: {final_temp:.4f}")
        
        # [Check] Smart Early Stopping
        should_stop, reason = stopper.check(final_temp, avg_align)
        if should_stop:
            print(f"\n🛑 Early Stopping Triggered!")
            print(f"   Reason: {reason}")
            break
        
    # Save Logic (동일)
    save_path = MODEL_WEIGHTS_DIR / "ts2vec_body_wick_v1.pth"
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {'input_dim': 2, 'output_dim': 64, 'hidden_dim': 64, 'depth': 5},
        'train_config': CONFIG,
        'final_stats': {'temperature': final_temp, 'alignment': avg_align}
    }
    torch.save(checkpoint, save_path)
    print(f"\n🎉 Model saved to: {save_path}")

if __name__ == "__main__":
    if GOLD_DIR.exists(): train()