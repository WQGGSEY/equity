import sys
import os
from pathlib import Path
import importlib
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

# =============================================================================
# [Setup] Project Path & Config
# =============================================================================
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import GOLD_DIR, MODEL_WEIGHTS_DIR, ACTIVE_FEATURES
from src.models.ts2vec_module import TSEncoder

# Figures Directory
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Checkpoint Path
CHECKPOINT_PATH = MODEL_WEIGHTS_DIR / "ts2vec_checkpoint.pth"

# [MPS Compatibility] Device Selection Logic
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# =============================================================================
# 1. Configuration
# =============================================================================
CONFIG = {
    'input_dim': 2,
    'output_dim': 64,
    'hidden_dim': 64,
    'depth': 10,
    'batch_size': 256,    # Reduced for MPS/Local safety
    'lr': 1e-3,           
    'temp_lr': 1e-3,      
    'epochs': 10,
    'window_size': 64,
    'init_temperature': 1.0,
    'device': get_device(),
    'eval_sample_size': 2000,
    'test_size': 0.2,     
    'random_seed': 42,
    'freeze_threshold_tau': 0.5  # Condition for freezing check
}

# Set Seeds
torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])
random.seed(CONFIG['random_seed'])

print(f"⚡ Device Selected: {CONFIG['device']}")

# =============================================================================
# 2. Utils: Metric Estimators
# =============================================================================
def compute_mle_id(embeddings, k=20):
    if len(embeddings) < k + 1: return 0.0
    embeddings = embeddings.astype(np.float64) 
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    distances = distances[:, 1:] + 1e-10
    
    r_k = distances[:, -1].reshape(-1, 1)
    r_j = distances[:, :-1]
    
    ratio = r_k / r_j
    log_ratio = np.log(ratio)
    sum_log = np.sum(log_ratio, axis=1)
    
    mle_per_point = (k - 1) / sum_log
    return np.mean(mle_per_point)

def compute_global_top1_acc(model, dataset, device, sample_size=2000):
    model.eval()
    n_samples = min(len(dataset), sample_size)
    if n_samples == 0: return 0.0, np.array([])
    
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    loader = DataLoader([dataset[i] for i in indices], batch_size=128, shuffle=False)
    
    emb_body_list = []
    emb_wick_list = []
    
    with torch.no_grad():
        for body, wick in loader:
            body, wick = body.to(device), wick.to(device)
            z_body = model(body).mean(dim=2)
            z_wick = model(wick).mean(dim=2)
            emb_body_list.append(z_body.cpu())
            emb_wick_list.append(z_wick.cpu())
            
    if not emb_body_list: return 0.0, np.array([])
            
    emb_body = torch.cat(emb_body_list, dim=0)
    emb_wick = torch.cat(emb_wick_list, dim=0)
    
    emb_body = F.normalize(emb_body, dim=1)
    emb_wick = F.normalize(emb_wick, dim=1)
    
    sim_matrix = torch.matmul(emb_body, emb_wick.T)
    preds = torch.argmax(sim_matrix, dim=1)
    targets = torch.arange(len(preds))
    
    acc = (preds == targets).float().mean().item()
    return acc, emb_body.numpy()

# =============================================================================
# 3. Dataset
# =============================================================================
def get_feature_class(module_path, class_name):
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

class FinancialContrastiveDataset(Dataset):
    def __init__(self, file_paths, window_size=64, mode="Train"):
        self.window_size = window_size
        self.samples = []
        self.error_count = 0
        
        fd_config = next((f for f in ACTIVE_FEATURES if f['class'] == 'DollarBarStationaryFeature'), None)
        preprocessor = None
        if fd_config:
            print(f"[{mode}] 🔄 Init FD: {fd_config['class']}")
            try:
                FeatureClass = get_feature_class(fd_config['module'], fd_config['class'])
                preprocessor = FeatureClass(**fd_config['params'])
            except Exception as e:
                print(f"[{mode}] ❌ Preprocessor Init Fail: {e}")

        print(f"[{mode}] 📂 Loading {len(file_paths)} files...")
        
        for f in tqdm(file_paths, desc=f"Loading {mode} Data"):
            try:
                df = pd.read_parquet(f)
                if df.empty: continue
                
                col_map = {c: c.capitalize() for c in df.columns}
                for c in df.columns:
                    lower_c = c.lower()
                    if 'vol' in lower_c: col_map[c] = 'Volume'
                    elif 'adj' in lower_c: col_map[c] = 'Adj Close'
                    elif 'open' in lower_c: col_map[c] = 'Open'
                    elif 'high' in lower_c: col_map[c] = 'High'
                    elif 'low' in lower_c: col_map[c] = 'Low'
                    elif 'close' in lower_c: col_map[c] = 'Close'
                df = df.rename(columns=col_map)

                required_cols = ['Open', 'High', 'Low', 'Close']
                if not all(rc in df.columns for rc in required_cols):
                    continue

                if preprocessor:
                    df_transformed = preprocessor.compute(df)
                else:
                    df_transformed = df

                if df_transformed.empty: continue

                rename_map = {}
                for c in df_transformed.columns:
                    if 'Open' in c: rename_map[c] = 'Open'
                    elif 'Close' in c: rename_map[c] = 'Close'
                    elif 'High' in c: rename_map[c] = 'High'
                    elif 'Low' in c: rename_map[c] = 'Low'
                df_transformed = df_transformed.rename(columns=rename_map)
                
                df_transformed = df_transformed[required_cols].dropna()
                if len(df_transformed) < window_size: continue

                data_body = df_transformed[['Open', 'Close']].values.astype(np.float32)
                data_wick = df_transformed[['High', 'Low']].values.astype(np.float32)
                
                stride = window_size // 2
                for i in range(0, len(df_transformed) - window_size, stride):
                    body_tensor = torch.from_numpy(data_body[i : i + window_size]).T
                    wick_tensor = torch.from_numpy(data_wick[i : i + window_size]).T
                    self.samples.append((body_tensor, wick_tensor))
                    
            except Exception as e:
                if self.error_count < 3:
                    print(f"[{mode}] ❌ Error in {Path(f).name}: {e}")
                    self.error_count += 1
                continue
                
        print(f"[{mode}] ✅ Ready. Total Samples: {len(self.samples)}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# =============================================================================
# 4. Loss & Training Logic
# =============================================================================
class HierarchicalContrastiveLoss(nn.Module):
    def __init__(self, init_temperature=0.07):
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.tensor(np.log(init_temperature), dtype=torch.float32)
        )
        self.is_frozen = False # Freeze Flag

    def get_temperature(self):
        return torch.exp(self.log_temperature)
        
    def freeze(self):
        """Freeze the temperature parameter."""
        if not self.is_frozen:
            self.log_temperature.requires_grad = False
            self.log_temperature.grad = None
            self.is_frozen = True
            print(f"\n❄️ Temperature Frozen at {self.get_temperature().item():.4f}")

    def forward(self, z1, z2):
        temperature = self.get_temperature()
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = torch.matmul(z1, z2.T) / temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(logits, labels), temperature

def calculate_tau_inst(z1, z2, current_tau):
    with torch.no_grad():
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim_matrix = torch.matmul(z1, z2.T)
        alignment = torch.diag(sim_matrix)
        
        mask = torch.eye(z1.size(0), device=z1.device).bool()
        logits = sim_matrix / current_tau
        logits.masked_fill_(mask, -1e9)
        p_ik = F.softmax(logits, dim=1)
        
        weighted_noise = torch.sum(p_ik * sim_matrix.pow(2), dim=1)
        return (weighted_noise / (alignment + 1e-6)).mean().item()

def train():
    device = CONFIG['device']
    print(f"🚀 Initializing Training with Ticker Split on {device}...")

    # 1. File Discovery & Split
    all_files = glob.glob(str(GOLD_DIR / "*.parquet"))
    if not all_files:
        print("❌ No data found in Gold directory.")
        return

    train_files, test_files = train_test_split(all_files, test_size=CONFIG['test_size'], random_state=CONFIG['random_seed'])
    print(f"📊 Total Files: {len(all_files)} | Train: {len(train_files)} | Test: {len(test_files)}")

    # 2. Datasets
    train_dataset = FinancialContrastiveDataset(train_files, window_size=CONFIG['window_size'], mode="Train")
    test_dataset = FinancialContrastiveDataset(test_files, window_size=CONFIG['window_size'], mode="Test")

    if len(train_dataset) == 0:
        print("❌ Train dataset empty.")
        return

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    
    # 3. Model & Setup
    model = TSEncoder(CONFIG['input_dim'], CONFIG['output_dim'], CONFIG['hidden_dim'], CONFIG['depth']).to(device)
    criterion = HierarchicalContrastiveLoss(init_temperature=CONFIG['init_temperature']).to(device)
    
    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': CONFIG['lr']},
        {'params': criterion.parameters(), 'lr': CONFIG['temp_lr']}
    ])
    
    # --- [RESUME LOGIC] ---
    start_epoch = 0
    history = {'epoch': [], 'loss': [], 'test_acc': [], 'test_mle_id': [], 'tau': [], 'tau_inst': []}
    
    if CHECKPOINT_PATH.exists():
        print(f"\n🔄 Found Checkpoint at {CHECKPOINT_PATH}")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            criterion.load_state_dict(checkpoint['criterion_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint['history']
            
            # Restore Frozen State
            if 'is_frozen' in checkpoint and checkpoint['is_frozen']:
                criterion.freeze()
                
            print(f"   ✅ Successfully Resumed from Epoch {start_epoch}")
        except Exception as e:
            print(f"   ⚠️ Failed to load checkpoint: {e}")
            print("   --> Starting from scratch.")
    else:
        print("\n✨ No checkpoint found. Starting fresh training.")

    # 4. Training Loop
    print("\n🔥 Training Start...")
    for epoch in range(start_epoch, CONFIG['epochs']):
        model.train()
        ep_loss, ep_tau, ep_tau_inst = 0, 0, 0
        
        # [Train Phase]
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for body, wick in pbar:
            body, wick = body.to(device), wick.to(device)
            optimizer.zero_grad()
            
            z1 = model(body).mean(dim=2)
            z2 = model(wick).mean(dim=2)
            
            loss, temp = criterion(z1, z2)
            
            # Monitoring Tau Inst (No clamping inside batch)
            current_tau = temp.item()
            tau_inst_val = calculate_tau_inst(z1, z2, current_tau)
            
            loss.backward()
            optimizer.step()
            
            ep_loss += loss.item()
            ep_tau += current_tau
            ep_tau_inst += tau_inst_val
            
            # Log current batch tau_inst
            pbar.set_postfix({
                'L': f"{loss.item():.3f}", 
                'Tau': f"{current_tau:.3f}", 
                'T_Inst': f"{tau_inst_val:.3f}",
                'F': str(criterion.is_frozen)
            })
            
        # [End of Epoch Logic]
        avg_loss = ep_loss / len(train_loader)
        avg_tau = ep_tau / len(train_loader)
        avg_tau_inst = ep_tau_inst / len(train_loader)
        
        # Epoch-based Freeze Check
        if not criterion.is_frozen:
            # If average temp dropped below 0.5 AND is lower than average instability barrier
            if avg_tau < CONFIG['freeze_threshold_tau'] and avg_tau < avg_tau_inst:
                criterion.freeze()
                print(f"   ❄️ FREEZE TRIGGERED: Avg Tau ({avg_tau:.4f}) < Barrier ({avg_tau_inst:.4f})")
        
        # [Evaluation Phase]
        test_acc, embeddings = compute_global_top1_acc(model, test_dataset, device, sample_size=CONFIG['eval_sample_size'])
        test_mle_id = compute_mle_id(embeddings)
        
        # Print with Tau_Inst
        print(f"   Epoch {epoch+1}: Loss {avg_loss:.4f} | Test Acc {test_acc*100:.2f}% | Tau {avg_tau:.4f} | Tau_Inst {avg_tau_inst:.4f}")
        
        # Logging
        history['epoch'].append(epoch+1)
        history['loss'].append(avg_loss)
        history['tau'].append(avg_tau)
        history['tau_inst'].append(avg_tau_inst)
        history['test_acc'].append(test_acc)
        history['test_mle_id'].append(test_mle_id)

        # --- [SAVE CHECKPOINT] ---
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'criterion_state_dict': criterion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'is_frozen': criterion.is_frozen # Save freeze state
        }, CHECKPOINT_PATH)
        
        torch.save(model.state_dict(), MODEL_WEIGHTS_DIR / "ts2vec_learnable_tau.pth")
        
    # 5. Final Plotting
    print(f"\n💾 Model & Checkpoint Saved.")
    
    # 1. MLE ID
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['test_mle_id'], marker='o', label='Test Set MLE ID')
    plt.title('Generalization: Intrinsic Dimension on Unseen Tickers')
    plt.xlabel('Epoch')
    plt.ylabel('Dimension')
    plt.grid(True)
    plt.savefig(FIGURES_DIR / "mle_id_generalization.png")
    plt.close()
    
    # 2. Test Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['test_acc'], marker='s', color='green', label='Test Set Top-1 Acc')
    plt.title('Generalization: Instance Discrimination on Unseen Tickers')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(FIGURES_DIR / "test_accuracy.png")
    plt.close()
    
    # 3. Temperature Dynamics
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['tau'], label='Learnable Tau', linewidth=2)
    plt.plot(history['epoch'], history['tau_inst'], label='Instability Barrier', linestyle='--')
    plt.title('Training Dynamics: Temperature vs Barrier')
    plt.xlabel('Epoch')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True)
    plt.savefig(FIGURES_DIR / "temperature_dynamics.png")
    plt.close()
    
    print(f"📊 Figures updated in {FIGURES_DIR}")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n🛑 Training Interrupted by User.")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        raise e  # 에러 내용을 확인하기 위해 다시 발생시킴
    finally:
        # 학습 종료 후 MPS(또는 CUDA) 캐시 삭제
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            print("\n🧹 MPS Cache Cleared.")
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\n🧹 CUDA Cache Cleared.")