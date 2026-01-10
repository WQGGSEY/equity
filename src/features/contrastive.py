import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from .base import BaseFeature
from .preprocessors import DollarBarStationaryFeature
from ..models.ts2vec_module import TSEncoder
from src.config import MODEL_WEIGHTS_DIR

# ==========================================
# [Helper] Micro Autoencoder
# ==========================================
class MicroAE(nn.Module):
    def __init__(self, input_dim=64, latent_dim=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

# ==========================================
# [Feature] Contrastive Feature Generator
# ==========================================
class Contrastive_OC_HL(BaseFeature):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Hyperparameters
        self.input_dim = self.params.get('input_dim', 2)
        self.hidden_dim = self.params.get('hidden_dim', 64)
        self.output_dim = self.params.get('output_dim', 64)
        self.depth = self.params.get('depth', 5)
        self.window_size = self.params.get('window_size', 64)
        
        # Compressor Path (Auto-defined)
        self.compressor_path = MODEL_WEIGHTS_DIR / "ts2vec_compressor.pth"
        
        # Device Check
        config_device = self.params.get('device', 'cpu')
        if torch.cuda.is_available() and config_device == 'cuda':
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available() and config_device == 'mps':
             self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # Load Global TSEncoder
        self.model_path = self.params.get('model_path', None)
        self.model = self._load_model()
        
        # Preprocessor
        self.preprocessor = DollarBarStationaryFeature(
            threshold=self.params.get('fd_threshold', 50_000),
            d=self.params.get('fd_d', 0.4)
        )
        
        # Load Compressor if exists
        self.compressor = self._load_compressor()

    def _load_model(self):
        model = TSEncoder(
            input_dim=self.input_dim, 
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim, 
            depth=self.depth
        ).to(self.device)
        
        if self.model_path and Path(self.model_path).exists():
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                model.eval()
            except Exception:
                pass
        return model

    def _load_compressor(self):
        """저장된 압축기 로드 (Worker 프로세스용)"""
        if self.compressor_path.exists():
            try:
                state = torch.load(self.compressor_path, map_location=self.device)
                latent_dim = state['latent_dim']
                
                comp = MicroAE(input_dim=self.output_dim, latent_dim=latent_dim).to(self.device)
                comp.load_state_dict(state['state_dict'])
                comp.eval()
                return comp
            except Exception:
                return None
        return None

    def train_and_save_compressor(self, df):
        """(Main 프로세스용) 압축기 학습 및 파일 저장"""
        print("🔧 [Calibration] Generating Embeddings for Calibration...")
        
        # 1. Generate Raw Embeddings
        # (Self.compute 내부 로직 일부 재사용)
        df_fd = self.preprocessor.compute(df)
        if df_fd.empty or len(df_fd) < self.window_size:
            print("   ⚠️ Not enough data for calibration.")
            return False
            
        fd_cols = ['FD_Open', 'FD_Close']
        if not all(c in df_fd.columns for c in fd_cols):
             if 'Open' in df_fd.columns and 'Close' in df_fd.columns:
                 df_input = df_fd[['Open', 'Close']].dropna()
             else:
                 return False
        else:
            df_input = df_fd[fd_cols].dropna()

        data_values = df_input.values.astype(np.float32).T 
        tensor_data = torch.from_numpy(data_values).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            full_out = self.model(tensor_data)
            pooled_out = F.avg_pool1d(full_out, kernel_size=self.window_size, stride=1)
            raw_embeddings = pooled_out.squeeze(0).transpose(0, 1).cpu().numpy()

        if len(raw_embeddings) < 500:
            print("   ⚠️ Sample size too small (<500).")
            return False

        # 2. Measure MLE
        print("   Running MLE Estimator...")
        sample_size = min(len(raw_embeddings), 3000)
        indices = np.random.choice(len(raw_embeddings), sample_size, replace=False)
        sample_data = raw_embeddings[indices]
        
        k = 20
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(sample_data)
        distances, _ = nbrs.kneighbors(sample_data)
        distances = distances[:, 1:] + 1e-10
        r_k = distances[:, -1].reshape(-1, 1)
        r_j = distances[:, :-1]
        mle = np.mean((k - 1) / (np.sum(np.log(r_k / r_j), axis=1) + 1e-9))
        
        target_dim = int(np.ceil(mle)) + 1
        print(f"   -> Measured MLE_ID: {mle:.2f} | Target Dim: {target_dim}")

        # 3. Train MicroAE
        compressor = MicroAE(input_dim=self.output_dim, latent_dim=target_dim).to(self.device)
        compressor.train()
        optimizer = optim.Adam(compressor.parameters(), lr=0.005)
        tensor_raw = torch.from_numpy(raw_embeddings).to(self.device)
        
        for _ in range(50):
            recon, _ = compressor(tensor_raw)
            loss = F.mse_loss(recon, tensor_raw)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # 4. Save to Disk
        torch.save({
            'state_dict': compressor.state_dict(),
            'latent_dim': target_dim
        }, self.compressor_path)
        
        self.compressor = compressor
        self.compressor.eval()
        print(f"   ✅ Compressor Saved to {self.compressor_path}")
        return True

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < self.window_size:
            return pd.DataFrame(index=df.index)

        # 1. FD Check & Compute
        fd_cols = ['FD_Open', 'FD_Close']
        if all(c in df.columns for c in fd_cols):
            df_input = df[fd_cols].dropna()
        else:
            try:
                df_fd = self.preprocessor.compute(df)
                if df_fd.empty: return pd.DataFrame(index=df.index)
                
                if all(c in df_fd.columns for c in fd_cols):
                    df_input = df_fd[fd_cols].dropna()
                elif 'Open' in df_fd.columns and 'Close' in df_fd.columns:
                    df_input = df_fd[['Open', 'Close']].dropna()
                else:
                    return pd.DataFrame(index=df.index)
            except:
                return pd.DataFrame(index=df.index)

        if len(df_input) < self.window_size:
            return pd.DataFrame(index=df.index)

        # 2. Inference
        data_values = df_input.values.astype(np.float32).T 
        tensor_data = torch.from_numpy(data_values).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            full_out = self.model(tensor_data)
            pooled_out = F.avg_pool1d(full_out, kernel_size=self.window_size, stride=1)
            raw_embeddings = pooled_out.squeeze(0).transpose(0, 1).cpu().numpy()

        # 3. Compression (Strict)
        # 압축기가 없으면 결과 반환 안 함 (데이터 일관성 위해)
        if self.compressor is None:
            # 혹시 파일이 늦게 생성되었을 수 있으니 한 번 더 로드 시도
            self.compressor = self._load_compressor()
            if self.compressor is None:
                return pd.DataFrame(index=df.index)

        with torch.no_grad():
            tensor_raw = torch.from_numpy(raw_embeddings).to(self.device)
            _, compressed_z = self.compressor(tensor_raw)
            final_embeddings = compressed_z.cpu().numpy()
            
        # latent_dim을 compressor에서 가져옴 (MicroAE의 linear layer shape 확인)
        latent_dim = self.compressor.encoder[-1].out_features
        feat_cols = [f'ts2vec_manifold_{i}' for i in range(latent_dim)]

        # 4. Result
        valid_indices = df_input.index[self.window_size - 1 :]
        min_len = min(len(final_embeddings), len(valid_indices))
        
        feat_df = pd.DataFrame(
            final_embeddings[:min_len], 
            index=valid_indices[:min_len], 
            columns=feat_cols
        )
        
        return feat_df.reindex(df.index)