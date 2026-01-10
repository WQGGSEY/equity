import torch
import numpy as np
import pandas as pd
from pathlib import Path
from .base import BaseFeature
from .preprocessors import DollarBarStationaryFeature
from ..models.ts2vec_module import TSEncoder

class Contrastive_OC_HL(BaseFeature):
    """
    TS2Vec (Open-Close vs High-Low) 기반 피처 생성기.
    학습된 TSEncoder를 사용하여, Body(Open, Close) 정보의 Latent Vector를 추출합니다.
    
    [Process]
    Raw OHLCV -> DollarBar & FD Preprocessing -> Sliding Window -> TS2Vec Encoder -> Feature Vector
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 1. 모델 하이퍼파라미터 설정 (학습된 모델의 설정과 일치해야 함)
        self.input_dim = self.params.get('input_dim', 2)     # Body: Open, Close
        self.hidden_dim = self.params.get('hidden_dim', 64)
        self.output_dim = self.params.get('output_dim', 64)
        self.depth = self.params.get('depth', 10)
        self.window_size = self.params.get('window_size', 64)
        
        # 2. 장치 설정 (Auto-detect)
        config_device = self.params.get('device', 'cpu')
        if torch.cuda.is_available() and config_device == 'cuda':
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available() and config_device == 'mps':
             self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # 3. 모델 로드
        self.model_path = self.params.get('model_path', None)
        self.model = self._load_model()
        
        # 4. 전처리기 초기화
        # 모델은 FD(Fractional Diff) 처리된 데이터를 학습했으므로, 인퍼런스 시에도 동일한 전처리가 필요함.
        # 파라미터는 config에서 전달받거나 기본값 사용
        self.preprocessor = DollarBarStationaryFeature(
            threshold=self.params.get('fd_threshold', 50_000),
            d=self.params.get('fd_d', 0.4)
        )

    def _load_model(self):
        """TSEncoder 모델 구조를 생성하고 가중치를 로드합니다."""
        model = TSEncoder(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth
        ).to(self.device)
        
        if self.model_path and Path(self.model_path).exists():
            try:
                # Weights 파일 로드 (map_location으로 장치 호환성 확보)
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                
                # Checkpoint Dict(학습 재개용)인지 State Dict(가중치만)인지 확인 후 로드
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                    
                model.eval() # 추론 모드 설정
            except Exception as e:
                print(f"❌ [Contrastive] Failed to load model from {self.model_path}: {e}")
        else:
            print(f"⚠️ [Contrastive] Model path not found or invalid: {self.model_path}")
            
        return model

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input: Raw OHLCV DataFrame
        Output: TS2Vec Embedding Features (aligned with index)
        """
        # 데이터 길이가 윈도우보다 작으면 계산 불가
        if df.empty or len(df) < self.window_size:
            return pd.DataFrame(index=df.index)

        # 1. FD 전처리 (Dollar Bar -> FracDiff -> Daily Alignment)
        try:
            df_fd = self.preprocessor.compute(df)
        except Exception:
            return pd.DataFrame(index=df.index)
            
        if df_fd.empty or len(df_fd) < self.window_size:
            return pd.DataFrame(index=df.index)

        # 2. Body View (Open, Close) 컬럼 선택
        # Preprocessor 결과 컬럼명 확인 (FD_Open 등)
        cols = ['FD_Open', 'FD_Close']
        if not all(c in df_fd.columns for c in cols):
             # 만약 전처리기가 컬럼명을 변경하지 않았다면 원본 컬럼명 사용
             if 'Open' in df_fd.columns and 'Close' in df_fd.columns:
                 cols = ['Open', 'Close']
             else:
                 # 필수 컬럼 부재 시 빈 DF 반환
                 return pd.DataFrame(index=df.index)

        # 데이터 텐서 변환 (N, 2)
        data_values = df_fd[cols].values.astype(np.float32)
        tensor_data = torch.from_numpy(data_values).to(self.device)

        # 3. Sliding Window 생성 (Efficient Unfold)
        # Input: (N, 2) -> Output: (N-W+1, 2, W)
        # TSEncoder는 (Batch, Channel, Length) 형태를 입력으로 받음
        windows = tensor_data.unfold(0, self.window_size, 1)
        
        # 4. Batch Inference
        embeddings = []
        batch_size = 256 # 메모리 상황에 따라 조절 가능
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch = windows[i : i + batch_size] # (B, 2, W)
                
                # Forward Pass -> (B, Output_Dim, W)
                out = self.model(batch)
                
                # Pooling: Instance Representation (Mean over time dimension)
                # 학습 시 Instance Discrimination을 위해 사용한 방식과 동일하게 적용
                out_pooled = out.mean(dim=2) # (B, Output_Dim)
                
                embeddings.append(out_pooled.cpu().numpy())
                
        if not embeddings:
            return pd.DataFrame(index=df.index)
            
        full_embeds = np.concatenate(embeddings, axis=0) # (N-W+1, D)
        
        # 5. DataFrame 생성 및 인덱스 정렬
        # 임베딩 벡터는 윈도우의 '마지막 시점'의 정보를 나타낸다고 가정하고 정렬
        # 예: 윈도우 크기가 64라면, 64번째 데이터 포인트(인덱스 63)부터 값이 생성됨
        valid_indices = df_fd.index[self.window_size - 1 :]
        
        # 컬럼명 생성 (예: ts2vec_ochl_0, ts2vec_ochl_1, ...)
        feat_cols = [f'ts2vec_{i}' for i in range(self.output_dim)]
        feat_df = pd.DataFrame(full_embeds, index=valid_indices, columns=feat_cols)
        
        # 원본 df 인덱스에 맞춰 Reindex (앞부분은 NaN 처리)
        feat_df = feat_df.reindex(df.index)
        
        return feat_df