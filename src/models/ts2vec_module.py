import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    """
    미래 정보를 참조하지 못하도록 Padding을 조절한 1D Convolution.
    일반적인 Conv1d는 양쪽에 패딩을 붙이지만, 
    여기서는 과거(왼쪽)에만 패딩을 붙이고 미래(오른쪽)는 잘라냅니다.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        
        # (Kernel-1) * Dilation 만큼 왼쪽에만 패딩을 주어 시계열 길이를 유지
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=self.padding, 
            dilation=dilation
        )

    def forward(self, x):
        # x: [Batch, Channel, Length]
        out = self.conv(x)
        
        # Conv1d가 오른쪽에 붙인 패딩(미래 영역)을 슬라이싱으로 제거
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        
        return out

class DilatedConvBlock(nn.Module):
    """
    Residual Connection + Dropout이 포함된 Causal Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(DilatedConvBlock, self).__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # 입력과 출력 채널이 다를 경우 차원을 맞춰주는 1x1 Conv
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        return self.relu2(out + residual)

class TSEncoder(nn.Module):
    """
    최종 Time-Series Encoder.
    Depth에 따라 Receptive Field가 기하급수적으로 늘어남.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64, depth=5, kernel_size=3, dropout=0.2):
        super(TSEncoder, self).__init__()
        
        self.input_fc = nn.Conv1d(input_dim, hidden_dim, 1)
        
        self.blocks = nn.ModuleList()
        
        # Dilated Convolutions Stacking (1, 2, 4, 8, ...)
        for i in range(depth):
            dilation = 2 ** i
            self.blocks.append(
                DilatedConvBlock(
                    hidden_dim, 
                    hidden_dim, 
                    kernel_size, 
                    dilation, 
                    dropout
                )
            )
            
        self.output_fc = nn.Conv1d(hidden_dim, output_dim, 1)

    def forward(self, x):
        # x: [Batch, Input_Dim, Seq_Len]
        x = self.input_fc(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.output_fc(x)
        # Output: [Batch, Output_Dim, Seq_Len]
        
        return x