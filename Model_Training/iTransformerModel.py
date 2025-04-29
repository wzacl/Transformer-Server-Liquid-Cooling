import torch
import torch.nn as nn
import math

# 位置編碼類
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            pe = self._extend_pe(x.size(1), x.device)
            return x + pe[:, :x.size(1), :]
        return x + self.pe[:, :x.size(1), :]
    
    def _extend_pe(self, length, device):
        pe = torch.zeros(1, length, self.pe.size(2))
        position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.pe.size(2), 2, device=device).float() * 
                           (-math.log(10000.0) / self.pe.size(2)))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[:len(div_term)])
        return pe

# 逆向數據嵌入類 - 用於iTransformer
class DataEmbedding_inverted(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.position_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        # 轉置為 [batch_size, input_dim, seq_len]，其中input_dim作為tokens
        x = x.permute(0, 2, 1)
        
        # 對每個變量token進行嵌入
        x = self.value_embedding(x)  # [batch_size, input_dim, d_model]
        x = self.position_encoding(x)
        return self.dropout(x)

# iTransformer 模型
class iTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_encoder_layers, num_decoder_layers, num_heads, dropout):
        super(iTransformerModel, self).__init__()
        
        # 初始化參數
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = None  # 將在forward中設置
        
        # 編碼器部分 - 反轉的嵌入
        self.encoder_embedding = DataEmbedding_inverted(seq_len=None, d_model=hidden_dim, dropout=dropout)
        
        # 編碼器層
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        # 預測器 - 將變量的隱藏表示映射到未來時間步
        self.predictor = nn.Linear(hidden_dim, output_dim * num_decoder_layers)
        
        # 輸出層
        self.t_cdu_out_idx = None  # 將在forward中設置
        
    def forward(self, src, num_steps):
        # src shape: [batch_size, seq_len, input_dim]
        # 設置seq_len和t_cdu_out_idx
        batch_size, self.seq_len, _ = src.shape
        
        # 通常第二個特徵是T_CDU_out，但我們在這裡設置一個默認值，後續可以根據具體情況修改
        self.t_cdu_out_idx = 1
        
        # 動態設置encoder_embedding的seq_len
        self.encoder_embedding.value_embedding = nn.Linear(self.seq_len, self.hidden_dim).to(src.device)
        
        # 編碼器嵌入 - 反轉的方式
        enc_out = self.encoder_embedding(src)  # [batch_size, input_dim, hidden_dim]
        
        # 通過Transformer編碼器
        enc_out = self.transformer_encoder(enc_out)  # [batch_size, input_dim, hidden_dim]
        
        # 預測未來時間步 - 專注於T_CDU_out變量
        # 提取T_CDU_out的隱藏表示
        t_cdu_hidden = enc_out[:, self.t_cdu_out_idx, :]  # [batch_size, hidden_dim]
        
        # 預測未來num_steps步的值
        pred = self.predictor(t_cdu_hidden)  # [batch_size, output_dim * num_steps]
        pred = pred.reshape(batch_size, -1, self.output_dim)  # [batch_size, num_steps, output_dim]
        
        # 只取需要的預測步數
        pred = pred[:, :num_steps, :]
        
        return pred 