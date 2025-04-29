import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """
    位置編碼模塊，為輸入序列添加位置信息
    """
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
        """
        Args:
            x: 輸入張量 [batch_size, seq_len, d_model]
        Returns:
            添加位置編碼的張量 [batch_size, seq_len, d_model]
        """
        if x.size(1) > self.pe.size(1):
            pe = self._extend_pe(x.size(1), x.device)
            return x + pe[:, :x.size(1), :]
        return x + self.pe[:, :x.size(1), :]
    
    def _extend_pe(self, length, device):
        """擴展位置編碼以適應更長的序列"""
        pe = torch.zeros(1, length, self.pe.size(2), device=device)
        position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.pe.size(2), 2, device=device).float() * (-math.log(10000.0) / self.pe.size(2)))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[:self.pe.size(2)//2])
        return pe

class DataEmbedding_inverted(nn.Module):
    """
    iTransformer的數據嵌入模塊，將輸入張量從時間序列表示轉換為變量token表示
    """
    def __init__(self, seq_len, d_model, dropout=0.1, use_norm=True):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.position_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: 輸入張量 [batch_size, seq_len, input_dim]
            mask: 可選的掩碼
        Returns:
            嵌入後的張量 [batch_size, input_dim, d_model]
        """
        # 轉置為變量token表示
        x = x.permute(0, 2, 1)  # [batch_size, input_dim, seq_len]
        
        # 嵌入每個變量token
        x = self.value_embedding(x)  # [batch_size, input_dim, d_model]
        
        # 位置編碼
        x = self.position_encoding(x)
        
        # 可選的LayerNorm
        if self.use_norm:
            x = self.norm(x)
            
        return self.dropout(x)

class FullAttention(nn.Module):
    """
    標準的多頭自註意力機制
    """
    def __init__(self, d_model, n_heads, dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.output_attention = output_attention
        
    def forward(self, queries, keys, values, attn_mask=None):
        """
        Args:
            queries, keys, values: 查詢、鍵、值張量 [batch_size, seq_len, d_model]
            attn_mask: 注意力掩碼
        Returns:
            輸出張量和注意力權重
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        
        # 映射查詢、鍵、值
        queries = self.query_proj(queries).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D/H]
        keys = self.key_proj(keys).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D/H]
        values = self.value_proj(values).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D/H]
        
        # 計算注意力分數
        attn_scores = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale  # [B, H, L, S]
        
        # 應用掩碼（如果提供）
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
            
        # 計算注意力權重並應用dropout
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, L, S]
        attn_weights = self.dropout(attn_weights)
        
        # 計算注意力輸出
        attn_output = torch.matmul(attn_weights, values)  # [B, H, L, D/H]
        attn_output = attn_output.transpose(1, 2).reshape(B, L, self.d_model)  # [B, L, D]
        attn_output = self.out_proj(attn_output)
        
        if self.output_attention:
            return attn_output, attn_weights
        else:
            return attn_output, None

class FeedForward(nn.Module):
    """
    前饋神經網絡，包含兩個線性變換和一個非線性激活函數
    """
    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"激活函數 {activation} 不支持")
            
    def forward(self, x):
        """
        Args:
            x: 輸入張量 [batch_size, seq_len, d_model]
        Returns:
            輸出張量 [batch_size, seq_len, d_model]
        """
        x = self.fc1(x)  # [batch_size, seq_len, d_ff]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [batch_size, seq_len, d_model]
        return x

class EncoderLayer(nn.Module):
    """
    Transformer編碼器層，包含自註意力機制和前饋神經網絡
    """
    def __init__(self, d_model, d_ff, n_heads, dropout=0.1, activation='gelu', output_attention=False):
        super(EncoderLayer, self).__init__()
        self.attention = FullAttention(d_model, n_heads, dropout, output_attention)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None):
        """
        Args:
            x: 輸入張量 [batch_size, seq_len, d_model]
            attn_mask: 注意力掩碼
        Returns:
            輸出張量和注意力權重
        """
        # 自註意力層
        attn_output, attn_weights = self.attention(
            self.norm1(x), self.norm1(x), self.norm1(x), attn_mask
        )
        x = x + self.dropout(attn_output)  # 殘差連接
        
        # 前饋網絡
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)  # 殘差連接
        
        return x, attn_weights

class Encoder(nn.Module):
    """
    Transformer編碼器，包含多個編碼器層
    """
    def __init__(self, d_model, d_ff, n_heads, n_layers, dropout=0.1, activation='gelu', output_attention=False):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, n_heads, dropout, activation, output_attention)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_attention = output_attention
        
    def forward(self, x, attn_mask=None):
        """
        Args:
            x: 輸入張量 [batch_size, seq_len, d_model]
            attn_mask: 注意力掩碼
        Returns:
            編碼後的張量和注意力權重
        """
        attns = [] if self.output_attention else None
        
        for layer in self.layers:
            x, attn = layer(x, attn_mask)
            if self.output_attention:
                attns.append(attn)
                
        x = self.norm(x)
        
        return x, attns

class iTransformerModel(nn.Module):
    """
    iTransformer模型，用於時間序列預測
    
    變量作為Tokens來捕捉多變量的相關性，而不是傳統Transformer中將時間步作為Tokens的方式
    """
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim=1, 
                 seq_len=None,
                 pred_len=1,
                 num_encoder_layers=2, 
                 num_decoder_layers=1,  # 用於預測頭
                 num_heads=8, 
                 dropout=0.1,
                 d_ff=None,
                 activation='gelu',
                 output_attention=False,
                 use_norm=True,
                 target_idx=None):
        super(iTransformerModel, self).__init__()
        
        # 初始化參數
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_norm = use_norm
        self.output_attention = output_attention
        self.target_idx = target_idx  # 目標變量的索引，例如T_CDU_out
        
        # 如果d_ff未指定，則設為hidden_dim的4倍
        if d_ff is None:
            d_ff = 4 * hidden_dim
            
        # 編碼器嵌入
        self.encoder_embedding = DataEmbedding_inverted(
            seq_len=seq_len, 
            d_model=hidden_dim, 
            dropout=dropout,
            use_norm=use_norm
        )
        
        # 編碼器
        self.encoder = Encoder(
            d_model=hidden_dim,
            d_ff=d_ff,
            n_heads=num_heads,
            n_layers=num_encoder_layers,
            dropout=dropout,
            activation=activation,
            output_attention=output_attention
        )
        
        # 預測頭 - 將編碼的變量表示映射到未來時間步
        self.predictor = nn.Linear(hidden_dim, pred_len)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: 輸入張量 [batch_size, seq_len, input_dim]
            mask: 可選的掩碼
        Returns:
            預測結果和注意力權重（如果output_attention=True）
        """
        B, L, N = x.shape  # batch_size, seq_len, num_variables
        
        # 數據歸一化（可選）
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
        
        # 動態設置seq_len（如果之前未設置）
        if self.seq_len is None:
            self.seq_len = L
            # 更新嵌入層以適應新的seq_len
            self.encoder_embedding.value_embedding = nn.Linear(self.seq_len, self.hidden_dim).to(x.device)
        
        # 設置目標變量索引（如果未指定）
        if self.target_idx is None:
            self.target_idx = 1  # 默認使用第二個變量（通常是T_CDU_out）
        
        # 編碼器嵌入 - 變量作為Tokens
        enc_out = self.encoder_embedding(x, mask)  # [batch_size, input_dim, hidden_dim]
        
        # 通過編碼器
        enc_out, attns = self.encoder(enc_out, attn_mask=None)  # [batch_size, input_dim, hidden_dim]
        
        # 預測 - 專注於目標變量
        # 方式1：只使用目標變量的編碼表示
        # target_feat = enc_out[:, self.target_idx, :]  # [batch_size, hidden_dim]
        # dec_out = self.predictor(target_feat).unsqueeze(-1)  # [batch_size, pred_len, 1]
        
        # 方式2：使用所有變量的編碼表示來預測所有變量
        dec_out = self.predictor(enc_out).transpose(1, 2)  # [batch_size, pred_len, input_dim]
        
        # 如果我們只關心特定變量的預測結果
        if self.output_dim == 1:
            dec_out = dec_out[:, :, self.target_idx:self.target_idx+1]  # [batch_size, pred_len, 1]
        
        # 反歸一化（如果使用了歸一化）
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out  # [batch_size, pred_len, output_dim] 