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
        pe[0, :, 1::2] = torch.cos(position * div_term[:self.pe.size(2)//2])
        return pe

# Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_encoder_layers, num_decoder_layers, num_heads, dropout):
        super(TransformerModel, self).__init__()
        
        # 初始化 output_dim
        self.output_dim = output_dim
        
        # 編碼器部分
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        # 解碼器部分
        self.decoder_embedding = nn.Linear(output_dim, hidden_dim)
        self.pos_decoder = PositionalEncoding(hidden_dim)
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_decoder_layers)
        
        # 輸出層
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 生成解碼器掩碼
        self.tgt_mask = None
        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, src, num_steps):
        # src shape: [batch_size, seq_len, input_dim]
        
        # 編碼器
        src_embedded = self.embedding(src)  # [batch_size, seq_len, hidden_dim]
        src_embedded = self.pos_encoder(src_embedded)
        memory = self.transformer_encoder(src_embedded)  # 生成 memory

        # 初始化解碼器輸入為最後一個T_CDU_out值
        # 獲取序列中最後一個T_CDU_out值的索引（T_CDU_out是第二個特徵）
        t_cdu_out_idx = 1  # 根據features列表中T_CDU_out的位置調整
        last_t_cdu_out = src[:, -1:, t_cdu_out_idx:t_cdu_out_idx+1]  # 獲取最後一個時間步的T_CDU_out值
        tgt = last_t_cdu_out  # 使用最後一個T_CDU_out值作為初始輸入
        
        # 解碼器（自回歸模式 - Auto-regressive decoding）
        outputs = []
        for step in range(num_steps):
            # 檢查解碼器是否正確使用自回歸方式：
            # 1. tgt包含了之前所有的預測結果，形狀為[batch_size, current_len, output_dim]
            # 2. 每一步都使用之前的預測作為輸入，而不是一次性生成所有預測
            
            tgt_embedded = self.decoder_embedding(tgt)  # [batch_size, tgt_len, hidden_dim]
            tgt_embedded = self.pos_decoder(tgt_embedded)
            
            # 修改 mask 生成邏輯，確保解碼器只能看到過去的輸出
            if self.tgt_mask is None or self.tgt_mask.size(0) != tgt.size(1):
                device = tgt.device
                mask = self._generate_square_subsequent_mask(tgt.size(1)).to(device)
                self.tgt_mask = mask

            output = self.transformer_decoder(tgt_embedded, memory, tgt_mask=self.tgt_mask)
            
            # 輸出層
            output = self.fc(output)  # [batch_size, tgt_len, output_dim]
            outputs.append(output[:, -1, :])
            
            # 自回歸解碼：使用當前輸出作為下一步輸入
            # 這確保了每一步的預測都依賴於前一步的預測，符合自回歸模式
            tgt = torch.cat((tgt, output[:, -1:, :]), dim=1)
        
        return torch.stack(outputs, dim=1)