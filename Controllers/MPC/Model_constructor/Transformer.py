# 位置編碼類
import torch
import torch.nn as nn
import math

# 位置編碼類
class PositionalEncoding(nn.Module):
    """
    位置編碼模組，用於為Transformer模型提供序列位置信息。
    
    位置編碼使用正弦和餘弦函數生成，能夠讓模型理解序列中元素的相對位置關係。
    
    Attributes:
        pe (torch.Tensor): 預計算的位置編碼矩陣，形狀為 [1, max_len, d_model]
    """
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置編碼模組。
        
        Args:
            d_model (int): 模型的維度大小
            max_len (int, optional): 支持的最大序列長度，默認為5000
        """
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
        前向傳播函數，將位置編碼添加到輸入張量。
        
        Args:
            x (torch.Tensor): 輸入張量，形狀為 [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: 添加位置編碼後的張量
        """
        if x.size(1) > self.pe.size(1):
            pe = self._extend_pe(x.size(1), x.device)
            return x + pe[:, :x.size(1), :]
        return x + self.pe[:, :x.size(1), :]
    
    def _extend_pe(self, length, device):
        """
        擴展位置編碼以適應更長的序列。
        
        Args:
            length (int): 需要的序列長度
            device (torch.device): 計算設備
            
        Returns:
            torch.Tensor: 擴展後的位置編碼
        """
        pe = torch.zeros(1, length, self.pe.size(2))
        position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.pe.size(2), 2, device=device).float() * 
                           (-math.log(10000.0) / self.pe.size(2)))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[:self.pe.size(2)//2])
        return pe

# Transformer 模型
class TransformerModel(nn.Module):
    """
    基於Transformer架構的序列到序列模型，用於時間序列預測。
    
    該模型包含編碼器和解碼器部分，能夠處理輸入序列並生成預測序列。
    特別適用於溫度控制系統中的預測控制。
    
    Attributes:
        output_dim (int): 輸出維度
        embedding (nn.Linear): 輸入特徵嵌入層
        pos_encoder (PositionalEncoding): 編碼器位置編碼
        transformer_encoder (nn.TransformerEncoder): Transformer編碼器
        decoder_embedding (nn.Linear): 解碼器輸入嵌入層
        pos_decoder (PositionalEncoding): 解碼器位置編碼
        transformer_decoder (nn.TransformerDecoder): Transformer解碼器
        fc (nn.Linear): 輸出層
        tgt_mask (torch.Tensor): 解碼器掩碼
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout):
        """
        初始化Transformer模型。
        
        Args:
            input_dim (int): 輸入特徵維度
            hidden_dim (int): 隱藏層維度
            output_dim (int): 輸出特徵維度
            num_layers (int): Transformer層數
            num_heads (int): 多頭注意力機制的頭數
            dropout (float): Dropout比率
        """
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
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 解碼器部分
        self.decoder_embedding = nn.Linear(output_dim, hidden_dim)
        self.pos_decoder = PositionalEncoding(hidden_dim)
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        
        # 輸出層
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 生成解碼器掩碼
        self.tgt_mask = None
        
    def _generate_square_subsequent_mask(self, sz):
        """
        生成用於解碼器的方形後續掩碼，確保預測時只能看到當前及之前的位置。
        
        Args:
            sz (int): 掩碼大小
            
        Returns:
            torch.Tensor: 生成的掩碼矩陣
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, src, num_steps):
        """
        模型前向傳播函數，處理輸入序列並生成預測序列。
        
        Args:
            src (torch.Tensor): 輸入序列，形狀為 [batch_size, seq_len, input_dim]
            num_steps (int): 需要預測的步數
            
        Returns:
            torch.Tensor: 預測序列，形狀為 [batch_size, num_steps, output_dim]
        """
        # src shape: [batch_size, seq_len, input_dim]
        
        # 編碼器
        src_embedded = self.embedding(src)  # [batch_size, seq_len, hidden_dim]
        src_embedded = self.pos_encoder(src_embedded)
        memory = self.transformer_encoder(src_embedded)  # 生成 memory

        # 初始化解码器输入為最後一個T_CDU_out值
        # 獲取序列中最後一個T_CDU_out值的索引（假設T_CDU_out是第二個特徵）
        t_cdu_out_idx = 1  # 根據features列表中T_CDU_out的位置調整
        last_t_cdu_out = src[:, -1:, t_cdu_out_idx:t_cdu_out_idx+1]  # 獲取最後一個時間步的T_CDU_out值
        tgt = last_t_cdu_out  # 使用最後一個T_CDU_out值作為初始輸入
        
        # 解碼器
        outputs = []
        for _ in range(num_steps):
            tgt_embedded = self.decoder_embedding(tgt)  # [batch_size, tgt_len, hidden_dim]
            tgt_embedded = self.pos_decoder(tgt_embedded)
            
            # 修改 mask 生成邏輯
            if self.tgt_mask is None or self.tgt_mask.size(0) != tgt.size(1):
                device = tgt.device
                mask = self._generate_square_subsequent_mask(tgt.size(1)).to(device)
                self.tgt_mask = mask

            output = self.transformer_decoder(tgt_embedded, memory, tgt_mask=self.tgt_mask)
            
            # 輸出層
            output = self.fc(output)  # [batch_size, tgt_len, output_dim]
            outputs.append(output[:, -1, :])
            
            # 使用當前輸出做為下一步輸入
            tgt = torch.cat((tgt, output[:, -1:, :]), dim=1)
        
        return torch.stack(outputs, dim=1)