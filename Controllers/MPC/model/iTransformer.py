import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.Transformer_EncDec import Encoder, EncoderLayer
from model.layers.SelfAttention_Family import FullAttention, AttentionLayer
from model.layers.Embed import DataEmbedding_inverted
import numpy as np


class iTransformer(nn.Module):
    """
    iTransformer: 倒置Transformer模型用於時間序列預測

    論文連結: https://arxiv.org/abs/2310.06625

    Args:
        seq_len (int): 輸入序列長度
        pred_len (int): 預測長度
        input_dim (int): 輸入特徵維度
        d_model (int): 模型隱藏維度
        n_heads (int): 多頭注意力的頭數
        e_layers (int): 編碼器層數
        d_ff (int): 前饋神經網絡的隱藏層維度
        dropout (float): Dropout比率
        activation (str): 激活函數類型，'relu'或'gelu'
        output_attention (bool): 是否輸出注意力權重
        use_norm (bool): 是否使用層歸一化
    """

    def __init__(self, 
                 seq_len, 
                 pred_len, 
                 input_dim,
                 d_model=512, 
                 n_heads=8, 
                 e_layers=3, 
                 d_ff=512, 
                 dropout=0.0, 
                 activation='gelu', 
                 output_attention=False,
                 use_norm=True):
        super(iTransformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.use_norm = use_norm
        self.input_dim = input_dim
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, 'fixed', 'h', dropout)
        
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=5, attention_dropout=dropout,
                                      output_attention=output_attention), 
                        d_model, 
                        n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Projection head
        self.projector = nn.Linear(d_model, pred_len, bias=True)
        
        # Output layer maps back to original feature dimension
        self.output_layer = nn.Linear(input_dim, input_dim)


        
    def forecast(self, x_enc, x_mark_enc=None):
        """
        執行預測

        Args:
            x_enc (torch.Tensor): 輸入序列 [Batch, Length, Channel]
            x_mark_enc (torch.Tensor): 輸入的時間戳記特徵，可選

        Returns:
            tuple: (預測結果 [Batch, pred_len, input_dim], 注意力權重)
        """
        # Save original dimensions
        B, L, N = x_enc.shape  # B: batch_size; L: seq_len; N: number of variates
        
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Embedding: B L N -> B N E  (inverted compared to vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # Encoder: B N E -> B N E  (inverted attention and processing)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Projection: B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)
        
        # Apply final output layer (optional)
        # dec_out = self.output_layer(dec_out)
        
        if self.use_norm:
            # De-Normalization
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            

        # 移除了ste_round操作，保持原始預測值
        # 不再對T_CDU_out特徵進行四捨五入處理
        # 直接返回原始預測結果和注意力權重

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        模型前向傳播

        Args:
            x_enc: 輸入序列
            x_mark_enc: 輸入的時間戳記特徵，可選
            x_dec: 解碼器輸入（可為未來步長的佔位），可選
            x_mark_dec: 解碼器輸入的時間戳記特徵，可選

        Returns:
            torch.Tensor 或 tuple: 預測結果或(預測結果, 注意力權重)
        """
        dec_out, attns = self.forecast(x_enc, x_mark_enc)
        
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out  # [B, pred_len, N] 