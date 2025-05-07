import torch
import torch.nn as nn
from model.iTransformer import iTransformer

class Model(nn.Module):
    """
    包裝iTransformer模型，使其與現有的系統兼容
    
    Args:
        configs: 配置對象，包含模型參數
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        # 從配置中獲取參數
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = getattr(configs, 'use_norm', True)
        
        # 初始化iTransformer模型
        self.model = iTransformer(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            input_dim=configs.input_dim if hasattr(configs, 'input_dim') else 8,  # 默認值為8個特徵
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            e_layers=configs.e_layers,
            d_ff=configs.d_ff if hasattr(configs, 'd_ff') else configs.d_model * 4,
            dropout=configs.dropout,
            activation=configs.activation,
            output_attention=configs.output_attention,
            use_norm=self.use_norm
        )
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        模型前向傳播
        
        Args:
            x_enc: 輸入序列 [Batch, Length, Channel]
            x_mark_enc: 輸入的時間戳記特徵，可選
            x_dec: 解碼器輸入，可選
            x_mark_dec: 解碼器輸入的時間戳記特徵，可選
        
        Returns:
            torch.Tensor: 預測結果
        """
        if self.output_attention:
            dec_out, attns = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], attns
        else:
            dec_out = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, pred_len, N]
    
    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        執行預測
        
        Args:
            x_enc: 輸入序列
            x_mark_enc: 輸入的時間戳記特徵，可選
            x_dec: 解碼器輸入，可選
            x_mark_dec: 解碼器輸入的時間戳記特徵，可選
            
        Returns:
            tuple: (預測結果, 注意力權重)
        """
        return self.model.forecast(x_enc, x_mark_enc) 