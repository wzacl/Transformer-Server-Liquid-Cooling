# !/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import os
import csv
import joblib
import seaborn as sns
import argparse
from datetime import datetime

# 檢查是否有可用的cuda設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
seq_length = 35 
num_steps = 8

# 讀取數據
training_data_dir = '/home/icmems/Documents/112033547/Training data'
testing_data_dir = '/home/icmems/Documents/112033547/Testing data'

# 讀取所有訓練數據
train_files = [os.path.join(training_data_dir, f) for f in os.listdir(training_data_dir) if f.endswith('.csv')]
test_files = [os.path.join(testing_data_dir, f) for f in os.listdir(testing_data_dir) if f.endswith('.csv')]

print(f"找到 {len(train_files)} 個訓練資料檔案")
print(f"找到 {len(test_files)} 個測試資料檔案")

# 讀取所有訓練資料
train_dfs = []
for file in train_files:
    df = pd.read_csv(file)
    train_dfs.append(df)
    print(f"已讀取訓練資料: {os.path.basename(file)}, 形狀: {df.shape}")

# 讀取所有測試資料
test_dfs = []
for file in test_files:
    df = pd.read_csv(file)
    test_dfs.append(df)
    print(f"已讀取測試資料: {os.path.basename(file)}, 形狀: {df.shape}")

# Define the CSV file to store results
results_file = '/home/icmems/Documents/112033547/Results/待分類/Train_results_含模型_1.5KW_新增PUMP資料_seq35/hyperparameter_results_all_future_steps.csv'
saving_folder='/home/icmems/Documents/112033547/Results/待分類/Train_results_含模型_1.5KW_新增PUMP資料_seq35'

# 選擇特徵和目標
features = ['T_GPU','T_CDU_out','T_CDU_in','T_env','T_air_out','fan_duty','pump_duty']
target = 'T_CDU_out'

# 進行數據預處理
# 建立歸一化類別
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# 建立歸一化映射區間
X_all = np.vstack([df[features].values for df in train_dfs])
y_all = np.vstack([df[[target]].values for df in train_dfs])
scaler_X.fit(X_all)
scaler_y.fit(y_all)

# 保存映射範圍
scaler_filename = f"{saving_folder}/Multi_step/1.5_1KWscalers.jlib"
os.makedirs(os.path.dirname(scaler_filename), exist_ok=True)
joblib.dump((scaler_X, scaler_y), scaler_filename)
print(f"映射範圍已保存至 {scaler_filename}")


# 分別處理兩組數據
def process_dataframe(df, features, target, scaler_X, scaler_y, seq_length):
    X = df[features].values
    y = df[[target]].values
    
    X_scaled = scaler_X.transform(X)  # 使用已經fit的scaler
    y_scaled = scaler_y.transform(y)  # 使用已經fit的scaler
    
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    y_tensor = torch.FloatTensor(y_scaled).to(device)
    
    # 單個數據集創建序列
    Xs, ys = [], []
    for i in range(len(X_tensor) - seq_length - num_steps + 1):
        Xs.append(X_tensor[i:(i + seq_length)])
        # 收集未來num_steps步的目標值
        future_targets = []
        for j in range(num_steps):
            future_targets.append(y_tensor[i + seq_length + j])
        ys.append(torch.stack(future_targets))
    return torch.stack(Xs), torch.stack(ys)


# 分別處理所有訓練數據集
train_data_list = []
for i, df in enumerate(train_dfs):
    X_train_seq, y_train_seq = process_dataframe(df, features, target, scaler_X, scaler_y, seq_length)
    train_data_list.append((X_train_seq, y_train_seq))
    print(f"已處理訓練資料 {i+1}/{len(train_dfs)}, 序列數量: {len(X_train_seq)}")

# 合併所有訓練數據序列以便評估
X_train_seq = torch.cat([seq[0] for seq in train_data_list], dim=0)
y_train_seq = torch.cat([seq[1] for seq in train_data_list], dim=0)
print(f"訓練資料總序列數: {len(X_train_seq)}")

# 處理所有測試數據集
test_data_list = []
for i, df in enumerate(test_dfs):
    X_test_seq, y_test_seq = process_dataframe(df, features, target, scaler_X, scaler_y, seq_length)
    test_data_list.append((X_test_seq, y_test_seq))
    print(f"已處理測試資料 {i+1}/{len(test_dfs)}, 序列數量: {len(X_test_seq)}")

# 確保至少有三個測試集以保持與原始代碼兼容
while len(test_data_list) < 3:
    print(f"警告: 測試數據集不足3個, 複製最後一個測試集來補足")
    test_data_list.append(test_data_list[-1])

X_test1_seq, y_test1_seq = test_data_list[0]
X_test2_seq, y_test2_seq = test_data_list[1]
X_test3_seq, y_test3_seq = test_data_list[2]

# 位置編碼類
class PositionalEncoding(nn.Module):
    """
    位置編碼模組，支持多種使用方式和編碼類型。
    
    參數:
        d_model (int): 嵌入維度
        max_len (int, optional): 最大序列長度，默認為5000
        encoding_type (str, optional): 位置編碼類型，可選 'fixed'（固定）或 'learned'（可學習），默認為'fixed'
        dropout (float, optional): Dropout比率，默認為0.1
        use_pe (bool, optional): 是否使用位置編碼，默認為True
    """
    def __init__(self, d_model, max_len=5000, encoding_type='fixed', dropout=0.1, use_pe=True):
        super(PositionalEncoding, self).__init__()
        self.use_pe = use_pe
        self.encoding_type = encoding_type
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        if encoding_type == 'fixed':
            # 使用傳統的固定位置編碼（sin/cos）
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        elif encoding_type == 'learned':
            # 使用可學習的位置編碼
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
        else:
            raise ValueError(f"不支持的編碼類型: {encoding_type}")

    def forward(self, x, position_ids=None):
        """
        前向傳播函數，添加位置編碼到輸入張量
        
        參數:
            x (torch.Tensor): 輸入張量，形狀為 [batch_size, seq_len, embedding_dim]
            position_ids (torch.Tensor, optional): 自定義位置索引，形狀為 [batch_size, seq_len]
        
        返回:
            torch.Tensor: 加入位置編碼後的張量
        """
        if not self.use_pe:
            return self.dropout(x)
            
        if self.encoding_type == 'fixed':
            if x.size(1) > self.pe.size(1):
                pe = self._extend_pe(x.size(1), x.device)
                return self.dropout(x + pe[:, :x.size(1), :])
            return self.dropout(x + self.pe[:, :x.size(1), :])
        elif self.encoding_type == 'learned':
            if x.size(1) > self.pe.size(1):
                # 擴展學習式位置編碼
                old_pe = self.pe
                self.pe = nn.Parameter(torch.randn(1, max(x.size(1), old_pe.size(1) * 2), self.d_model).to(x.device))
                with torch.no_grad():
                    self.pe[:, :old_pe.size(1), :] = old_pe
            return self.dropout(x + self.pe[:, :x.size(1), :])
    
    def _extend_pe(self, length, device):
        """擴展固定位置編碼到更長的序列"""
        pe = torch.zeros(1, length, self.d_model, device=device)
        position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * 
                           (-math.log(10000.0) / self.d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[:len(div_term)])
        return pe

# Transformer 模型
class TransformerModel(nn.Module):
    """
    Transformer模型，用於時間序列預測。
    
    參數:
        input_dim (int): 輸入維度（特徵數）
        hidden_dim (int): 隱藏層維度
        output_dim (int): 輸出維度
        num_encoder_layers (int): 編碼器層數
        num_decoder_layers (int): 解碼器層數
        num_heads (int): 注意力頭數
        dropout (float): Dropout比率
        use_pe (bool, optional): 是否使用位置編碼，默認為True
        encoding_type (str, optional): 位置編碼類型，可選 'fixed'或'learned'，默認為'fixed'
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_encoder_layers, num_decoder_layers, 
                 num_heads, dropout, use_pe=True, encoding_type='fixed'):
        super(TransformerModel, self).__init__()
        
        # 初始化 output_dim
        self.output_dim = output_dim
        
        # 編碼器部分
        self.embedding = DataEmbedding_inverted(
            seq_len=input_dim,
            d_model=hidden_dim,
            dropout=dropout,
            use_pe=use_pe,
            encoding_type=encoding_type
        )
        self.pos_encoder = PositionalEncoding(
            d_model=hidden_dim,
            dropout=dropout,
            use_pe=use_pe,
            encoding_type=encoding_type
        )
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        # 解碼器部分
        self.decoder_embedding = DataEmbedding_inverted(
            seq_len=output_dim,
            d_model=hidden_dim,
            dropout=dropout,
            use_pe=use_pe,
            encoding_type=encoding_type
        )
        self.pos_decoder = PositionalEncoding(
            d_model=hidden_dim,
            dropout=dropout,
            use_pe=use_pe,
            encoding_type=encoding_type
        )
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
        """生成一個方形的後續掩碼，用於解碼器自注意力層"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, src, num_steps, position_ids=None):
        """
        前向傳播函數
        
        參數:
            src (torch.Tensor): 源序列，形狀為 [batch_size, seq_len, input_dim]
            num_steps (int): 預測步數
            position_ids (torch.Tensor, optional): 自定義位置索引
            
        返回:
            torch.Tensor: 預測結果，形狀為 [batch_size, num_steps, output_dim]
        """
        # src shape: [batch_size, seq_len, input_dim]
        
        # 編碼器
        src_embedded = self.embedding(src, position_ids)  # [batch_size, seq_len, hidden_dim]
        memory = self.transformer_encoder(src_embedded)  # 生成 memory

        # 初始化解碼器輸入為最後一個T_CDU_out值
        # 獲取序列中最後一個T_CDU_out值的索引（T_CDU_out是第二個特徵）
        t_cdu_out_idx = features.index('T_CDU_out')
        last_t_cdu_out = src[:, -1:, t_cdu_out_idx:t_cdu_out_idx+1]  # 獲取最後一個時間步的T_CDU_out值
        tgt = last_t_cdu_out  # 使用最後一個T_CDU_out值作為初始輸入
        
        # 解碼器（自回歸模式 - Auto-regressive decoding）
        outputs = []
        for step in range(num_steps):
            # 檢查解碼器是否正確使用自回歸方式：
            # 1. tgt包含了之前所有的預測結果，形狀為[batch_size, current_len, output_dim]
            # 2. 每一步都使用之前的預測作為輸入，而不是一次性生成所有預測
            
            tgt_embedded = self.decoder_embedding(tgt, position_ids)  # [batch_size, tgt_len, hidden_dim]
            
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


# 逆向數據嵌入類 - 用於iTransformer
class DataEmbedding_inverted(nn.Module):
    """
    逆向數據嵌入類，用於iTransformer架構。
    將特徵維度視為token維度，進行嵌入處理。
    
    參數:
        seq_len (int): 序列長度
        d_model (int): 模型維度
        dropout (float, optional): Dropout比率，默認為0.1
        use_pe (bool, optional): 是否使用位置編碼，默認為True
        encoding_type (str, optional): 位置編碼類型，可選 'fixed'（固定）或 'learned'（可學習），默認為'fixed'
    """
    def __init__(self, seq_len, d_model, dropout=0.1, use_pe=True, encoding_type='fixed'):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(seq_len if seq_len is not None else 1, d_model)
        self.position_encoding = PositionalEncoding(
            d_model=d_model, 
            dropout=dropout, 
            use_pe=use_pe,
            encoding_type=encoding_type
        )
        self.dropout = nn.Dropout(p=dropout)
        self.use_pe = use_pe

    def forward(self, x, position_ids=None):
        """
        前向傳播函數
        
        參數:
            x (torch.Tensor): 輸入張量，形狀為 [batch_size, seq_len, input_dim]
            position_ids (torch.Tensor, optional): 自定義位置索引
            
        返回:
            torch.Tensor: 經過嵌入後的張量，形狀為 [batch_size, input_dim, d_model]
        """
        # x: [batch_size, seq_len, input_dim]
        # 轉置為 [batch_size, input_dim, seq_len]，其中input_dim作為tokens
        x = x.permute(0, 2, 1)
        
        # 對每個變量token進行嵌入
        x = self.value_embedding(x)  # [batch_size, input_dim, d_model]
        
        # 應用位置編碼
        x = self.position_encoding(x, position_ids)
        return x  # 位置編碼類中已經應用了dropout

# 初始化模型


class TransformerTrainer:
    def __init__(self, model, criterion, optimizer, num_epochs, batch_size, device, scaler_y):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.scaler_y = scaler_y
        self.train_losses = []

    def train(self, train_data_list):
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            total_batches = 0
            
            for X_train_seq, y_train_seq in train_data_list:
                for i in range(0, len(X_train_seq), self.batch_size):
                    batch_X = X_train_seq[i:i+self.batch_size].to(self.device)
                    batch_y = y_train_seq[i:i+self.batch_size].to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X, num_steps=batch_y.size(1))
                    
                    # 使用整個序列計算損失
                    loss = self.criterion(outputs, batch_y)  # outputs 和 batch_y 的形狀應該是 [batch_size, num_steps, output_dim]
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    total_batches += 1
            
            #avg_loss = total_loss / total_batches
            self.train_losses.append(total_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}],batch_size:{self.batch_size}, Loss: {total_loss:.4f}')
        
        return self.train_losses

    def evaluate(self, X_seq, y_seq):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_seq, num_steps=y_seq.size(1)).cpu().numpy()
            y_true = y_seq.cpu().numpy()
            
            # 将 predictions 从三维转换为二维
            batch_size, num_steps, output_dim = predictions.shape
            predictions_2d = predictions.reshape(-1, output_dim)
            
            # 反正規化
            predictions_2d = self.scaler_y.inverse_transform(predictions_2d)
            
            # 将 predictions 转换回三维
            predictions = predictions_2d.reshape(batch_size, num_steps, output_dim)
            
            y_true = self.scaler_y.inverse_transform(y_true.reshape(-1, 1)).reshape(batch_size, num_steps, 1)
            
            return y_true, predictions

    def calculate_metrics(self, y_true, y_pred):
        # 将三维数组转换为二维数组
        y_true_2d = y_true.reshape(-1, y_true.shape[-1])
        y_pred_2d = y_pred.reshape(-1, y_pred.shape[-1])
        
        mse = mean_squared_error(y_true_2d, y_pred_2d)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_2d, y_pred_2d)
        r2 = r2_score(y_true_2d, y_pred_2d)
        return mse, rmse, mae, r2

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def plot_predictions(self, true_values, predicted_values, title, filename):
        plt.figure(figsize=(12, 5))
        
        num_steps = predicted_values.shape[1]  # 取出有多少預測步
        
        # 注意：此處需要添加輸入序列的資料
        # 獲取測試資料的最後一個批次用於示範
        test_batch_size = min(100, len(true_values))  # 限制繪圖點數量，避免圖形過密
        
        # 定義時間步序列
        past_steps = seq_length
        future_steps = num_steps
        total_steps = past_steps + future_steps
        
        # 創建時間步座標軸
        timeline = np.arange(-past_steps+1, future_steps+1)
        current_step_idx = past_steps - 1  # 當前時間步的索引
        
        # 繪製過去的序列（假設）- 實際應用時應使用真實的過去序列資料
        past_data = np.linspace(true_values[0, 0, 0]*0.8, true_values[0, 0, 0], past_steps)
        future_true = np.array([true_values[0, i, 0] for i in range(future_steps)])
        
        # 獲取預測序列
        future_pred = np.array([predicted_values[0, i, 0] for i in range(future_steps)])
        
        # 繪製過去資料點 - 藍色
        plt.plot(timeline[:past_steps], past_data, 'o-', color='blue', label='History Data', linewidth=2)
        
        # 繪製未來實際值 - 藍色虛線圓點
        plt.plot(timeline[past_steps:], future_true, 'o--', color='blue', alpha=0.5, label='Actual Future Values')
        
        # 繪製預測值 - 紅色
        plt.plot(timeline[past_steps:], future_pred, 'o-', color='red', label='Predicted Future Values', linewidth=2)
        
        # 添加垂直線標記當前時間步
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.7)
        plt.text(0.1, plt.ylim()[1]*0.95, 'Current Time Step', fontsize=10)
        
        # 添加水平軸標籤
        x_ticks = timeline
        x_labels = []
        for t in x_ticks:
            if t == 0:
                x_labels.append("k")
            elif t < 0:
                x_labels.append(f"k{t}")
            else:
                x_labels.append(f"k+{t}")
        
        plt.xticks(x_ticks, x_labels)
        
        # 設置圖表標題和軸標籤
        plt.title(title)
        plt.xlabel('Time Step')
        plt.ylabel('Temperature (°C)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper left')
        
        # 保存圖表
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        # 繪製更詳細的分析圖 - 對所有測試數據
        plt.figure(figsize=(14, 6))
        
        # 繪製真實值和預測值的比較
        for step in range(num_steps):
            step_true = true_values[:test_batch_size, step, 0]
            step_pred = predicted_values[:test_batch_size, step, 0]
            
            plt.subplot(2, num_steps//2 + (1 if num_steps % 2 != 0 else 0), step+1)
            plt.scatter(step_true, step_pred, alpha=0.5)
            
            # 添加完美預測的對角線
            min_val = min(np.min(step_true), np.min(step_pred))
            max_val = max(np.max(step_true), np.max(step_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.title(f'Step {step+1} Prediction')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{filename}_step_analysis.jpg')
        plt.close()

    def calculate_trend_accuracy(self, y_true, y_pred):
        """計算趨勢方向預測準確率
        
        Args:
            y_true: 真實值，形狀為 [batch_size, num_steps, output_dim]
            y_pred: 預測值，形狀為 [batch_size, num_steps, output_dim]
            
        Returns:
            total_trend_accuracy: 總體趨勢方向準確率
            step_trend_accuracies: 每個預測步的趨勢方向準確率
        """
        batch_size, num_steps, _ = y_true.shape
        
        # 計算真實趨勢方向 (1:上升, 0:不變, -1:下降)
        true_trends = np.zeros((batch_size, num_steps-1))
        pred_trends = np.zeros((batch_size, num_steps-1))
        
        for i in range(num_steps-1):
            # 計算連續時間步之間的差異
            true_diff = y_true[:, i+1, 0] - y_true[:, i, 0]
            pred_diff = y_pred[:, i+1, 0] - y_pred[:, i, 0]
            
            # 確定趨勢方向
            true_trends[:, i] = np.sign(true_diff)
            pred_trends[:, i] = np.sign(pred_diff)
        
        # 計算方向預測正確的數量
        correct_predictions = (true_trends == pred_trends).astype(int)
        
        # 計算總體準確率
        total_correct = np.sum(correct_predictions)
        total_predictions = batch_size * (num_steps-1)
        total_trend_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
        
        # 計算每個步驟的準確率
        step_trend_accuracies = []
        for i in range(num_steps-1):
            step_correct = np.sum(correct_predictions[:, i])
            step_accuracy = step_correct / batch_size if batch_size > 0 else 0
            step_trend_accuracies.append(step_accuracy)
        
        return total_trend_accuracy, step_trend_accuracies

    def plot_trend_analysis(self, true_values, predicted_values, title, filename):
        """繪製趨勢分析圖
        
        Args:
            true_values: 真實值，形狀為 [batch_size, num_steps, output_dim]
            predicted_values: 預測值，形狀為 [batch_size, num_steps, output_dim]
            title: 圖表標題
            filename: 輸出文件名
        """
        plt.figure(figsize=(14, 10))
        
        # 提取樣本數據用於展示 (使用第一個批次)
        sample_true = true_values[0, :, 0]
        sample_pred = predicted_values[0, :, 0]
        num_steps = len(sample_true)
        
        # 計算樣本的趨勢方向
        true_trends = np.zeros(num_steps-1)
        pred_trends = np.zeros(num_steps-1)
        
        for i in range(num_steps-1):
            true_trends[i] = np.sign(sample_true[i+1] - sample_true[i])
            pred_trends[i] = np.sign(sample_pred[i+1] - sample_pred[i])
        
        # 绘制温度曲线和趋势
        plt.subplot(3, 1, 1)
        time_steps = np.arange(num_steps)
        plt.plot(time_steps, sample_true, 'b-o', label='Actual Values')
        plt.plot(time_steps, sample_pred, 'r-o', label='Predicted Values')
        plt.title(f'{title} - Temperature Comparison')
        plt.xlabel('Prediction Step')
        plt.ylabel('Temperature (°C)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 绘制趋势方向
        plt.subplot(3, 1, 2)
        trend_time_steps = np.arange(num_steps-1) + 0.5
        
        # 使用条形图表示趋势方向
        bar_width = 0.35
        plt.bar(trend_time_steps - bar_width/2, true_trends, bar_width, color='blue', alpha=0.6, label='Actual Trend')
        plt.bar(trend_time_steps + bar_width/2, pred_trends, bar_width, color='red', alpha=0.6, label='Predicted Trend')
        
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        plt.title('Temperature Trend Direction (1: Increasing, 0: Unchanged, -1: Decreasing)')
        plt.xlabel('Between Prediction Steps')
        plt.ylabel('Trend Direction')
        plt.xticks(np.arange(num_steps-1) + 0.5, [f'{i}-{i+1}' for i in range(num_steps-1)])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 计算趋势准确率
        trend_accuracy, step_accuracies = self.calculate_trend_accuracy(true_values, predicted_values)
        
        # 绘制每个步骤的趋势准确率
        plt.subplot(3, 1, 3)
        plt.bar(np.arange(len(step_accuracies)), step_accuracies, color='green', alpha=0.7)
        plt.axhline(y=trend_accuracy, color='red', linestyle='--', label=f'Overall Accuracy: {trend_accuracy:.2f}')
        plt.title('Trend Direction Prediction Accuracy by Step')
        plt.xlabel('Between Prediction Steps')
        plt.ylabel('Accuracy')
        plt.xticks(np.arange(len(step_accuracies)), [f'{i}-{i+1}' for i in range(len(step_accuracies))])
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{filename}_trend_analysis.jpg')
        plt.close()

    def plot_detailed_comparison(self, true_values, predicted_values, title, filename):
        """繪製預測結果與真實數據的詳細比較圖
        
        Args:
            true_values: 真實值，形狀為 [batch_size, num_steps, output_dim]
            predicted_values: 預測值，形狀為 [batch_size, num_steps, output_dim]
            title: 圖表標題
            filename: 輸出文件名
        """
        # 取得維度資訊
        batch_size, num_steps, _ = true_values.shape
        
        # 限制繪圖樣本數量，避免圖形過密
        plot_samples = min(200, batch_size)
        
        # 創建誤差數據
        errors = predicted_values[:plot_samples, :, 0] - true_values[:plot_samples, :, 0]
        
        # 計算每個時間步的預測指標
        step_metrics = []
        for step in range(num_steps):
            step_true = true_values[:plot_samples, step, 0]
            step_pred = predicted_values[:plot_samples, step, 0]
            step_errors = errors[:, step]
            
            # 計算每個步驟的指標
            step_mse = mean_squared_error(step_true, step_pred)
            step_rmse = np.sqrt(step_mse)
            step_mae = mean_absolute_error(step_true, step_pred)
            step_r2 = r2_score(step_true, step_pred)
            
            step_metrics.append({
                'mse': step_mse,
                'rmse': step_rmse,
                'mae': step_mae,
                'r2': step_r2,
                'mean_error': np.mean(step_errors),
                'std_error': np.std(step_errors)
            })
        
        # 1. 預測值與真實值散點圖比較（所有時間步）
        plt.figure(figsize=(15, 12))
        
        # 畫出每個時間步的散點圖
        for step in range(num_steps):
            plt.subplot(3, (num_steps+1)//3 + (1 if num_steps % 3 != 0 else 0), step+1)
            
            step_true = true_values[:plot_samples, step, 0]
            step_pred = predicted_values[:plot_samples, step, 0]
            
            # 散點圖
            plt.scatter(step_true, step_pred, alpha=0.5, s=10)
            
            # 添加完美預測的對角線
            min_val = min(np.min(step_true), np.min(step_pred))
            max_val = max(np.max(step_true), np.max(step_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # 計算並顯示相關係數
            correlation = np.corrcoef(step_true, step_pred)[0, 1]
            metrics = step_metrics[step]
            plt.title(f'Step {step+1}\nR²: {metrics["r2"]:.3f}, Corr: {correlation:.3f}')
            plt.xlabel('Actual Values (°C)')
            plt.ylabel('Predicted Values (°C)')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{filename}_scatter_comparison.jpg')
        plt.close()
        
        # 2. 誤差分析圖
        plt.figure(figsize=(15, 10))
        
        # 繪製誤差熱圖
        plt.subplot(2, 2, 1)
        sns_heatmap = plt.imshow(errors[:min(50, plot_samples)].T, aspect='auto', cmap='coolwarm')
        plt.colorbar(sns_heatmap, label='Error (°C)')
        plt.title('Prediction Error Heatmap (First 50 Samples)')
        plt.xlabel('Sample Index')
        plt.ylabel('Prediction Time Step')
        plt.yticks(np.arange(num_steps), [f'Step {i+1}' for i in range(num_steps)])
        
        # 繪製每個時間步的誤差分佈圖
        plt.subplot(2, 2, 2)
        
        # 使用小提琴圖顯示誤差分佈
        violin_data = [errors[:, step] for step in range(num_steps)]
        violin_positions = np.arange(1, num_steps+1)
        
        violin_parts = plt.violinplot(violin_data, positions=violin_positions, 
                                     showmeans=True, showmedians=True)
        
        # 設置小提琴圖的顏色
        for pc in violin_parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        
        # 添加水平線標記零誤差
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Error Distribution for Each Prediction Step')
        plt.xlabel('Prediction Time Step')
        plt.ylabel('Error (°C)')
        plt.grid(True, alpha=0.3)
        plt.xticks(violin_positions, [f'Step {i+1}' for i in range(num_steps)])
        
        # 繪製誤差隨時間變化的走勢
        plt.subplot(2, 2, 3)
        
        # 計算每個時間步的平均誤差和標準差
        mean_errors = np.array([metrics['mean_error'] for metrics in step_metrics])
        std_errors = np.array([metrics['std_error'] for metrics in step_metrics])
        
        # 繪製平均誤差線及其標準差範圍
        time_steps = np.arange(1, num_steps+1)
        plt.plot(time_steps, mean_errors, 'o-', label='Mean Error')
        plt.fill_between(time_steps, mean_errors - std_errors, mean_errors + std_errors, 
                        alpha=0.2, label='Standard Deviation Range')
        
        # 添加水平線標記零誤差
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Prediction Error Trend Over Time Steps')
        plt.xlabel('Prediction Time Step')
        plt.ylabel('Mean Error (°C)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 繪製每個時間步的性能指標
        plt.subplot(2, 2, 4)
        
        # 繪製MSE、RMSE和MAE
        metrics_to_plot = ['rmse', 'mae']
        colors = ['blue', 'green']
        
        for i, metric_name in enumerate(metrics_to_plot):
            metric_values = [metrics[metric_name] for metrics in step_metrics]
            plt.plot(time_steps, metric_values, 'o-', color=colors[i], label=metric_name.upper())
        
        # 繪製R²（使用次座標軸）
        ax2 = plt.twinx()
        r2_values = [metrics['r2'] for metrics in step_metrics]
        ax2.plot(time_steps, r2_values, 'o-', color='red', label='R²')
        ax2.set_ylabel('R² Score', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(-0.1, 1.1)  # R²的範圍通常在-∞到1之間，但在圖中限制範圍
        
        # 原坐標軸設置
        plt.title('Performance Metrics for Each Prediction Step')
        plt.xlabel('Prediction Time Step')
        plt.ylabel('Error Metrics')
        plt.grid(True, alpha=0.3)
        
        # 合併兩個坐標軸的圖例
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
        
        plt.tight_layout()
        plt.savefig(f'{filename}_error_analysis.jpg')
        plt.close()
        
        # 3. 所有樣本的時間序列圖（抽樣顯示）
        plt.figure(figsize=(15, 10))
        
        # 選擇要顯示的樣本數量
        num_samples_to_show = min(5, plot_samples)
        
        # 為每個樣本繪製時間序列圖
        for i in range(num_samples_to_show):
            plt.subplot(num_samples_to_show, 1, i+1)
            
            time_steps = np.arange(1, num_steps+1)
            sample_true = true_values[i, :, 0]
            sample_pred = predicted_values[i, :, 0]
            
            plt.plot(time_steps, sample_true, 'bo-', label='Actual Values')
            plt.plot(time_steps, sample_pred, 'ro-', label='Predicted Values')
            
            # 計算此樣本的指標
            sample_mae = mean_absolute_error(sample_true, sample_pred)
            sample_r2 = r2_score(sample_true, sample_pred)
            
            plt.title(f'Time Series Comparison for Sample {i+1} (MAE: {sample_mae:.3f}, R²: {sample_r2:.3f})')
            plt.xlabel('Prediction Time Step')
            plt.ylabel('Temperature (°C)')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{filename}_time_series_samples.jpg')
        plt.close()
        
        # 保存每個時間步的指標摘要
        with open(f'{filename}_step_metrics.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'MSE', 'RMSE', 'MAE', 'R²', 'Mean Error', 'Std Error'])
            
            for i, metrics in enumerate(step_metrics):
                writer.writerow([
                    i+1, 
                    metrics['mse'], 
                    metrics['rmse'], 
                    metrics['mae'], 
                    metrics['r2'], 
                    metrics['mean_error'], 
                    metrics['std_error']
                ])

    def save_metrics(self, folder_name, train_metrics, test_results, hyperparams, 
                     train_trend_accuracy, test_trend_results):
        metrics_text = (
            f"超參數:\n"
            f"seq_length: {hyperparams['seq_length']}\n"
            f"batch_size: {hyperparams['batch_size']}\n"
            f"input_dim: {hyperparams['input_dim']}\n"
            f"hidden_dim: {hyperparams['hidden_dim']}\n"
            f"output_dim: {hyperparams['output_dim']}\n"
            f"num_encoder_layers: {hyperparams['num_encoder_layers']}\n"
            f"num_decoder_layers: {hyperparams['num_decoder_layers']}\n"
            f"num_heads: {hyperparams['num_heads']}\n"
            f"dropout: {hyperparams['dropout']}\n"
            f"num_epochs: {hyperparams['num_epochs']}\n"
            f"use_pe: {hyperparams.get('use_pe', True)}\n"
            f"encoding_type: {hyperparams.get('encoding_type', 'fixed')}\n\n"
            "訓練集指標:\n"
            f"MSE: {train_metrics[0]:.4f}\n"
            f"RMSE: {train_metrics[1]:.4f}\n"
            f"MAE: {train_metrics[2]:.4f}\n"
            f"R2 分数: {train_metrics[3]:.4f}\n"
            f"趨勢準確率: {train_trend_accuracy:.4f}\n\n"
        )
        
        # 添加所有測試集指標
        for i, (test_name, _, _, metrics) in enumerate(test_results):
            _, trend_accuracy, _ = test_trend_results[i]
            metrics_text += (
                f"測試集 {test_name} 指標:\n"
                f"MSE: {metrics[0]:.4f}\n"
                f"RMSE: {metrics[1]:.4f}\n"
                f"MAE: {metrics[2]:.4f}\n"
                f"R2 分数: {metrics[3]:.4f}\n"
                f"趨勢準確率: {trend_accuracy:.4f}\n\n"
            )

        with open(f'{folder_name}/metrics.txt', 'w') as f:
            f.write(metrics_text)

    def ensure_folder_exists(self, folder_name):
        if not os.path.exists('Training_Pictures'):
            os.makedirs('Training_Pictures')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)



batch_size_list = [512,256]
hidden_dim_list = [16,32]  
output_dim = 1
num_encoder_layers_list = [1,2]
num_decoder_layers_list = [1,2]
num_heads_list = [2,4]
dropout_list = [0.005,0.001]
num_epoch_list = [200] 
input_dim = len(features)

# 确保目录存在
os.makedirs(os.path.dirname(results_file), exist_ok=True)

# Ensure the CSV file has headers if it does not already exist
if not os.path.exists(results_file):
    with open(results_file, mode='w', newline='') as file:
        # 基本列標題
        headers = ['seq_length', 'num_steps', 'batch_size', 'hidden_dim', 
                   'num_encoder_layers', 'num_decoder_layers', 'num_heads', 
                   'dropout', 'num_epochs', 'use_pe', 'encoding_type',
                   'train_MSE', 'train_RMSE', 'train_MAE', 'train_R2', 'train_trend_accuracy']
        
        # 根據測試集數量動態添加測試指標標題
        for i in range(len(test_data_list)):
            test_num = i + 1
            headers.extend([f'test{test_num}_MSE', f'test{test_num}_RMSE', 
                           f'test{test_num}_MAE', f'test{test_num}_R2', f'test{test_num}_trend_accuracy'])
        
        writer = csv.writer(file)
        writer.writerow(headers)
        print(f"創建了結果文件: {results_file}")
else:
    print(f"使用現有結果文件: {results_file}")

# 進行多個測試集的評估
def evaluate_all_datasets(trainer, test_data_list, folder_name, results_file, hyperparams):
    # 先評估訓練資料
    y_train_true, y_train_pred = trainer.evaluate(X_train_seq, y_train_seq)
    train_metrics = trainer.calculate_metrics(y_train_true, y_train_pred)
    
    # 計算訓練資料的趨勢準確率
    train_trend_accuracy, train_step_accuracies = trainer.calculate_trend_accuracy(y_train_true, y_train_pred)
    print(f"訓練集趨勢預測準確率: {train_trend_accuracy:.4f}")
    
    # 評估所有測試資料
    test_results = []
    test_trend_results = []
    
    for i, (X_test_seq, y_test_seq) in enumerate(test_data_list):
        test_name = f"test{i+1}"
        y_test_true, y_test_pred = trainer.evaluate(X_test_seq, y_test_seq)
        test_metrics = trainer.calculate_metrics(y_test_true, y_test_pred)
        test_results.append((test_name, y_test_true, y_test_pred, test_metrics))
        
        # 計算測試資料的趨勢準確率
        test_trend_accuracy, test_step_accuracies = trainer.calculate_trend_accuracy(y_test_true, y_test_pred)
        test_trend_results.append((test_name, test_trend_accuracy, test_step_accuracies))
        print(f"測試集 {test_name} 趨勢預測準確率: {test_trend_accuracy:.4f}")
        
        # 繪製並保存預測圖，使用英文標題
        trainer.plot_predictions(
            y_test_true, 
            y_test_pred, 
            f'CDU Outlet Temperature Prediction (Test Set {i+1})', 
            f'{folder_name}/2KWCDU_prediction_{test_name}.jpg'
        )
        
        # 繪製並保存趨勢分析圖
        trainer.plot_trend_analysis(
            y_test_true,
            y_test_pred,
            f'CDU Outlet Temperature Trend Analysis (Test Set {i+1})',
            f'{folder_name}/2KWCDU_trend_{test_name}'
        )
        
        # 繪製並保存詳細比較圖
        trainer.plot_detailed_comparison(
            y_test_true,
            y_test_pred,
            f'CDU Outlet Temperature Detailed Comparison (Test Set {i+1})',
            f'{folder_name}/2KWCDU_detailed_{test_name}'
        )
    
    # 繪製並保存訓練集預測圖，使用英文標題
    trainer.plot_predictions(
        y_train_true, 
        y_train_pred, 
        'CDU Outlet Temperature Prediction (Training Set)', 
        f'{folder_name}/2KWCDU_prediction_train.jpg'
    )
    
    # 繪製並保存訓練集趨勢分析圖
    trainer.plot_trend_analysis(
        y_train_true,
        y_train_pred,
        'CDU Outlet Temperature Trend Analysis (Training Set)',
        f'{folder_name}/2KWCDU_trend_train'
    )
    
    # 繪製並保存訓練集詳細比較圖
    trainer.plot_detailed_comparison(
        y_train_true,
        y_train_pred,
        'CDU Outlet Temperature Detailed Comparison (Training Set)',
        f'{folder_name}/2KWCDU_detailed_train'
    )
    
    # 保存評估指標到文本文件，加入趨勢準確率
    save_all_metrics(folder_name, train_metrics, test_results, hyperparams, 
                     train_trend_accuracy, test_trend_results)
    
    # 將結果添加到CSV文件，加入趨勢準確率
    save_results_to_csv(results_file, hyperparams, train_metrics, test_results,
                       train_trend_accuracy, test_trend_results)
    
    return train_metrics, test_results, train_trend_accuracy, test_trend_results

# 保存所有指標
def save_all_metrics(folder_name, train_metrics, test_results, hyperparams, 
                    train_trend_accuracy, test_trend_results):
    metrics_text = (
        f"超參數:\n"
        f"seq_length: {hyperparams['seq_length']}\n"
        f"batch_size: {hyperparams['batch_size']}\n"
        f"input_dim: {hyperparams['input_dim']}\n"
        f"hidden_dim: {hyperparams['hidden_dim']}\n"
        f"output_dim: {hyperparams['output_dim']}\n"
        f"num_encoder_layers: {hyperparams['num_encoder_layers']}\n"
        f"num_decoder_layers: {hyperparams['num_decoder_layers']}\n"
        f"num_heads: {hyperparams['num_heads']}\n"
        f"dropout: {hyperparams['dropout']}\n"
        f"num_epochs: {hyperparams['num_epochs']}\n"
        f"use_pe: {hyperparams.get('use_pe', True)}\n"
        f"encoding_type: {hyperparams.get('encoding_type', 'fixed')}\n\n"
        "訓練集指標:\n"
        f"MSE: {train_metrics[0]:.4f}\n"
        f"RMSE: {train_metrics[1]:.4f}\n"
        f"MAE: {train_metrics[2]:.4f}\n"
        f"R2 分数: {train_metrics[3]:.4f}\n"
        f"趨勢準確率: {train_trend_accuracy:.4f}\n\n"
    )
    
    # 添加所有測試集指標
    for i, (test_name, _, _, metrics) in enumerate(test_results):
        _, trend_accuracy, _ = test_trend_results[i]
        metrics_text += (
            f"測試集 {test_name} 指標:\n"
            f"MSE: {metrics[0]:.4f}\n"
            f"RMSE: {metrics[1]:.4f}\n"
            f"MAE: {metrics[2]:.4f}\n"
            f"R2 分数: {metrics[3]:.4f}\n"
            f"趨勢準確率: {trend_accuracy:.4f}\n\n"
        )

    with open(f'{folder_name}/metrics.txt', 'w') as f:
        f.write(metrics_text)

# 保存結果到CSV
def save_results_to_csv(results_file, hyperparams, train_metrics, test_results,
                       train_trend_accuracy, test_trend_results):
    # 格式化CSV行
    row = [
        hyperparams['seq_length'], 
        hyperparams['num_steps'], 
        hyperparams['batch_size'], 
        hyperparams['hidden_dim'], 
        hyperparams['num_encoder_layers'], 
        hyperparams['num_decoder_layers'], 
        hyperparams['num_heads'], 
        hyperparams['dropout'], 
        hyperparams['num_epochs'],
        hyperparams.get('use_pe', True),
        hyperparams.get('encoding_type', 'fixed'),
        # 訓練指標
        train_metrics[0], train_metrics[1], train_metrics[2], train_metrics[3],
        train_trend_accuracy  # 添加訓練趨勢準確率
    ]
    
    # 添加所有測試集的指標
    for i, (_, _, _, metrics) in enumerate(test_results):
        _, trend_accuracy, _ = test_trend_results[i]
        row.extend([metrics[0], metrics[1], metrics[2], metrics[3], trend_accuracy])
    
    # 寫入CSV
    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

# Training loop to search for optimal hyperparameters
for batch_size in batch_size_list:
    for hidden_dim in hidden_dim_list:
        for num_encoder_layers in num_encoder_layers_list:
            for num_decoder_layers in num_decoder_layers_list:
                for num_heads in num_heads_list:
                    for dropout in dropout_list:
                        for num_epochs in num_epoch_list:
                            # 模型參數初始化 - 增加位置編碼相關參數
                            use_pe = True  # 預設使用位置編碼
                            encoding_type = 'fixed'  # 預設使用固定位置編碼
                            
                            model = TransformerModel(
                                input_dim=input_dim, 
                                hidden_dim=hidden_dim, 
                                output_dim=output_dim, 
                                num_encoder_layers=num_encoder_layers, 
                                num_decoder_layers=num_decoder_layers, 
                                num_heads=num_heads, 
                                dropout=dropout,
                                use_pe=use_pe,
                                encoding_type=encoding_type
                            ).to(device)
                            
                            # 使用MAE作為損失函數 
                            criterion = nn.L1Loss(reduction='sum')
                            optimizer = optim.Adam(model.parameters())

                            trainer = TransformerTrainer(
                                model=model,
                                criterion=criterion,
                                optimizer=optimizer,
                                num_epochs=num_epochs,
                                batch_size=batch_size,
                                device=device,
                                scaler_y=scaler_y
                            )
                            train_losses = trainer.train(train_data_list)

                            # Create a folder for the current set of hyperparameters
                            folder_name = (f"{saving_folder}/multi_seq{seq_length}_steps{num_steps}_batch{batch_size}_"
                                        f"hidden{hidden_dim}_encoder{num_encoder_layers}_decoder{num_decoder_layers}_"
                                        f"heads{num_heads}_dropout{dropout}_epoch{num_epochs}_"
                                        f"use_pe{use_pe}_encoding{encoding_type}")
                            trainer.ensure_folder_exists(folder_name)
                            
                            # 指定保存模型的文件夾
                            model_save_path = f"{folder_name}/2KWCDU_Transformer_model.pth"

                            # 確保文件夾存在
                            if not os.path.exists(folder_name):
                                os.makedirs(folder_name)

                            # 保存模型
                            torch.save(model.state_dict(), model_save_path)
                            print(f"模型已保存至 {model_save_path}")

                            # 收集超參數
                            hyperparams = {
                                'seq_length': seq_length,
                                'num_steps': num_steps,
                                'batch_size': batch_size,
                                'input_dim': input_dim,
                                'hidden_dim': hidden_dim,
                                'output_dim': output_dim,
                                'num_encoder_layers': num_encoder_layers,
                                'num_decoder_layers': num_decoder_layers,
                                'num_heads': num_heads,
                                'dropout': dropout,
                                'num_epochs': num_epochs,
                                'use_pe': use_pe,
                                'encoding_type': encoding_type
                            }

                            # 評估所有數據集
                            _ = evaluate_all_datasets(
                                trainer, 
                                test_data_list, 
                                folder_name, 
                                results_file, 
                                hyperparams
                            )
                            
                            print(f"完成模型訓練和評估")

print(f"所有結果已保存至 {results_file}")


