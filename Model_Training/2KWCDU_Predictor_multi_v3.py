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
# 檢查是否有可用的cuda設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
seq_length = 20
num_steps = 8


# 讀取數據
df1 = pd.read_csv('Training data/10.28_TrainingData_GPU1KW(218V_8A)_Pump[40-100].csv')
df2 = pd.read_csv('Training data/11.04_TrainingData_GPU1.5KW(285V_8A)_Pump[40-100].csv')
df3 = pd.read_csv('Testing data/11.06_Testingdata_GPU1.5KW(285V_8A).csv')
df4 = pd.read_csv('Testing data/11.04_Testingdata_GPU1KW(218V_8A).csv')

# Define the CSV file to store results
results_file = 'Train_results_含模型_1.5KW_and_1KW/Multi_step/hyperparameter_results_2.csv'
saving_folder='Train_results_含模型_1.5KW_and_1KW'

# 選擇特徵和目標
features = ['T_GPU','T_heater','T_CDU_in','T_env','T_air_in','T_air_out','fan_duty','pump_duty','GPU_Watt(KW)']
target = 'T_CDU_out'
#進行數據預處理
# 建立歸一化類別
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
# 建立歸一化映射區間
X_all = np.vstack((df1[features].values, df2[features].values))
y_all = np.vstack((df1[[target]].values, df2[[target]].values))
scaler_X.fit(X_all)
scaler_y.fit(y_all)


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
    for i in range(len(X_tensor) - seq_length):
        Xs.append(X_tensor[i:(i + seq_length)])
        ys.append(y_tensor[i + seq_length])
    return torch.stack(Xs), torch.stack(ys)


# 分別處理兩個訓練數據集
X_train_seq1, y_train_seq1 = process_dataframe(df1, features, target, scaler_X, scaler_y, seq_length)
X_train_seq2, y_train_seq2 = process_dataframe(df2, features, target, scaler_X, scaler_y, seq_length)

# 合併訓練數據(數據在前面已包成多個batch因此不會造成錯誤的序列關係)
X_train_seq = torch.cat([X_train_seq1, X_train_seq2], dim=0)
y_train_seq = torch.cat([y_train_seq1, y_train_seq2], dim=0)

# 處理測試數據集
X_test1_seq, y_test1_seq = process_dataframe(df3, features, target, scaler_X, scaler_y, seq_length)
X_test2_seq, y_test2_seq = process_dataframe(df4, features, target, scaler_X, scaler_y, seq_length)
#X_test3_seq, y_test3_seq = process_dataframe(df5, features, target, scaler_X, scaler_y, seq_length)



# 位置編碼類
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout):
        super(TransformerModel, self).__init__()
        
        # 初始化 output_dim
        self.output_dim = output_dim
        
        # 編碼器部分
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 解碼器部分
        self.decoder_embedding = nn.Linear(output_dim, hidden_dim)
        self.pos_decoder = PositionalEncoding(hidden_dim)
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        
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
        src_embedded = src_embedded.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]
        src_embedded = self.pos_encoder(src_embedded)
        memory = self.transformer_encoder(src_embedded)  # 生成 memory

        # 初始化解码器输入为全零，确保形状与 src 的 batch_size 匹配
        tgt = torch.zeros(src.size(0), 1, self.output_dim).to(src.device)
        
        # 解碼器
        outputs = []
        for _ in range(num_steps):
            tgt_embedded = self.decoder_embedding(tgt)  # [batch_size, tgt_len, hidden_dim]
            tgt_embedded = tgt_embedded.permute(1, 0, 2)  # [tgt_len, batch_size, hidden_dim]
            tgt_embedded = self.pos_decoder(tgt_embedded)
            
            # 确保 tgt_mask 的形状与 tgt_embedded 的形状匹配
            if self.tgt_mask is None or self.tgt_mask.size(0) != tgt_embedded.size(0):
                device = tgt.device
                mask = self._generate_square_subsequent_mask(tgt_embedded.size(0)).to(device)
                self.tgt_mask = mask

            output = self.transformer_decoder(tgt_embedded, memory, tgt_mask=self.tgt_mask)
            output = output.permute(1, 0, 2)  # [batch_size, tgt_len, hidden_dim]
            
            # 輸出層
            output = self.fc(output)  # [batch_size, tgt_len, output_dim]
            outputs.append(output[:, -1, :])
            
            # 使用當前輸出做為下一步輸入
            tgt = torch.cat((tgt, output[:, -1:, :]), dim=1)
        
        return torch.stack(outputs, dim=1)

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
                    loss = self.criterion(outputs[:, -1, :], batch_y)
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    total_batches += 1
            
            avg_loss = total_loss / total_batches
            self.train_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}')
        
        return self.train_losses

    def evaluate(self, X_seq, y_seq):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_seq, num_steps=y_seq.size(1))[:, -1, :].cpu().numpy()
            y_true = y_seq.cpu().numpy()
            
            # 反正規化
            predictions = self.scaler_y.inverse_transform(predictions)
            y_true = self.scaler_y.inverse_transform(y_true)
            
            return y_true, predictions

    def calculate_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
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
        for i in range(predicted_values.shape[1]):  # 對每個預測步長進行繪制
            plt.plot(predicted_values[:, i], label=f'Predicted Step {i+1}')
        plt.plot(true_values, label='True Values', color='black', linewidth=2)
        plt.title(title)
        plt.xlabel('time(s)')
        plt.ylabel('Temperature(℃)')
        plt.legend()
        plt.savefig(filename)
        plt.close()

    def save_metrics(self, folder_name, train_metrics, test1_metrics, test2_metrics):
        metrics_text = (
            f"超參數:\n"
            f"seq_length: {seq_length}\n"
            f"batch_size: {batch_size}\n"
            f"input_dim: {input_dim}\n"
            f"hidden_dim: {hidden_dim}\n"
            f"output_dim: {output_dim}\n"
            f"num_layers: {num_layers}\n"
            f"num_heads: {num_heads}\n"
            f"dropout: {dropout}\n\n"
            f"num_epochs: {num_epochs}\n\n"
            "訓練集指標:\n"
            f"MSE: {train_metrics[0]:.4f}\n"
            f"RMSE: {train_metrics[1]:.4f}\n"
            f"MAE: {train_metrics[2]:.4f}\n"
            f"R2 分数: {train_metrics[3]:.4f}\n\n"
            "測試集1指標:\n"
            f"MSE: {test1_metrics[0]:.4f}\n"
            f"RMSE: {test1_metrics[1]:.4f}\n"
            f"MAE: {test1_metrics[2]:.4f}\n"
            f"R2 分数: {test1_metrics[3]:.4f}\n\n"
            "測試集2指標:\n"
            f"MSE: {test2_metrics[0]:.4f}\n"
            f"RMSE: {test2_metrics[1]:.4f}\n"
            f"MAE: {test2_metrics[2]:.4f}\n"
            f"R2 分数: {test2_metrics[3]:.4f}\n"
        )

        with open(f'{folder_name}/metrics.txt', 'w') as f:
            f.write(metrics_text)

    def ensure_folder_exists(self, folder_name):
        if not os.path.exists('Training_Pictures'):
            os.makedirs('Training_Pictures')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)



batch_size_list = [512]
hidden_dim_list = [8]  
output_dim = 1
num_layers = 1
num_heads = 8
dropout_list = [0.01]
num_epoch_list = [300] 
input_dim = len(features)
# 包裝訓練資料
train_data_list = [
    (X_train_seq1, y_train_seq1),
    (X_train_seq2, y_train_seq2)
]





# 确保目录存在
os.makedirs(os.path.dirname(results_file), exist_ok=True)

# Ensure the CSV file has headers if it does not already exist
if not os.path.exists(results_file):
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['seq_length', 'num_steps', 'batch_size', 'hidden_dim', 'num_layers', 'num_heads', 'dropout', 'num_epochs', 
                         'train_MSE', 'train_RMSE', 'train_MAE', 'train_R2',
                         'test1_MSE', 'test1_RMSE', 'test1_MAE', 'test1_R2',
                         'test2_MSE', 'test2_RMSE', 'test2_MAE', 'test2_R2'])

# Training loop to search for optimal hyperparameters
for batch_size in batch_size_list:
    for hidden_dim in hidden_dim_list:
        for dropout in dropout_list:
            for num_epochs in num_epoch_list:
                model = TransformerModel(input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout).to(device)
                criterion = nn.MSELoss()
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
                folder_name = (f"{saving_folder}/Multi_step/multi_seq{seq_length}_steps{num_steps}_batch{batch_size}_"
                               f"hidden{hidden_dim}_layers{num_layers}_"
                               f"heads{num_heads}_dropout{dropout}_epoch{num_epochs}")
                trainer.ensure_folder_exists(folder_name)
                # 指定保存模型的文件夾
                model_save_path = f"{folder_name}/2KWCDU_Transformer_model.pth"

                # 確保文件夾存在
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

                # 保存模型
                torch.save(model, model_save_path)
                print(f"模型已保存至 {model_save_path}")


                # Evaluate model
                y_train_true, y_train_pred = trainer.evaluate(X_train_seq, y_train_seq)
                y_test1_true, y_test1_pred = trainer.evaluate(X_test1_seq, y_test1_seq)
                y_test2_true, y_test2_pred = trainer.evaluate(X_test2_seq, y_test2_seq)

                # Calculate evaluation metrics
                train_metrics = trainer.calculate_metrics(y_train_true, y_train_pred)
                test1_metrics = trainer.calculate_metrics(y_test1_true, y_test1_pred)
                test2_metrics = trainer.calculate_metrics(y_test2_true, y_test2_pred)



                # Save metrics to text file
                trainer.save_metrics(folder_name, train_metrics, test1_metrics, test2_metrics)

                # Plot and save predictions
                trainer.plot_predictions(y_train_true, y_train_pred, 'CDU Outlet Temperature Prediction (Training Set)', f'{folder_name}/2KWCDU_prediction_train.jpg')
                trainer.plot_predictions(y_test1_true, y_test1_pred, 'CDU Outlet Temperature Prediction (Test Set 1)', f'{folder_name}/2KWCDU_prediction_test1.jpg')
                trainer.plot_predictions(y_test2_true, y_test2_pred, 'CDU Outlet Temperature Prediction (Test Set 2)', f'{folder_name}/2KWCDU_prediction_test2.jpg')

                # Append results to CSV
                with open(results_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([seq_length, num_steps, batch_size, hidden_dim, num_layers, num_heads, dropout, num_epochs,
                                     train_metrics[0], train_metrics[1], train_metrics[2], train_metrics[3],
                                     test1_metrics[0], test1_metrics[1], test1_metrics[2], test1_metrics[3],
                                     test2_metrics[0], test2_metrics[1], test2_metrics[2], test2_metrics[3]])

                # 保存scaler
                scaler_save_path = f"{folder_name}/minmax_scaler.pkl"
                joblib.dump(scaler_X, scaler_save_path)

print(f"All results have been saved to {results_file}")






