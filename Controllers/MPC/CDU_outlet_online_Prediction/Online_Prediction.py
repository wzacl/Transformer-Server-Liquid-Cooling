# /usr/bin/python3
import time
import sys
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Control_Unit')

import ADAMScontroller
import pwmcontroller as ctrl
import multi_channel_pwmcontroller as multi_ctrl
from simple_pid import PID
import matplotlib.pyplot as plt
import numpy as np
import joblib
import torch
import torch.nn as nn
from collections import deque
import math
from sklearn.preprocessing import MinMaxScaler
import os

adam_port = '/dev/ttyUSB0'
fan1_port = '/dev/ttyAMA4'
fan2_port = '/dev/ttyAMA5'
pump_port = '/dev/ttyAMA3'

#設置實驗資料放置的資料夾
exp_name = '/home/inventec/Desktop/2KWCDU_修改版本/data_manage/Real_time_Prediction_data'
#設置實驗資料檔案名稱
exp_var = 'Real_time_Prediction_data_GPU15KW_1(285V_8A)'
#設置實驗資料標題
custom_headers = ['time', 'T_GPU', 'T_heater', 'T_CDU_in', 'T_CDU_out', 'T_env', 'T_air_in', 'T_air_out', 'TMP8', 'fan_duty', 'pump_duty', 'GPU_Watt(KW)']

adam = ADAMScontroller.DataAcquisition(exp_name, exp_var, port=adam_port, csv_headers=custom_headers)
fan1 = multi_ctrl.multichannel_PWMController(fan1_port)
fan2 = multi_ctrl.multichannel_PWMController(fan2_port)
pump = ctrl.XYKPWMController(pump_port)

print('模型初始化.....')

# 修改模型和scaler路徑
model_path = '/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/2KWCDU_Transformer_model.pth'
# 設定MinMaxScaler的路徑，此scaler用於將輸入數據歸一化到[0,1]區間
# 該scaler是在訓練模型時保存的，確保預測時使用相同的數據縮放方式
scaler_path = '/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/1.5_1KWscalers.jlib' 



# 設置初始轉速
pump_duty=40
pump.set_duty_cycle(pump_duty)
fan_duty=60
fan1.set_all_duty_cycle(fan_duty)
fan2.set_all_duty_cycle(fan_duty)

# 設置ADAM控制器
adam.start_adam()
adam.update_duty_cycles(fan_duty, pump_duty)

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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout):
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
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, src, num_steps):
        # src shape: [batch_size, seq_len, input_dim]
        
        # 編碼器
        src_embedded = self.embedding(src)  # [batch_size, seq_len, hidden_dim]
        src_embedded = self.pos_encoder(src_embedded)
        memory = self.transformer_encoder(src_embedded)  # 生成 memory

        # 初始化解码器输入为全零，确保形状與 src 的 batch_size 匹配
        tgt = torch.zeros(src.size(0), 1, self.output_dim).to(src.device)
        
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

# 加載模型和scaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
model = TransformerModel(input_dim=7, hidden_dim=8, output_dim=1, num_layers=1, num_heads=8, dropout=0.01)
model.load_state_dict(model_state_dict)
model.eval()

# 使用 joblib 加載訓練好的 scaler
scaler = joblib.load(scaler_path)

# 確保 scaler 是 MinMaxScaler 的實例
if isinstance(scaler, dict):
    raise ValueError("Loaded scaler is a dictionary, expected MinMaxScaler instance. Please check the scaler file.")

# 創建數據緩存
time_window = 20  # 時間窗口大小
history_buffer = deque(maxlen=time_window)
features = ['T_GPU', 'T_CDU_in', 'T_env', 'T_air_in', 'T_air_out', 'fan_duty', 'pump_duty']

# 修改預測數據記錄結構
prediction_data = {
    'timestamps': [],
    'actual_temps': [],
    'predicted_sequence': []  # 儲存8個時間步的預測
}

'''''

# 初始化繪圖
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
line1, = ax.plot([], [], 'r-', label='實際溫度')
line2, = ax.plot([], [], 'b--', label='預測溫度')
ax.set_xlabel('時間 (s)')
ax.set_ylabel('溫度 (°C)')
ax.legend()
'''''
def get_current_features(data):
    """獲取當前時間步的特徵"""
    # 確保輸入數據長度為7個特徵
    if len(data) != 7:
        raise ValueError(f"輸入數據必須包含7個特徵,當前數據長度為:{len(data)}")
    
    # 將單個時間步的數據轉換為2D數組並進行縮放
    data_2d = np.array(data).reshape(1, -1)
    if isinstance(scaler, tuple):
        scaled_data = scaler[0].transform(data_2d)
    else:
        scaled_data = scaler.transform(data_2d)
    
    return scaled_data[0]

def prepare_sequence_data(history_buffer):
    if len(history_buffer) != 20:
        print(f"歷史數據長度錯誤: {len(history_buffer)}，需要 20 個時間步")
        return None
    sequence = np.array(list(history_buffer))
    if sequence.shape != (20, 7):
        print(f"數據形狀錯誤: {sequence.shape}，需要 (20, 7)")
        return None
    return torch.FloatTensor(sequence).unsqueeze(0).to(device)

def inverse_transform_predictions(scaled_predictions):
    # 確保 predictions 為 2D 陣列 (8,1)
    if len(scaled_predictions.shape) == 1:
        scaled_predictions = scaled_predictions.reshape(-1, 1)

    # 確保 `scaler` 為 `tuple` 時只使用 `scaler_y`
    if isinstance(scaler, tuple):
        return scaler[1].inverse_transform(scaled_predictions)[:, 0]
    else:
        return scaler.inverse_transform(scaled_predictions)[:, 0]


def update_plot():
    """更新實時溫度曲線"""
    if len(prediction_data['timestamps']) > 0:
        time_diff = [t - prediction_data['timestamps'][0] for t in prediction_data['timestamps']]
        
        # 更新實際溫度曲線
        line1.set_data(time_diff, prediction_data['actual_temps'])
        
        # 更新預測溫度曲線（顯示未來8步預測）
        if len(prediction_data['predicted_sequence']) > 0:
            last_time = time_diff[-1]
            pred_times = [last_time + i*5 for i in range(8)]  # 假設每步5秒
            line2.set_data(pred_times, prediction_data['predicted_sequence'][-1])
        
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.1)

# 主循环
while True:
    try:
        # 獲取當前數據並確保有7個特徵
        data = [
                adam.buffer[0],  # T_GPU
                adam.buffer[2],  # T_CDU_in
                adam.buffer[4],  # T_env
                adam.buffer[5],  # T_air_in
                adam.buffer[6],  # T_air_out
                adam.buffer[8],  # fan_duty
                adam.buffer[9]  # pump_duty
            ]


        # 更新歷史緩存 (存入已縮放的數據)
        current_features = get_current_features(data)
        history_buffer.append(current_features)
        
        # 準備輸入數據 (數據已經過縮放)
        input_tensor = prepare_sequence_data(history_buffer)
        
        # 當歷史數據足夠時進行預測
        if input_tensor is not None:
            # 執行預測
            with torch.no_grad():
                scaled_predictions = model(input_tensor, num_steps=8)[0].cpu().numpy()
                print(f"scaled_predictions 形狀: {scaled_predictions.shape}")

            
            # 將預測結果轉換回原始範圍
            predicted_sequence = inverse_transform_predictions(scaled_predictions)
            
            # 記錄結果
            current_time = time.time()
            prediction_data['timestamps'].append(current_time)
            prediction_data['actual_temps'].append(data[4])  # T_CDU_out 位於索引3
            prediction_data['predicted_sequence'].append(predicted_sequence)
            
            # 更新图表
            #update_plot()
            
            # 打印預測結果
            print(f"當前溫度: {data[3]:.2f}°C")
            print(f"未來8步預測溫度: {predicted_sequence}")
        
        time.sleep(1)  # 控制採樣頻率

    except Exception as e:
        print(f"預測錯誤: {str(e)}")
        time.sleep(1)




