import time
import sys
sys.path.append('/home/inventec/Desktop/2KWCDU/code_manage/Control_Unit')

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
import pickle  # 用于加载保存的scaler

adam_port = '/dev/ttyUSB0'
fan1_port = '/dev/ttyAMA4'
fan2_port = '/dev/ttyAMA5'
pump_port = '/dev/ttyAMA3'

#設置實驗資料放置的資料夾
exp_name = '/home/inventec/Desktop/2KWCDU/data_manage/Real_time_Prediction_data'
#設置實驗資料檔案名稱
exp_var = 'Real_time_Prediction_data_GPU15KW_1(285V_8A)'
#設置實驗資料標題
custom_headers = ['time', 'T_GPU', 'T_heater', 'T_CDU_in', 'T_CDU_out', 'T_env', 'T_air_in', 'T_air_out', 'TMP8', 'fan_duty', 'pump_duty','T_w_delta', 'GPU_Watt(KW)']

adam = ADAMScontroller.DataAcquisition(exp_name, exp_var, port=adam_port, csv_headers=custom_headers)
fan1 = multi_ctrl.multichannel_PWMController(fan1_port)
fan2 = multi_ctrl.multichannel_PWMController(fan2_port)
pump = ctrl.XYKPWMController(pump_port)

# 修改模型和scaler加载路径
model_path = 'Predict_Model/multi_seq20_steps8_batch512_hidden8_layers1_heads8_dropout0.005_epoch300/2KWCDU_Transformer_model.pth'
# 設定MinMaxScaler的路徑，此scaler用於將輸入數據歸一化到[0,1]區間
# 該scaler是在訓練模型時保存的，確保預測時使用相同的數據縮放方式
scaler_path = 'Predict_Model/multi_seq20_steps8_batch512_hidden8_layers1_heads8_dropout0.005_epoch300/minmax_scaler.pkl' 

# 加載模型和scaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device)
model.eval()

# 加載訓練好的scaler
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# 位置編碼器類
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

# Transformer模型類
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout):
        super(TransformerModel, self).__init__()
        self.output_dim = output_dim
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.decoder_embedding = nn.Linear(output_dim, hidden_dim)
        self.pos_decoder = PositionalEncoding(hidden_dim)
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.tgt_mask = None
        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, src, num_steps):
        src_embedded = self.embedding(src)
        src_embedded = src_embedded.permute(1, 0, 2)
        src_embedded = self.pos_encoder(src_embedded)
        memory = self.transformer_encoder(src_embedded)

        tgt = torch.zeros(src.size(0), 1, self.output_dim).to(src.device)
        
        outputs = []
        for _ in range(num_steps):
            tgt_embedded = self.decoder_embedding(tgt)
            tgt_embedded = tgt_embedded.permute(1, 0, 2)
            tgt_embedded = self.pos_decoder(tgt_embedded)
            
            if self.tgt_mask is None or self.tgt_mask.size(0) != tgt_embedded.size(0):
                device = tgt.device
                mask = self._generate_square_subsequent_mask(tgt_embedded.size(0)).to(device)
                self.tgt_mask = mask

            output = self.transformer_decoder(tgt_embedded, memory, tgt_mask=self.tgt_mask)
            output = output.permute(1, 0, 2)
            output = self.fc(output)
            outputs.append(output[:, -1, :])
            tgt = torch.cat((tgt, output[:, -1:, :]), dim=1)
        
        return torch.stack(outputs, dim=1)

# 創建數據緩存
time_window = 20  # 時間窗口大小
history_buffer = deque(maxlen=time_window)
features = ['T_GPU', 'T_heater', 'T_CDU_in', 'T_env', 'T_air_in', 'T_air_out', 'fan_duty', 'pump_duty', 'GPU_Watt(KW)']

# 修改預測數據記錄結構
prediction_data = {
    'timestamps': [],
    'actual_temps': [],
    'predicted_sequence': []  # 儲存8個時間步的預測
}

# 初始化繪圖
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
line1, = ax.plot([], [], 'r-', label='實際溫度')
line2, = ax.plot([], [], 'b--', label='預測溫度')
ax.set_xlabel('時間 (s)')
ax.set_ylabel('溫度 (°C)')
ax.legend()

def get_current_features(data):
    """獲取當前時間步的特徵"""
    return np.array([
        data['T_GPU'],
        data['T_heater'],
        data['T_CDU_in'],
        data['T_env'],
        data['T_air_in'],
        data['T_air_out'],
        data['fan_duty'],
        data['pump_duty'],
        data['GPU_Watt(KW)']
    ])

def prepare_sequence_data(history_buffer):
    """準備並預處理序列數據"""
    # 轉換歷史數據為numpy數組
    sequence = np.array(list(history_buffer))
    
    # 使用scaler進行特徵標準化
    sequence_scaled = scaler.transform(sequence)
    
    # 轉換為PyTorch張量並添加batch維度
    return torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)

def inverse_transform_predictions(scaled_predictions):
    """將預測結果轉換回原始範圍"""
    # 確保預測值的形狀正確（如果需要，添加特徵維度）
    if len(scaled_predictions.shape) == 1:
        scaled_predictions = scaled_predictions.reshape(-1, 1)
    
    # 反向轉換預測值
    return scaler.inverse_transform(scaled_predictions)[:, 0]  # 只取第一個特徵（溫度）

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
        # 獲取當前數據
        data = adam.get_data()
        
        # 更新歷史緩存
        current_features = get_current_features(data)
        history_buffer.append(current_features)
        
        # 當歷史數據足夠時進行預測
        if len(history_buffer) == time_window:
            # 準備並預處理輸入數據
            input_tensor = prepare_sequence_data(history_buffer)
            
            # 执行预测
            with torch.no_grad():
                scaled_predictions = model(input_tensor, num_steps=8)[0].cpu().numpy()
            
            # 將預測結果轉換回原始範圍
            predicted_sequence = inverse_transform_predictions(scaled_predictions)
            
            # 記錄結果
            current_time = time.time()
            prediction_data['timestamps'].append(current_time)
            prediction_data['actual_temps'].append(data['T_CDU_out'])
            prediction_data['predicted_sequence'].append(predicted_sequence)
            
            # 更新图表
            update_plot()
            
            # 打印預測結果
            print(f"當前溫度: {data['T_CDU_out']:.2f}°C")
            print(f"未來8步預測溫度: {predicted_sequence}")
        
        time.sleep(1)  # 控制採樣頻率

    except Exception as e:
        print(f"預測錯誤: {str(e)}")
        time.sleep(1)




