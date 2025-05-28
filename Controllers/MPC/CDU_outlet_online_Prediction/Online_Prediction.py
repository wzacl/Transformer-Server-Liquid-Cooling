# /usr/bin/python3
'''''
GB_PID_pump.py

GB_PID控制器，利用Guaranteed Bounded PID控制泵轉速

本研究中的晶片瓦數對應的電源供應器參數設置如下
1KW：220V_8A
1.5KW：285V_8A
1.9KW：332V_8A

對應的風扇與泵最低轉速如下
泵：40% duty cycle
風扇：30% duty cycle
'''''
import time
import sys
# 添加專案根目錄到 Python 路徑
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本')
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Control_Unit')
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Controllers/MPC/Model_constructor')
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Controllers/MPC')
import ADAMScontroller
import pwmcontroller as ctrl
import multi_channel_pwmcontroller as multi_ctrl
from simple_pid import PID
import matplotlib.pyplot as plt
import numpy as np
import time
import joblib
# 修正模型導入路徑 - 與 SA_iTransformer.py 保持一致
from code_manage.Controllers.MPC.model import Model
import torch
import torch.nn as nn
from collections import deque
import math
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import csv
import random
import Model_tester as mt
import Sequence_Window_Processor
from tabulate import tabulate

class ModelConfig:
    """
    模型配置類，統一管理模型參數
    """
    def __init__(self, input_dim=7, d_model=16, n_heads=8, e_layers=1, d_ff=16, 
                 dropout=0.01, seq_len=40, pred_len=8, embed='timeF', freq='h',
                 class_strategy='cls', activation='gelu', output_attention=False, use_norm=True):
        """
        初始化模型配置
        
        Args:
            input_dim (int): 輸入特徵維度
            d_model (int): 模型隱藏層維度
            n_heads (int): 注意力頭數
            e_layers (int): 編碼器層數
            d_ff (int): 前饋網絡維度
            dropout (float): Dropout比率
            seq_len (int): 輸入序列長度
            pred_len (int): 預測序列長度
            embed (str): 嵌入類型
            freq (str): 時間頻率
            class_strategy (str): 分類策略
            activation (str): 激活函數
            output_attention (bool): 是否輸出注意力權重
            use_norm (bool): 是否使用層歸一化
        """
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embed = embed
        self.freq = freq
        self.class_strategy = class_strategy
        self.activation = activation
        self.output_attention = output_attention
        self.use_norm = use_norm

adam_port = '/dev/ttyUSB0'
fan1_port = '/dev/ttyAMA4'
fan2_port = '/dev/ttyAMA5'
pump_port = '/dev/ttyAMA3'
#選擇模型
test_model='iTransformer_no_air_out_seq25_pred8_dmodel16_dff32_nheads2_elayers1_dropout0.01_lr0.0001_batchsize512_epochs140'

# 更新時間窗口大小
time_window = 25

#設置實驗資料放置的資料夾
exp_name = f'/home/inventec/Desktop/2KWCDU_修改版本/data_manage/Real_time_Prediction_data/iTransformer/{test_model}'
#設置實驗資料檔案名稱
exp_var = 'all_random_test'
#設置實驗資料標題
custom_headers = ['time', 'T_GPU', 'T_heater', 'T_CDU_in', 'T_CDU_out', 'T_env', 'T_air_in', 'T_air_out', 'TMP8', 'fan_duty', 'pump_duty', 'GPU_Watt(KW)']

adam = ADAMScontroller.DataAcquisition(exp_name=exp_name, exp_var=exp_var, port=adam_port, csv_headers=custom_headers)
fan1 = multi_ctrl.multichannel_PWMController(fan1_port)
fan2 = multi_ctrl.multichannel_PWMController(fan2_port)
pump = ctrl.XYKPWMController(pump_port)
print('模型初始化.....')


# 修改模型和scaler路徑
model_path = f'/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/iTransformer/iTransformer_no_air_out_seq25_pred8_dmodel16_dff32_nheads2_elayers1_dropout0.01_lr0.0001_batchsize512_epochs140/best_model.pth'
# 設定MinMaxScaler的路徑，此scaler用於將輸入數據歸一化到[0,1]區間
# 該scaler是在訓練模型時保存的，確保預測時使用相同的數據縮放方式
scaler_path = f'/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/iTransformer/iTransformer_no_air_out_seq25_pred8_dmodel16_dff32_nheads2_elayers1_dropout0.01_lr0.0001_batchsize512_epochs140/scalers.jlib' 
# 檢查文件是否存在,如果不存在則創建並寫入標題行
prediction_file = f'/home/inventec/Desktop/2KWCDU_修改版本/data_manage/Real_time_Prediction/iTransformer/iTransformer_no_air_out_seq25_pred8_dmodel16_dff32_nheads2_elayers1_dropout0.01_lr0.0001_batchsize512_epochs140/Model_test_{exp_var}.csv'
if not os.path.exists(prediction_file):
    os.makedirs(os.path.dirname(prediction_file), exist_ok=True)
    with open(prediction_file, 'w') as f:
        f.write('timestamp,actual_temp(CDU_out),actual_temp(GPU),fan_duty,pump_duty,pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,pred_7,pred_8\n')

# 設置初始轉速
pump_duty=60
pump.set_duty_cycle(pump_duty)
fan_duty=60
fan1.set_all_duty_cycle(fan_duty)
fan2.set_all_duty_cycle(fan_duty)

# 設置ADAM控制器
adam.start_adam()
adam.update_duty_cycles(fan_duty, pump_duty)
time.sleep(1)

# 加載模型和scaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 使用統一的模型配置
model_config = ModelConfig(
    input_dim=6,
    d_model=16,
    n_heads=2,
    e_layers=1,
    d_ff=32,
    dropout=0.01,
    seq_len=25,
    pred_len=8
)

# 創建模型實例 - 修正初始化方式
model = Model(
    model_config
).to(device)

# 載入模型權重 - 修正加載方式
checkpoint = torch.load(model_path, map_location=device)
if 'model_state_dict' in checkpoint:
    # 檢查點包含模型狀態字典
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    # 直接嘗試加載
    model.load_state_dict(checkpoint)

# 設置為評估模式
model.eval()

seq_window_processor = Sequence_Window_Processor.SequenceWindowProcessor(
    window_size=time_window,
    adams_controller=adam,  # 你的 ADAMScontroller 物件
    scaler_path=scaler_path,  # 你的 Scaler 檔案
    device="cpu"
)



# 修改預測數據記錄結構
prediction_data = {
    'timestamp': [],
    'actual_temps(CDU_out)': [],
    'actual_temps(GPU)': [],
    'T_env': [],
    'fan_duty': [],
    'pump_duty': [],
    'predicted_sequence': []  # 儲存8個時間步的預測
}

# 創建 Model_tester 物件
model_tester = mt.Model_tester(fan1=fan1, fan2=fan2, pump=pump, adam=adam)

# 選擇測試模式 (1: 只變動風扇, 2: 只變動泵, 3: 隨機變動)
model_tester.start_test(3,900)  # 這裡選擇隨機變動測試


while model_tester.phase != "end":
    try:
        model_tester.update_test()

        # ✅ 確保 window_data 已準備好
        input_tensor = seq_window_processor.get_window_data(normalize=False)

        if input_tensor is None:  # 修正條件，應該等待數據準備好
            time.sleep(1)
            continue

        # 獲取當前數據
        data = [
            adam.buffer[0],  # T_GPU
            adam.buffer[2],  # T_CDU_in
            adam.buffer[3],  # T_CDU_out
            adam.buffer[4],  # T_env
            adam.buffer[5],  # T_air_in
            adam.buffer[6],  # T_air_out
            adam.buffer[8],  # fan_duty
            adam.buffer[9]   # pump_duty
        ]

        # 執行預測
        if input_tensor is not None:
            # 記錄推論開始時間
            inference_start = time.time()
            
            with torch.no_grad():
                # 檢查模型輸出
                model_output = model(seq_window_processor.transform_input_data(input_tensor))
                scaled_predictions = model_output[0].cpu().numpy()  # 獲取縮放後的預測結果
                # 使用修改後的反轉縮放方法
                predicted_temps = seq_window_processor.inverse_transform_predictions(scaled_predictions)  # 反轉縮放
                
                # 使用修改後的反轉縮放方法
                predicted_sequence = seq_window_processor.inverse_transform_predictions(scaled_predictions)  # 反轉縮放
            
            # 計算推論時間
            inference_duration = time.time() - inference_start
        else:
            predicted_sequence = None
            inference_duration = 0.0

        # 記錄結果
        current_time = time.time()
        
        # 只在有有效預測結果時才記錄和處理數據
        if predicted_sequence is not None:
            keys = ['timestamp', 'actual_temps(CDU_out)', 'actual_temps(GPU)', 'T_env', 'fan_duty', 'pump_duty', 'predicted_sequence']
            values = [current_time, adam.buffer[3], adam.buffer[0], adam.buffer[4], adam.buffer[8], adam.buffer[9], predicted_sequence]
            for key, value in zip(keys, values):
                prediction_data[key].append(value)

            # 寫入預測數據到 CSV 檔案
            with open(prediction_file, 'a', newline='') as f:
                writer = csv.writer(f)
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
                row = [timestamp, adam.buffer[3], data[0], adam.buffer[8], adam.buffer[9]] + [f'{temp:.2f}' for temp in predicted_sequence]
                writer.writerow(row)

            # 系統狀態數據
            system_status_data = [
                ["🌡️ 當前出口溫度", f"{adam.buffer[3]:.2f}°C"],
                ["💻 當前晶片溫度", f"{data[0]:.2f}°C"],
                ["🌍 當前環境溫度", f"{data[3]:.2f}°C"],
                ["💨 當前進風溫度", f"{data[4]:.2f}°C"],
                ["🌬️ 當前出風溫度", f"{data[5]:.2f}°C"],
                ["🔄 當前風扇轉速", f"{data[6]:.2f}%"],
                ["💧 當前泵轉速", f"{data[7]:.2f}%"]
            ]

            # 預測結果數據
            prediction_results_data = [
                ["🔮 未來8步預測溫度", predicted_sequence.tolist()],  # 修正為列表，以確保可讀性
                ["📏 scaled_predictions 形狀", scaled_predictions.shape],
                ["⏱️ 模型推論時間", f"{inference_duration:.4f} 秒"]
            ]

            # 打印系統狀態表格
            print("🌟==================== 系統狀態 ====================🌟")
            print(tabulate(system_status_data, tablefmt="grid"))

            # 打印預測結果表格
            print("\n🌟==================== 預測結果 ====================🌟")
            print(tabulate(prediction_results_data, tablefmt="grid"))
        else:
            print("⚠️ 等待數據準備中...")
        
        time.sleep(1)

    except ValueError as e:
        print(f"⚠️ 錯誤: {str(e)}")
        time.sleep(1)

    except KeyboardInterrupt:
        print("實驗結束，程序已安全退出。")
        adam.stop_adam()
        
        break

    except Exception as e:
        print(f"❌ 預測錯誤: {str(e)}")
        time.sleep(1)
adam.stop_adam()
fan1.set_all_duty_cycle(60)
fan2.set_all_duty_cycle(60)
pump.set_duty_cycle(60)
print("🔴 實驗結束，程序已安全退出。")


adam.stop_adam()
fan1.set_all_duty_cycle(60)
fan2.set_all_duty_cycle(60)
pump.set_duty_cycle(60)






