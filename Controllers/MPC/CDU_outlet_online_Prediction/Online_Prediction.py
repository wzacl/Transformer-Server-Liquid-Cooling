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
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Control_Unit')
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Controllers/MPC/Model_constructor')
import ADAMScontroller
import pwmcontroller as ctrl
import multi_channel_pwmcontroller as multi_ctrl
from simple_pid import PID
import matplotlib.pyplot as plt
import numpy as np
import time
import joblib
import Transformer
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
import Data_Processor as dp

adam_port = '/dev/ttyUSB0'
fan1_port = '/dev/ttyAMA4'
fan2_port = '/dev/ttyAMA5'
pump_port = '/dev/ttyAMA3'

#設置實驗資料放置的資料夾
exp_name = '/home/inventec/Desktop/2KWCDU_修改版本/data_manage/Real_time_Prediction_data'
#設置實驗資料檔案名稱
exp_var = 'Real_time_Prediction_data_GPU15KW_1(285V_8A)_test_fan_pump_3.csv'
#設置實驗資料標題
custom_headers = ['time', 'T_GPU', 'T_heater', 'T_CDU_in', 'T_CDU_out', 'T_env', 'T_air_in', 'T_air_out', 'TMP8', 'fan_duty', 'pump_duty', 'GPU_Watt(KW)']

adam = ADAMScontroller.DataAcquisition(exp_name=exp_name, exp_var=exp_var, port=adam_port, csv_headers=custom_headers)
fan1 = multi_ctrl.multichannel_PWMController(fan1_port)
fan2 = multi_ctrl.multichannel_PWMController(fan2_port)
pump = ctrl.XYKPWMController(pump_port)

print('模型初始化.....')

# 修改模型和scaler路徑
model_path = '/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/2KWCDU_Transformer_model.pth'
# 設定MinMaxScaler的路徑，此scaler用於將輸入數據歸一化到[0,1]區間
# 該scaler是在訓練模型時保存的，確保預測時使用相同的數據縮放方式
scaler_path = '/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/1.5_1KWscalers.jlib' 
figure_path = '/home/inventec/Desktop/2KWCDU_修改版本/data_manage/Real_time_Prediction/temperature_prediction_fan_pump_change_3.png'
# 檢查文件是否存在,如果不存在則創建並寫入標題行
prediction_file = '/home/inventec/Desktop/2KWCDU_修改版本/data_manage/Real_time_Prediction/Model_test_change_fan_pump_3.csv'
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

# 加載模型和scaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
model = Transformer.TransformerModel(input_dim=7, hidden_dim=8, output_dim=1, num_layers=1, num_heads=8, dropout=0.01)
model.load_state_dict(model_state_dict)
model.eval()

Data_Processor = dp.Data_Processor(scaler_path, device)

# 創建數據緩存
time_window = 20  # 時間窗口大小
history_buffer = deque(maxlen=time_window)

# 修改預測數據記錄結構
prediction_data = {
    'timestamp': [],
    'actual_temps(CDU_out)': [],
    'actual_temps(GPU)': [],
    'fan_duty': [],
    'pump_duty': [],
    'predicted_sequence': []  # 儲存8個時間步的預測
}

# 創建 Model_tester 物件
model_tester = mt.Model_tester(fan1=fan1, fan2=fan2, pump=pump, adam=adam)

# 選擇測試模式 (1: 只變動風扇, 2: 只變動泵, 3: 隨機變動)
model_tester.start_test(3)  # 這裡選擇隨機變動測試


# 主循环
while True:
    model_tester.update_test()
    try:
        # 獲取當前數據並確保有7個特徵
        data = [
            adam.buffer[0],  # T_GPU
            adam.buffer[2],  # T_CDU_in
            adam.buffer[4],  # T_env
            adam.buffer[5],  # T_air_in
            adam.buffer[6],  # T_air_out
            adam.buffer[8],  # fan_duty
            adam.buffer[9]   # pump_duty
        ]

        # 更新歷史緩存 (存入已縮放的數據)
        current_features = Data_Processor.get_current_features(data)
        history_buffer.append(current_features)
        
        # 準備輸入數據 (數據已經過縮放)
        input_tensor = Data_Processor.prepare_sequence_data(history_buffer)
        
        # 當歷史數據足夠時進行預測
        if input_tensor is not None:
            
            # 執行預測
            with torch.no_grad():
                scaled_predictions = model(input_tensor, num_steps=8)[0].cpu().numpy()
                print(f"scaled_predictions 形狀: {scaled_predictions.shape}")

            # 將預測結果轉換回原始範圍
            predicted_sequence = Data_Processor.inverse_transform_predictions(scaled_predictions)
            
            # 記錄結果
            # 獲取當前時間
            current_time = time.time()
            
            # 將數據保存到 prediction_data 字典中
            prediction_data['timestamp'].append(current_time)
            prediction_data['actual_temps(CDU_out)'].append(adam.buffer[3])  # T_CDU_out 位於索引3
            prediction_data['actual_temps(GPU)'].append(data[0])  # T_GPU 位於索引0
            prediction_data['fan_duty'].append(adam.buffer[8])  # fan_duty 位於索引8
            prediction_data['pump_duty'].append(adam.buffer[9])  # pump_duty 位於索引9
            prediction_data['predicted_sequence'].append(predicted_sequence)
            
            # 寫入預測數據到CSV檔案
            with open(prediction_file, 'a', newline='') as f:
                writer = csv.writer(f)
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
                row = [timestamp, adam.buffer[3], data[0], adam.buffer[8], adam.buffer[9]] + [f'{temp:.2f}' for temp in predicted_sequence]
                writer.writerow(row)
            
            # 打印預測結果
            print("==================== 系統狀態 ====================")
            print(f"當前出口溫度:     {adam.buffer[3]:.2f}°C")
            print(f"當前晶片溫度:     {data[0]:.2f}°C")
            print("\n==================== 預測結果 ====================")
            print(f"未來8步預測溫度: {predicted_sequence}")

        
        time.sleep(1)  # 控制採樣頻率

    except KeyboardInterrupt:
        print("實驗結束，程序已安全退出。")
        adam.stop_adam()
        
        break

    except Exception as e:
        print(f"預測錯誤: {str(e)}")
        time.sleep(1)




