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
import Transformer_enc_dec
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

adam_port = '/dev/ttyUSB0'
fan1_port = '/dev/ttyAMA4'
fan2_port = '/dev/ttyAMA5'
pump_port = '/dev/ttyAMA3'
#選擇模型
test_model='multi_seq35_steps8_batch512_hidden16_encoder1_decoder1_heads2_dropout0.005_epoch200'

# 從模型名稱中提取超參數
model_params = {}
params_str = test_model.split('_')
for param in params_str:
    if 'seq' in param:
        model_params['seq_len'] = int(param.replace('seq', ''))
    elif 'hidden' in param:
        model_params['hidden_dim'] = int(param.replace('hidden', ''))
    elif 'encoder' in param:
        model_params['num_encoder_layers'] = int(param.replace('encoder', ''))
    elif 'decoder' in param:
        model_params['num_decoder_layers'] = int(param.replace('decoder', ''))
    elif 'heads' in param:
        model_params['num_heads'] = int(param.replace('heads', ''))
    elif 'dropout' in param:
        model_params['dropout'] = float(param.replace('dropout', ''))

# 更新時間窗口大小
time_window = model_params['seq_len']

#設置實驗資料放置的資料夾
exp_name = f'/home/inventec/Desktop/2KWCDU_修改版本/data_manage/Real_time_Prediction_data/only1.5KW_USE_T_env/{test_model}'
#設置實驗資料檔案名稱
exp_var = 'GPU15KW_1(285V_8A)_fan_test'
#設置實驗資料標題
custom_headers = ['time', 'T_GPU', 'T_heater', 'T_CDU_in', 'T_CDU_out', 'T_env', 'T_air_in', 'T_air_out', 'TMP8', 'fan_duty', 'pump_duty', 'GPU_Watt(KW)']

adam = ADAMScontroller.DataAcquisition(exp_name=exp_name, exp_var=exp_var, port=adam_port, csv_headers=custom_headers)
fan1 = multi_ctrl.multichannel_PWMController(fan1_port)
fan2 = multi_ctrl.multichannel_PWMController(fan2_port)
pump = ctrl.XYKPWMController(pump_port)
print('模型初始化.....')


# 修改模型和scaler路徑
model_path = f'/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/{test_model}/2KWCDU_Transformer_model.pth'
# 設定MinMaxScaler的路徑，此scaler用於將輸入數據歸一化到[0,1]區間
# 該scaler是在訓練模型時保存的，確保預測時使用相同的數據縮放方式
scaler_path = f'/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/{test_model}/1.5_1KWscalers.jlib' 
# 檢查文件是否存在,如果不存在則創建並寫入標題行
prediction_file = f'/home/inventec/Desktop/2KWCDU_修改版本/data_manage/Real_time_Prediction/only1.5KW_USE_T_env/{test_model}/Model_test_{exp_var}.csv'
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
time.sleep(2)

# 加載模型和scaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
model = Transformer_enc_dec.TransformerModel(
    input_dim=7, 
    hidden_dim=model_params['hidden_dim'], 
    output_dim=1, 
    num_encoder_layers=model_params['num_encoder_layers'], 
    num_decoder_layers=model_params['num_decoder_layers'], 
    num_heads=model_params['num_heads'], 
    dropout=model_params['dropout']
)
model.load_state_dict(model_state_dict)
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
model_tester.start_test(1)  # 這裡選擇隨機變動測試


while model_tester.phase != "end":
    try:
        model_tester.update_test()

        # ✅ 更新來自 ADAMS 的數據，確保滑動窗口數據是最新的
        #seq_window_processor.update_from_adam()

        # ✅ 確保 window_data 已準備好
        input_tensor = seq_window_processor.get_window_data(normalize=True)

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
        with torch.no_grad():
            inference_start_time = time.time()
            scaled_predictions = model(input_tensor, num_steps=8)[0].cpu().numpy()
            inference_end_time = time.time()
            inference_duration = inference_end_time - inference_start_time

        # 將預測結果轉換回原始範圍
        predicted_sequence = seq_window_processor.inverse_transform_predictions(scaled_predictions.reshape(-1, 1),smooth=True).flatten()

        # 記錄結果
        current_time = time.time()
        keys = ['timestamp', 'actual_temps(CDU_out)', 'actual_temps(GPU)', 'T_env', 'fan_duty', 'pump_duty', 'predicted_sequence']
        values = [current_time, adam.buffer[3], data[0], data[2], adam.buffer[8], adam.buffer[9], predicted_sequence]
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






