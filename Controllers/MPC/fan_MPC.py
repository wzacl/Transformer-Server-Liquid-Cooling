import time
import sys
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Control_Unit')
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Controllers/MPC/Model_constructor')
import ADAMScontroller
import pwmcontroller as ctrl
import multi_channel_pwmcontroller as multi_ctrl
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque
import math
import os
import csv
import random
import torch
import Data_Processor as dp
import Optimal_algorithm.FHO as fho

adam_port = '/dev/ttyUSB0'
fan1_port = '/dev/ttyAMA4'
fan2_port = '/dev/ttyAMA5'
pump_port = '/dev/ttyAMA3'

#設置實驗資料放置的資料夾
exp_name = '/home/inventec/Desktop/2KWCDU_修改版本/data_manage/Fan_MPC_FHO_data'
#設置實驗資料檔案名稱
exp_var = 'Fan_MPC_data_GPU15KW_1(285V_8A)_FHO.csv'
#設置實驗資料標題
custom_headers = ['time', 'T_GPU', 'T_heater', 'T_CDU_in', 'T_CDU_out', 'T_env', 'T_air_in', 'T_air_out', 'TMP8', 'fan_duty', 'pump_duty', 'GPU_Watt(KW)']

adam = ADAMScontroller.DataAcquisition(exp_name=exp_name, exp_var=exp_var, port=adam_port, csv_headers=custom_headers)
fan1 = multi_ctrl.multichannel_PWMController(fan1_port)
fan2 = multi_ctrl.multichannel_PWMController(fan2_port)
pump = ctrl.XYKPWMController(pump_port)

# 設定MinMaxScaler的路徑，此scaler用於將輸入數據歸一化到[0,1]區間
# 該scaler是在訓練模型時保存的，確保預測時使用相同的數據縮放方式
scaler_path = '/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/1.5_1KWscalers.jlib' 
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
Data_Processor = dp.Data_Processor(scaler_path, device)
# 創建數據緩存，大小須符合輸入序列長度
time_window = 20  # 時間窗口大小
history_buffer = deque(maxlen=time_window)
fho_optimizer = fho.FirehawkOptimizer(adam, time_window, num_firehawks=10, max_iter=50, P_max=100, target_temp=25)

#設置控制頻率
control_frequency = 3  # 控制頻率 (s)

while True:
    optimal_fan_speed, optimal_cost = fho_optimizer.optimize()
    if optimal_fan_speed is not None:
        fan1.set_all_duty_cycle(optimal_fan_speed)
        fan2.set_all_duty_cycle(optimal_fan_speed)
        print(f"Optimal Fan Speed: {optimal_fan_speed}% with Cost: {optimal_cost:.2f}")
    else:
        print("❌ 數據蒐集中，等待數據蒐集完成")
        time.sleep(control_frequency)


