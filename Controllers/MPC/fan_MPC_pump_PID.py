import time
import sys
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Control_Unit')
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Controllers/MPC/Model_constructor')
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Controllers/GB_PID')
import ADAMScontroller
import pwmcontroller as ctrl
import multi_channel_pwmcontroller as multi_ctrl
from simple_pid import PID
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque
import math
import os
import csv
import random
import Optimal_algorithm.FHO as fho
import GB_PID.GB_PID_pump as Pump_pid
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

# 創建數據緩存，大小須符合輸入序列長度
time_window = 20  # 時間窗口大小
history_buffer = deque(maxlen=time_window)
fho_optimizer = fho.FirehawkOptimizer(adam, time_window, num_firehawks=10, max_iter=50, P_max=100, target_temp=25)


#設置風扇控制頻率
control_frequency = 3  # 控制頻率 (s)

#設置泵PID控制器
pump_pid = Pump_pid.GB_PID_pump(target=25, Guaranteed_Bounded_PID_range=0.5,sample_time=1)

while True:
    counter = 0
    target = 68
    sample_time = 1  # 定義取樣時間
    Guaranteed_Bounded_PID_range = 0.5
    Controller = Pump_pid.GB_PID_pump(target, Guaranteed_Bounded_PID_range, sample_time)
    adam.start_adam()
    Temperatures = adam.buffer.tolist()
    if any(Temperatures):
        # 獲取溫度數據
        T_GPU = Temperatures[0]  # 定義 T_GPU 變量
        T_CDU_out = Temperatures[3]
        T_env = Temperatures[4]
        
        print(f"T_GPU: {T_GPU} | T_CDU_out: {T_CDU_out} | T_env: {T_env}")
        print(f"counter: {counter} | pump speed: {pump_duty}")
        print("----------------------------------------")

        # 使用 GB_PID 計算控制輸出
        control_temp = Controller.GB_PID(T_GPU, target)
        pump_duty = round(Controller.controller(control_temp) / 10) * 10
            
        # 更新泵的轉速
        pump.set_duty_cycle(pump_duty)
        adam.update_duty_cycles(pump_duty)

        counter += 1

    optimal_fan_speed, optimal_cost = fho_optimizer.optimize()
    fan1.set_all_duty_cycle(optimal_fan_speed)
    fan2.set_all_duty_cycle(optimal_fan_speed)
    adam.update_duty_cycles(fan_duty)
    print(f"Optimal Fan Speed: {optimal_fan_speed}% with Cost: {optimal_cost:.2f}")
    time.sleep(control_frequency)