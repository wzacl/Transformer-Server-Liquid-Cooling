#usr/bin/env python3
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
import Optimal_algorithm.Progressive_Fan_Optimizer as po
import GB_PID_pump as Pump_pid

adam_port = '/dev/ttyUSB0'
fan1_port = '/dev/ttyAMA4'
fan2_port = '/dev/ttyAMA5'
pump_port = '/dev/ttyAMA3'

#設置實驗資料放置的資料夾
exp_name = '/home/inventec/Desktop/2KWCDU_修改版本/data_manage/control_data/Fan_MPC_PO_data'
#設置實驗資料檔案名稱
exp_var = 'Fan_MPC_data_GPU1.5KW_1(285V_8A)_PO_test_smooth_2.csv'
#設置實驗資料標題
custom_headers = ['time', 'T_GPU', 'T_heater', 'T_CDU_in', 'T_CDU_out', 'T_env', 'T_air_in', 'T_air_out', 'TMP8', 'fan_duty', 'pump_duty']

adam = ADAMScontroller.DataAcquisition(exp_name=exp_name, exp_var=exp_var, port=adam_port, csv_headers=custom_headers)
fan1 = multi_ctrl.multichannel_PWMController(fan1_port)
fan2 = multi_ctrl.multichannel_PWMController(fan2_port)
pump = ctrl.XYKPWMController(pump_port)

# 設定MinMaxScaler的路徑，此scaler用於將輸入數據歸一化到[0,1]區間
# 該scaler是在訓練模型時保存的，確保預測時使用相同的數據縮放方式
scaler_path = '/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400/1.5_1KWscalers.jlib' 
# 設置初始轉速
pump_duty=60
pump.set_duty_cycle(pump_duty)
fan_duty=60
fan1.set_all_duty_cycle(fan_duty)
fan2.set_all_duty_cycle(fan_duty)

# 設置ADAM控制器
adam.start_adam()
adam.update_duty_cycles(fan_duty, pump_duty)
#設置PO優化器
P_max = 100
target_temp = 28
base_step_size = 5
tolerance = 5
stability_factor = 0.1
decision_history_size = 2

# 使用帶有溫度預測平滑處理功能的PO優化器
# 平滑處理會修正預測序列中的首點跳變問題，使溫度變化更符合物理特性
print("⚡ 初始化PO優化器 (搭載預測溫度平滑處理功能)")
po_optimizer = po.ProgressiveFanOptimizer(
    adam=adam, 
    window_size=35, 
    P_max=P_max, 
    target_temp=target_temp,
    base_step_size=base_step_size, 
    tolerance=tolerance, 
    stability_factor=stability_factor,
    decision_history_size=decision_history_size
)


#設置風扇控制頻率
control_frequency = 4  # 控制頻率 (s)

#設置泵PID控制器
counter = 0
GPU_target = 71
sample_time = 1  # 定義取樣時間
Guaranteed_Bounded_PID_range = 0.5
Controller = Pump_pid.GB_PID_pump(target=GPU_target, Guaranteed_Bounded_PID_range=Guaranteed_Bounded_PID_range, sample_time=sample_time)
try:
    # 設置停止條件
    running = True
    
    while running:
        try:
            Temperatures = adam.buffer.tolist()
            if any(Temperatures):
                # 獲取溫度數據
                T_GPU = Temperatures[0]  # 定義 T_GPU 變量
                T_CDU_out = Temperatures[3]
                T_env = Temperatures[4]
                fan_duty = Temperatures[8]
                pump_duty = Temperatures[9]
                
                print(f"🌡️ 溫度數據 | T_GPU: {T_GPU:.2f}°C | T_CDU_out: {T_CDU_out:.2f}°C | T_env: {T_env:.2f}°C")
                print(f"⚙️ 運行狀態 | 計數器: {counter} | 泵速: {pump_duty}% | 風扇: {fan_duty}%")
                print(f"📝 按下 Ctrl+C 可以手動停止程序")
                
                # 使用 GB_PID 計算控制輸出
                control_temp = Controller.GB_PID(T_GPU, GPU_target)
                pump_duty = round(Controller.controller(control_temp) / 10) * 10
                    
                # 更新泵的轉速
                pump.set_duty_cycle(pump_duty)
                adam.update_duty_cycles(pump_duty=pump_duty)
                time.sleep(1)

                counter += 1

                # 使用新的控制頻率來調整PO的優化頻率
                if counter % control_frequency == 0:
                    print("🔄 執行風扇PO優化...")
                    start_time = time.time()
                    optimal_fan_speed, optimal_cost = po_optimizer.optimize()
                    optimization_time = time.time() - start_time
                    if optimal_fan_speed is not None:
                        fan1.set_all_duty_cycle(int(optimal_fan_speed))
                        fan2.set_all_duty_cycle(int(optimal_fan_speed))
                        adam.update_duty_cycles(fan_duty=int(optimal_fan_speed))
                        print(f"✅ 風扇優化完成 | 最佳風扇轉速: {optimal_fan_speed}% | 成本: {optimal_cost:.2f} | 優化時間: {optimization_time:.2f}秒")
                    else:
                        print("❌ 數據蒐集中，等待數據蒐集完成...")
                else:
                    print(f"⏳ 泵PID控制中 | 目標溫度: {GPU_target}°C | 控制溫度: {control_temp:.2f}°C")
                
                print("================================================")
                
        except KeyboardInterrupt:
            print("\n🛑 程序已被手動停止")
            running = False
            break
        
except Exception as e:
    print(f"發生錯誤: {e}")
finally:
    # 清理資源
    adam.stop_adam()
    fan1.set_all_duty_cycle(20)
    fan2.set_all_duty_cycle(20)
    pump.set_duty_cycle(40)
    print("程序已結束，資源已釋放")