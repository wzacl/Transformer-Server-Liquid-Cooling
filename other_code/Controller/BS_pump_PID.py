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
import Optimal_algorithm.Binary_search as BS
import GB_PID_pump as Pump_pid

adam_port = '/dev/ttyUSB0'
fan1_port = '/dev/ttyAMA4'
fan2_port = '/dev/ttyAMA5'
pump_port = '/dev/ttyAMA3'

#設置實驗資料放置的資料夾
exp_name = '/home/inventec/Desktop/2KWCDU_修改版本/data_manage/control_data/Fan_MPC_BSO_data'
#設置實驗資料檔案名稱
exp_var = 'Fan_MPC_data_GPU1.5KW_1(285V_8A)_BSO_test.csv'
#設置實驗資料標題
custom_headers = ['time', 'T_GPU', 'T_heater', 'T_CDU_in', 'T_CDU_out', 'T_env', 'T_air_in', 'T_air_out', 'TMP8', 'fan_duty', 'pump_duty']

adam = ADAMScontroller.DataAcquisition(exp_name=exp_name, exp_var=exp_var, port=adam_port, csv_headers=custom_headers)
fan1 = multi_ctrl.multichannel_PWMController(fan1_port)
fan2 = multi_ctrl.multichannel_PWMController(fan2_port)
pump = ctrl.XYKPWMController(pump_port)

# 設置初始轉速
pump_duty = 60
pump.set_duty_cycle(pump_duty)
fan_duty = 60
fan1.set_all_duty_cycle(fan_duty)
fan2.set_all_duty_cycle(fan_duty)

# 設置ADAM控制器
adam.start_adam()
adam.update_duty_cycles(fan_duty, pump_duty)

# 初始化二分搜索優化器
print("⚡ 初始化二分搜索優化器")
bs_optimizer = BS.BinarySearchOptimizer(
    adam=adam,
    window_size=35,
    P_max=500,
    target_temp=28,
    max_iter=8,
    min_speed=30,
    max_speed=100,
    tolerance=1,
    model_path='/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400/2KWCDU_Transformer_model.pth',
    scaler_path='/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400/1.5_1KWscalers.jlib'
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
                
                if counter % control_frequency == 0:
                    print("🔄 執行風扇BS優化...")
                    start_time = time.time()
                    optimal_fan_speed, optimal_cost = bs_optimizer.optimize()
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
        except Exception as e:
            print(f"❌ 執行過程中發生錯誤: {e}")
            time.sleep(1)

except KeyboardInterrupt:
    print("\n⚠️ 程序被用戶中斷")

finally:
    # 清理資源
    adam.stop_threading('buffer')
    adam.stop_threading('adam')
    adam.closeport()
    fan1.set_all_duty_cycle(40)
    fan2.set_all_duty_cycle(40)
    pump.set_duty_cycle(100)
    print("🔒 實驗結束，所有裝置已恢復到安全狀態。")