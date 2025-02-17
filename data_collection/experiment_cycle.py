'''''
本程式碼主要用於資料蒐集

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
import ADAMScontroller
import pwmcontroller as ctrl
import multi_channel_pwmcontroller as multi_ctrl
import csv
import os
import json
from datetime import datetime, timedelta


# 初始化控制器
adam_port = '/dev/ttyUSB0'
pump_port = '/dev/ttyAMA3'
fan1_port = '/dev/ttyAMA5'
fan2_port = '/dev/ttyAMA4'

#設置實驗資料放置的資料夾
exp_name = '/home/inventec/Desktop/2KWCDU_修改版本/data_manage/Model_Training_data'
#設置實驗資料檔案名稱
exp_var = 'Training_data_GPU15KW_1(285V_8A)'
#設置保存進度的jason文件名稱
experiment_progress='Training_data_GPU15KW_1(285V_8A).json'
#選取變數設置的csv檔案    
settings_file = '/home/inventec/Desktop/2KWCDU_修改版本/Experimental_parameter_setting/training_data_final_1.csv'
#設定檔案標題
custom_headers = ['time', 'T_GPU', 'T_heater', 'T_CDU_in', 'T_CDU_out', 'T_env', 'T_air_in', 'T_air_out', 'TMP8', 'fan_duty', 'pump_duty', 'GPU_Watt']

# 創建控制器物件
adam = ADAMScontroller.DataAcquisition(exp_name=exp_name, exp_var=exp_var, port=adam_port, csv_headers=custom_headers)
fan1 = multi_ctrl.multichannel_PWMController(port=fan1_port)
fan2 = multi_ctrl.multichannel_PWMController(port=fan2_port)
pump = ctrl.XYKPWMController(port=pump_port)

# 設置初始轉速
pump_duty=40
pump.set_duty_cycle(pump_duty)
fan_duty=30
fan1.set_all_duty_cycle(fan_duty)
fan2.set_all_duty_cycle(fan_duty)

# 設置ADAM控制器
adam.start_adam()

def read_settings(file_path):
    settings = []
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)
        print(f"CSV 列名: {csv_reader.fieldnames}")
        
        fieldnames = [name.strip() for name in csv_reader.fieldnames]
        print(f"處理後的列名: {fieldnames}")
        
        for row in csv_reader:
            cleaned_row = {k.strip(): v.strip() for k, v in row.items()}
            settings.append({
                'fan_speed': int(cleaned_row['fan_speed']),
                'pump_speed': int(cleaned_row['pump_speed']),
                'duration': int(cleaned_row['duration'])
            })
    return settings

def save_progress(step):
    with open(experiment_progress, 'w') as f:
        json.dump({'current_step': step}, f)

def load_progress():
    if os.path.exists(experiment_progress):
        with open(experiment_progress, 'r') as f:
            return json.load(f)['current_step']
    return 0

def calculate_total_duration(settings):
    return sum(setting['duration'] for setting in settings)

try:
    settings = read_settings(settings_file)
    current_step = load_progress()
    
    total_duration = calculate_total_duration(settings[current_step:])
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=total_duration)
    
    print(f"實驗開始時間: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"預計完成時間: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"預計總時長: {timedelta(seconds=total_duration)}")

    for i in range(current_step, len(settings)):
        setting = settings[i]
        current_time = datetime.now()
        remaining_duration = calculate_total_duration(settings[i:])
        updated_end_time = current_time + timedelta(seconds=remaining_duration)
        
        print(f"\n當前步驟: {i+1}/{len(settings)}")
        print(f"設置風扇速度為 {setting['fan_speed']}, 泵速度為 {setting['pump_speed']}, 持續 {setting['duration']} 秒")
        print(f"更新後的預計完成時間: {updated_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"剩餘時間: {timedelta(seconds=remaining_duration)}")
        
        # 設置風扇速度
        fan1.set_all_duty_cycle(f"{setting['fan_speed']:03d}")
        fan2.set_all_duty_cycle(f"{setting['fan_speed']:03d}")
        
        # 設置泵速度
        pump.set_duty_cycle(f"{setting['pump_speed']:03d}")
        
        # 更新 ADAM 控制器中的工作週期數據
        adam.update_duty_cycles(setting['fan_speed'], setting['pump_speed'])
        
        # 等待指定的持續時間
        step_start_time = time.time()
        while time.time() - step_start_time < setting['duration']:
            # 讀取和顯示數據
            with adam.buffer_lock:
                current_data = adam.buffer.tolist()
            elapsed_time = time.time() - step_start_time
            remaining_time = setting['duration'] - elapsed_time
            print(f"當前數據: {current_data}")
            print(f"當前步驟剩餘時間: {timedelta(seconds=int(remaining_time))}", end='\r')
            time.sleep(1)  # 每秒更新一次
        
        save_progress(i + 1)

except KeyboardInterrupt:
    print("\n實驗被用戶中斷")
finally:
    adam.stop_threading('buffer')
    adam.stop_threading('adam')
    adam.closeport()
    fan1.set_all_duty_cycle(100)
    fan2.set_all_duty_cycle(100)
    pump.set_duty_cycle(100)
    print('實驗結束.')
