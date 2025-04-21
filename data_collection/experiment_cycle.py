#usr/bin/env python3
'''''
本程式碼主要用於資料蒐集

本研究中的晶片瓦數對應的電源供應器參數設置如下
1KW：220V_8A
1.5KW：285V_8A
1.85KW：325V_8A

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
from tabulate import tabulate


# 初始化控制器
adam_port = '/dev/ttyUSB0'
pump_port = '/dev/ttyAMA3'
fan1_port = '/dev/ttyAMA5'
fan2_port = '/dev/ttyAMA4'

#實驗設置變數放置資料夾
setting_folder = '/home/inventec/Desktop/2KWCDU_修改版本/code_manage/data_collection/Experimental_parameter_setting'

# 設置當前資料蒐集瓦數和對應的電壓電流
power_settings = {
    '1.0KW': '220V_8A',
    '1.5KW': '285V_8A',
    '1.85KW': '325V_8A',
}

# 選擇實驗功率
power_level = '1.0KW'  # 可以修改為其他功率級別: '1.0KW', '1.5KW', '1.85KW'
power_detail = power_settings[power_level]
# 選擇實驗類型
experiment_type = 'pump_slice'  # 可選: 'pump_slice', 'fan_down', 'basic_feature'

# 設置實驗資料放置的資料夾
exp_name = '/home/inventec/Desktop/2KWCDU_修改版本/data_manage/Model_Training_data'

# 根據實驗類型選擇對應的設置檔案
if experiment_type == 'pump_slice':
    settings_file = f'{setting_folder}/experiment_setting_pump_slice_10%.csv'
    type_suffix = 'pump_cycle'
elif experiment_type == 'basic_feature':
    settings_file = f'{setting_folder}/experiment_setting_basic_feature.csv'
    type_suffix = 'basic_features'
elif experiment_type == 'fan_down':
    settings_file = f'{setting_folder}/experiment_setting_fan_down_10%.csv'
    type_suffix = 'fan_down'
else:
    print("請選擇正確的實驗類型")
    sys.exit()

# 自動生成實驗資料檔案名稱
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
exp_var = f'Training_data_GPU{power_level}({power_detail})_{type_suffix}_{timestamp}'

# 設置保存進度的jason文件名稱
experiment_progress = f'{exp_var}.json'

print(f"將使用設置檔案: {settings_file}")
print(f"數據將保存為: {exp_var}.csv")

#設定檔案標題
custom_headers = ['time', 'T_GPU', 'T_heater', 'T_CDU_in', 'T_CDU_out', 'T_env', 'T_air_in', 'T_air_out', 'TMP8', 'fan_duty', 'pump_duty']

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

adam.update_duty_cycles(fan_duty=fan_duty, pump_duty=pump_duty)

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
        
        # 設置風扇和泵速度
        fan1.set_all_duty_cycle(f"{setting['fan_speed']:03d}")
        fan2.set_all_duty_cycle(f"{setting['fan_speed']:03d}")
        pump.set_duty_cycle(f"{setting['pump_speed']:03d}")
        
        # 更新ADAM控制器的數據記錄
        adam.update_duty_cycles(fan_duty=setting['fan_speed'], pump_duty=setting['pump_speed'])
        
        # 等待指定的持續時間
        step_start_time = time.time()
        last_display_time = 0
        
        while time.time() - step_start_time < setting['duration']:
            current_time = time.time()
            # 每秒更新一次數據顯示
            if current_time - last_display_time >= 1:
                # 清除終端
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # 讀取和顯示數據
                with adam.buffer_lock:
                    current_data = adam.buffer.tolist()
                
                # 計算時間相關資訊
                elapsed_time = current_time - step_start_time
                remaining_time = setting['duration'] - elapsed_time
                total_remaining = calculate_total_duration(settings[i:]) - elapsed_time
                progress_percent = (elapsed_time / setting['duration']) * 100
                
                # 顯示基本實驗資訊
                print(f"=== 實驗資訊 ===")
                print(f"• 實驗類型: {experiment_type} ({type_suffix})")
                print(f"• GPU功率: {power_level} ({power_detail})")
                print(f"• 數據檔案: {exp_var}.csv")
                print("")
                
                # 顯示進度資訊
                progress_info = [
                    ["當前步驟", f"{i+1}/{len(settings)}"],
                    ["步驟設置", f"風扇: {setting['fan_speed']}%, 泵: {setting['pump_speed']}%, 持續: {setting['duration']}秒"],
                    ["步驟進度", f"{elapsed_time:.1f}秒/{setting['duration']}秒 ({progress_percent:.1f}%)"],
                    ["步驟剩餘時間", f"{timedelta(seconds=int(remaining_time))}"],
                    ["總剩餘時間", f"{timedelta(seconds=int(total_remaining))}"],
                    ["預計完成時間", f"{(datetime.now() + timedelta(seconds=total_remaining)).strftime('%Y-%m-%d %H:%M:%S')}"]
                ]
                print(tabulate(progress_info, headers=["項目", "數值"], tablefmt="grid"))
                print("")
                
                # 如果有溫度數據則顯示
                if current_data and len(current_data) >= 9:
                    temp_data = [
                        ["GPU溫度", f"{current_data[0]:.2f}°C"],
                        ["加熱器溫度", f"{current_data[1]:.2f}°C"],
                        ["CDU進水溫度", f"{current_data[2]:.2f}°C"],
                        ["CDU出水溫度", f"{current_data[3]:.2f}°C"],
                        ["環境溫度", f"{current_data[4]:.2f}°C"],
                        ["進氣溫度", f"{current_data[5]:.2f}°C"],
                        ["出氣溫度", f"{current_data[6]:.2f}°C"]
                    ]
                    print("=== 溫度數據 ===")
                    print(tabulate(temp_data, headers=["感測器", "溫度"], tablefmt="grid"))
                    print("")
                    
                    # 顯示控制設置
                    control_data = [
                        ["風扇轉速", f"{current_data[8]}%"],
                        ["泵轉速", f"{current_data[9]}%"]
                    ]
                    print("=== 當前控制設置 ===")
                    print(tabulate(control_data, headers=["裝置", "轉速"], tablefmt="grid"))
                
                # 更新最後顯示時間
                last_display_time = current_time
                
            # 短暫睡眠以減少CPU使用率
            time.sleep(0.1)
        
        save_progress(i + 1)

except KeyboardInterrupt:
    print("\n實驗被用戶中斷")
finally:
    adam.stop_threading('buffer')
    adam.stop_threading('adam')
    adam.closeport()
    fan1.set_all_duty_cycle(60)
    fan2.set_all_duty_cycle(60)
    pump.set_duty_cycle(100)
    print('實驗結束.')
