'''''
本程式碼主要用於隨機測試資料蒐集，本研究中的晶片瓦數對應的電源供應器參數設置如下
1KW：220V_8A
1.5KW：285V_8A
1.9KW：332V_8A
'''''
import time
import sys
sys.path.append('/home/inventec/Desktop/2KWCDU')
import ADAMScontroller
import pwmcontroller as ctrl
import multi_channel_pwmcontroller as multi_ctrl
import json
import os
import csv
import random
from datetime import datetime, timedelta


# 初始化控制器
adam_port = '/dev/ttyUSB0'
fan1_port = '/dev/ttyAMA5'
fan2_port = '/dev/ttyAMA4'
pump_port = '/dev/ttyAMA3'
#設置實驗資料放置的資料夾
exp_name = '/home/inventec/Desktop/2KWCDU/data_collection/Model_testing_data'
#設置實驗資料檔案名稱
exp_var = 'Testingdata_GPU1.5W(218V_8A)_5%_1'

custom_headers = ['time','T_GPU','T_heater','T_CDU_in','T_CDU_out','T_env','T_air_in','T_air_out','TMP8','fan_duty','pump_duty','GPU_Watt(KW)']
experiment_set='/home/inventec/Desktop/2KWCDU/data_collection/experiment_setting_random.csv'
# 創建控制器物件
adam = ADAMScontroller.DataAcquisition(exp_name=exp_name, exp_var=exp_var, port=adam_port, csv_headers=custom_headers)
fan1 = multi_ctrl.multichannel_PWMController(fan1_port)
fan2 = multi_ctrl.multichannel_PWMController(fan2_port)
pump = ctrl.XYKPWMController(pump_port)

# 設置ADAM控制器
adam.setup_directories()
adam.start_data_buffer()
adam.start_adam_controller()

def read_settings(file_path):
    settings = []
    try:
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # 跳過標題行
            for row in csv_reader:
                if len(row) < 2:
                    print(f"警告：行 {csv_reader.line_num} 格式不正確，跳過此行")
                    continue
                settings.append({
                    'experiment_iteration': int(row[0]),
                    'experiment_duration': int(row[1])
                })
    except FileNotFoundError:
        print(f"錯誤：找不到文件 '{file_path}'")
    except csv.Error as e:
        print(f"錯誤：讀取 CSV 文件時出錯 - {e}")
    except ValueError as e:
        print(f"錯誤：轉換數據類型時出錯 - {e}")
    
    if not settings:
        print("警告：沒有讀取到有效的設置數據")
    
    return settings

def save_progress(step):
    with open('random_experiment_progress.json', 'w') as f:
        json.dump({'current_step': step}, f)

def load_progress():
    if os.path.exists('random_experiment_progress.json'):
        with open('random_experiment_progress.json', 'r') as f:
            return json.load(f)['current_step']
    return 0

class RandomExperiment:
    def __init__(self, adam, fan1, fan2, pump, total_experiment_time):
        self.adam = adam
        self.fan1 = fan1
        self.fan2 = fan2
        self.pump = pump
        self.total_experiment_time = total_experiment_time
        self.current_fan_duty=0
        self.current_pump_duty=0
        self.keep_random_exp = True
        self.experiment_start_time = None
        self.experiment_progress = 0
        self.min_adjustment_interval = 5  # 最小調整間隔，單位為秒
        self.start_time = None
        self.estimated_end_time = None

    def fan_set_duty_all(self, duty):
        self.fan1.set_all_duty_cycle(duty)
        self.fan2.set_all_duty_cycle(duty)
        self.current_fan_duty=duty
        

    def pump_set_duty(self, duty):
        self.pump.set_duty_cycle(duty)
        self.current_pump_duty=duty

    def update_adam_duty(self):
        self.adam.update_duty_cycles(self.current_fan_duty, self.current_pump_duty)

    def random_experiment_loop(self):
        fan_speeds = [30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]  # 隨機從30到100選擇一個轉速，變動單位為5
        pump_speeds =  [40,45,50,55,60,65,70,75,80,85,90,95,100]  # 隨機從40到100選擇一個轉速，變動單位為2
        
        self.start_time = datetime.now()
        self.estimated_end_time = self.start_time + timedelta(seconds=self.total_experiment_time)
        
        self.fan_duty = random.choice(fan_speeds)
        self.pump_duty = random.choice(pump_speeds)
        self.fan_set_duty_all(self.fan_duty)
        self.pump_set_duty(self.pump_duty)

        fan_next_change = time.time() + random.randint(7, 100)
        pump_next_change = time.time() + random.randint(3, 100)
        last_adjustment_time = time.time()

        while self.keep_random_exp:
            current_time = time.time()
            elapsed_time = current_time - self.experiment_start_time
            remaining_time = self.total_experiment_time - elapsed_time
            self.experiment_progress = min(100, (elapsed_time / self.total_experiment_time) * 100)
            
            # 每10秒更新一次進度信息
            if int(elapsed_time) % 10 == 0:
                print(f"\r進度: {self.experiment_progress:.1f}% | "
                      f"剩餘時間: {timedelta(seconds=int(remaining_time))} | "
                      f"預計完成時間: {self.estimated_end_time.strftime('%H:%M:%S')}", 
                      end="", flush=True)
            
            if elapsed_time >= self.total_experiment_time:
                self.keep_random_exp = False
                break
            
            if current_time >= fan_next_change and current_time - last_adjustment_time >= self.min_adjustment_interval:
                fan_duty = random.choice(fan_speeds)
                self.fan_set_duty_all(fan_duty)
                fan_next_change = current_time + random.randint(10, 100)
                last_adjustment_time = current_time

            elif current_time >= pump_next_change and current_time - last_adjustment_time >= self.min_adjustment_interval:
                pump_duty = random.choice(pump_speeds)
                self.pump_set_duty(pump_duty)
                pump_next_change = current_time + random.randint(5, 100)
                last_adjustment_time = current_time

            # 每隔 update_interval 秒更新一次ADAM控制器
            
            self.update_adam_duty()
                

            time.sleep(0.1)  # 縮短檢查間隔，以確保及時更新

        self.update_adam_duty()  # 確保最後的設置也被記錄

try:
    settings = read_settings(experiment_set)
    if not settings:
        raise ValueError("沒有有效的實驗設置")
    
    current_step = load_progress()
    
    # 計算總實驗時間
    total_duration = sum(setting['experiment_duration'] for setting in settings[current_step:])
    estimated_total_end_time = datetime.now() + timedelta(seconds=total_duration)
    print(f"開始執行實驗，共 {len(settings)-current_step} 組")
    print(f"預計總實驗時間: {timedelta(seconds=total_duration)}")
    print(f"預計總完成時間: {estimated_total_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    for i in range(current_step, len(settings)):
        setting = settings[i]
        
        print(f"\n開始隨機試驗 {setting['experiment_iteration']}: 持續時間 {setting['experiment_duration']} 秒")
        print(f"剩餘組數: {len(settings)-i-1}")
        
        random_exp = RandomExperiment(adam, fan1, fan2, pump, setting['experiment_duration'])
        random_exp.experiment_start_time = time.time()
        random_exp.random_experiment_loop()
        
        save_progress(i + 1)
        print(f"\n隨機試驗 {setting['experiment_iteration']} 完成")

except KeyboardInterrupt:
    print("實驗被用戶中斷")
except ValueError as e:
    print(f"錯誤：{e}")
except Exception as e:
    print(f"發生未預期的錯誤：{e}")
finally:
    adam.stop_threading('buffer')
    adam.stop_threading('adam')
    adam.closeport()
    fan1.set_all_duty_cycle(100)
    fan2.set_all_duty_cycle(100)
    pump.set_duty_cycle(100)
    print('實驗結束.')
