# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 13:06:36 2022

@author: IEC100631
"""

import serial
import threading
import time
import csv
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
#from tabulate import tabulate
class DataAcquisition:
    def __init__(self, 
                 exp_var,  # 將沒有默認值的參數放在前面
                 exp_name='./experiment',  # 帶默認值的參數放在後面
                 port='/dev/ttyUSB0',
                 buffer_size=8,
                 data_update_rate=1,
                 csv_headers=None):
        self.exp_name = exp_name
        self.port = port
        self.buffer_size = buffer_size
        self.data_update_rate = data_update_rate
        self.buffer = np.zeros(buffer_size + 2)  # 增加 2 個位置用於風扇和泵的工作週期
        self.flag_buffer = True
        self.flag_adam = True
        self.buffer_lock = threading.Lock()
        self.ser_adam1 = None
        self.exp_var= exp_var
        self.csv_headers = csv_headers or ['time', 'T_GPU', 'T_heater', 'T_CDU_in', 'T_CDU_out', 'T_env', 'T_air_in', 'T_air_out', 'TMP8', 'fan_duty', 'pump_duty', 'GPU_Watt(KW)']
        self.setup_directories()
    

    def setup_directories(self):
        
        if not os.path.exists(self.exp_name):
            os.makedirs(self.exp_name)

    def data_buffer(self):
        
        file_path = os.path.join(self.exp_name, self.exp_var +'.csv')
        try:
            with open(file_path, mode='a+', newline='') as csv_file:
                writer = csv.writer(csv_file)
                if csv_file.tell() == 0:  # 如果文件是空的,寫入標題
                    writer.writerow(self.csv_headers)
                print('Buffer start!')
                while self.flag_buffer:
                    with self.buffer_lock:  # 確保資料同步
                        new_data = [time.ctime()]
                        new_data.extend(self.buffer)
                        #print("Writing data to CSV:", new_data)  # Debug: Print the data being written
                        
                        current_time = time.ctime()
                        #table_header=['GPU溫度','加熱器溫度','CDU出水溫(GPU入水溫)','CDU回水溫(GPU出水溫)','t4','環境溫度','t6','t7','t8','t9']
                        new_data = [current_time] + self.buffer.tolist()
                        #將註解部分打開後可以使用表格的方式將寫入csv檔案的資料呈現
                        #table_data = [new_data] 
                        #print("Writing data to CSV:")  # Debug: Print the data being written
                        #print(tabulate(table_data, headers=table_header, tablefmt='grid'))
                        writer.writerow(new_data)
                        csv_file.flush()
                    time.sleep(1)  # 這個變數控制暫存器儲存資訊的頻率
        except Exception as e:
            print(f"Error writing to CSV file: {e}")
        finally:
            print('Buffer stop')

    def start_data_buffer(self):
        th_buffer = threading.Thread(target=self.data_buffer)
        th_buffer.start()
        print('Start buffer')

    def value_extraction(self, value):
        value = value.decode('utf-8')
        value = value[2:-1]
        value = value.split('+')
        return value

    def reading_from_adam(self):
        while self.flag_adam:
            try:
                self.ser_adam1.write(b'#01\r')
                if self.ser_adam1.in_waiting > 0:
                    data_raw_1 = self.ser_adam1.read_until(b'\r')
                    data_1 = self.value_extraction(data_raw_1)
                    with self.buffer_lock:  # 確保資料同步
                        self.buffer[:len(data_1)] = [float(v) for v in data_1]
                    #print("Updated buffer:", self.buffer)  # Debug: Print the updated buffer
                time.sleep(self.data_update_rate)
            except Exception as e:
                print(f"Error reading from ADAM: {e}")
        print('Stop reading from ADAM')

    def start_adam_controller(self):
        self.ser_adam1 = self.openport(self.port, 9600, timeout=None)
        if self.ser_adam1 and self.ser_adam1.is_open:
            th_adam = threading.Thread(target=self.reading_from_adam)
            th_adam.start()
            print('Start ADAM controller')
        else:
            print('Failed to open ADAM port')

    def openport(self, portx, bps=9600, timeout=None):
        try:
            ser = serial.Serial(portx, bps, timeout=timeout)
            print('Start pyserial ' + portx)
            return ser
        except Exception as e:
            print(f"Error opening serial port {portx}: {e}")
            return None

    def closeport(self):
        if self.ser_adam1:
            try:
                self.ser_adam1.close()
                if not self.ser_adam1.is_open:
                    print('Close pyserial')
                else:
                    print('Fail closing')
            except Exception as e:
                print(f"Error closing serial port: {e}")

    def stop_threading(self, work):
        if work == 'buffer':
            self.flag_buffer = False
        elif work == 'adam':
            self.flag_adam = False
        else:
            print('please input the threading you want to stop')
        return('stop threading:')

    def update_duty_cycles(self, fan_duty, pump_duty):
        with self.buffer_lock:
            self.buffer[-2:] = [fan_duty, pump_duty]
    def update_else_data(self, data_list):
        """
        更新額外的數據
        :param data_list: 包含要更新的數據的列表
        """
        with self.buffer_lock:
            if isinstance(data_list, list):
                for data in data_list:
                    self.buffer = np.append(self.buffer, data)
            else:
                self.buffer = np.append(self.buffer, data_list)

    def plot_experiment_results(self):
        """
        繪製實驗結果圖
        """
        try:
            # 使用新版本 pandas 的參數名稱
            df = pd.read_csv(
                os.path.join(self.exp_name, self.exp_var + '.csv'),
                on_bad_lines='skip'  # 替換 error_bad_lines
            )
            
            # 檢查數據列是否存在
            required_columns = ['T_GPU', 'T_CDU_out', 'fan_duty', 'pump_duty']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("缺少必要的數據列")
            
            # 丟棄前兩秒的資料
            df = df.iloc[2:].reset_index(drop=True)
            
            plt.figure(figsize=(12,8))
            
            # 創建主圖
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            
            # 繪製溫度數據
            ax1.plot(df['T_GPU'], 'r-', label='GPU Temperature')
            ax1.plot(df['T_CDU_out'], 'b-', label='CDU Outlet Temperature')
            
            # 繪製控制信號
            ax2.plot(df['fan_duty'], 'g--', label='Fan Duty')
            ax2.plot(df['pump_duty'], 'y--', label='Pump Duty')
            
            # 設置標籤和標題
            ax1.set_xlabel('Time(s)')
            ax1.set_ylabel('Temperature(°C)')
            ax2.set_ylabel('Duty Cycle(%)')
            plt.title('Experiment Temperature Response')
            
            # 合併圖例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
            
            # 保存圖片
            plt.savefig(os.path.join(self.exp_name, self.exp_var + '.png'))
            plt.show()
            print(f"實驗結果圖已保存到 {os.path.join(self.exp_name, self.exp_var)}.png")
            
        except Exception as e:
            print(f"繪圖時發生錯誤: {e}")
            print("請檢查數據文件格式是否正確")

    def start_adam(self):
        self.setup_directories()
        self.start_data_buffer()
        self.start_adam_controller()

            
if __name__ == "__main__":
    exp_name = '2024.8.30測試'  # 實驗存取資料夾
    exp_var  = 'test1128'  #實驗變數
    data_acquisition = DataAcquisition(exp_var=exp_var,exp_name=exp_name,port='/dev/ttyUSB0')

    print('Start experiment~!')
    data_acquisition.setup_directories()#記得修改檔案名稱
    data_acquisition.start_data_buffer()
    data_acquisition.start_adam_controller()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        data_acquisition.stop_threading('buffer')
        data_acquisition.stop_threading('adam')
        data_acquisition.closeport()
        print('Experiment ended.')
