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
    """數據採集類，用於從ADAM控制器讀取數據並保存到CSV文件。
    
    該類提供了與ADAM控制器通信、數據緩存和數據記錄的功能。
    
    Attributes:
        exp_name (str): 實驗數據保存的目錄路徑。
        port (str): ADAM控制器的串口路徑。
        buffer_size (int): 數據緩衝區大小。
        data_update_rate (float): 數據更新頻率（秒）。
        buffer (numpy.ndarray): 存儲溫度和控制信號的緩衝區。
        flag_buffer (bool): 控制數據緩衝線程的標誌。
        flag_adam (bool): 控制ADAM讀取線程的標誌。
        buffer_lock (threading.Lock): 緩衝區的線程鎖。
        ser_adam1 (serial.Serial): ADAM控制器的串口對象。
        exp_var (str): 實驗數據文件名。
        csv_headers (list): CSV文件的標題行。
        th_buffer (threading.Thread): 數據緩衝線程。
        th_adam (threading.Thread): ADAM讀取線程。
        data_updated_event (threading.Event): 數據更新事件。
    """
    
    def __init__(self, 
                 exp_var,  # 將沒有默認值的參數放在前面
                 exp_name='./experiment',  # 帶默認值的參數放在後面
                 port='/dev/ttyUSB0',
                 buffer_size=8,
                 data_update_rate=1,
                 csv_headers=None):
        """初始化DataAcquisition類的實例。
        
        Args:
            exp_var (str): 實驗數據文件名。
            exp_name (str, optional): 實驗數據保存的目錄路徑。默認為'./experiment'。
            port (str, optional): ADAM控制器的串口路徑。默認為'/dev/ttyUSB0'。
            buffer_size (int, optional): 數據緩衝區大小。默認為8。
            data_update_rate (float, optional): 數據更新頻率（秒）。默認為1。
            csv_headers (list, optional): CSV文件的標題行。如果為None，則使用默認標題。
        """
        self.exp_name = exp_name
        self.port = port
        self.buffer_size = buffer_size
        self.data_update_rate = data_update_rate
        self.buffer = np.zeros(buffer_size + 4)  # 增加 2 個位置用於風扇和泵的工作週期，2個位置用於目標溫度
        self.flag_buffer = True
        self.flag_adam = True
        self.buffer_lock = threading.Lock()
        self.ser_adam1 = None
        self.exp_var= exp_var
        self.csv_headers = csv_headers or ['time', 'T_GPU', 'T_heater', 'T_CDU_in', 'T_CDU_out', 'T_env', 'T_air_in', 'T_air_out', 'TMP8', 'fan_duty', 'pump_duty', 'GPU_Watt(KW)']
        self.th_buffer = None
        self.th_adam = None
        self.data_updated_event = threading.Event()
    

    def setup_directories(self):
        """創建實驗數據保存目錄。
        
        如果目錄不存在，則創建它。
        """
        if not os.path.exists(self.exp_name):
            os.makedirs(self.exp_name)

    def data_buffer(self):
        """將數據緩衝區的內容寫入CSV文件。
        
        這個方法在單獨的線程中運行，定期將緩衝區的數據寫入CSV文件。
        如果文件不存在，則創建文件並寫入標題行。
        """
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
        """啟動數據緩衝線程。
        
        創建並啟動一個新線程來運行data_buffer方法。
        """
        self.th_buffer = threading.Thread(target=self.data_buffer)
        self.th_buffer.start()
        print('Start buffer')

    def value_extraction(self, value):
        """從ADAM控制器返回的原始數據中提取數值。
        
        Args:
            value (bytes): ADAM控制器返回的原始數據。
            
        Returns:
            list: 提取的數值列表。
        """
        value = value.decode('utf-8')
        value = value[2:-1]
        value = value.split('+')
        return value

    def reading_from_adam(self):
        """從ADAM控制器讀取數據。
        
        這個方法在單獨的線程中運行，定期從ADAM控制器讀取數據並更新緩衝區。
        """
        while self.flag_adam:
            try:
                self.ser_adam1.write(b'#01\r')
                if self.ser_adam1.in_waiting > 0:
                    data_raw_1 = self.ser_adam1.read_until(b'\r')
                    data_1 = self.value_extraction(data_raw_1)
                    with self.buffer_lock:  # 確保資料同步
                        self.buffer[:len(data_1)] = [float(v) for v in data_1]
                    #print("Updated buffer:", self.buffer)  # Debug: Print the updated buffer
                self.data_updated_event.set()
                time.sleep(self.data_update_rate)
            except Exception as e:
                print(f"Error reading from ADAM: {e}")
        print('Stop reading from ADAM')

    def start_adam_controller(self):
        """啟動ADAM控制器讀取線程。
        
        打開串口並啟動一個新線程來運行reading_from_adam方法。
        """
        self.ser_adam1 = self.openport(self.port, 9600, timeout=None)
        if self.ser_adam1 and self.ser_adam1.is_open:
            self.th_adam = threading.Thread(target=self.reading_from_adam)
            self.th_adam.start()
            print('Start ADAM controller')
        else:
            print('Failed to open ADAM port')

    def openport(self, portx, bps=9600, timeout=None):
        """打開串口。
        
        Args:
            portx (str): 串口路徑。
            bps (int, optional): 波特率。默認為9600。
            timeout (float, optional): 超時時間（秒）。默認為None。
            
        Returns:
            serial.Serial or None: 如果成功，返回串口對象；否則返回None。
        """
        try:
            ser = serial.Serial(portx, bps, timeout=timeout)
            print('Start pyserial ' + portx)
            return ser
        except Exception as e:
            print(f"Error opening serial port {portx}: {e}")
            return None

    def closeport(self):
        """關閉串口。
        
        關閉與ADAM控制器的串口連接。
        """
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
        """停止指定的線程。
        
        Args:
            work (str): 要停止的線程類型，可以是'buffer'或'adam'。
            
        Returns:
            str: 確認消息。
        """
        if work == 'buffer':
            self.flag_buffer = False
        elif work == 'adam':
            self.flag_adam = False
        else:
            print('please input the threading you want to stop')
        return 'stop threading:'

    def join_threads(self):
        """等待所有線程完成。
        
        等待數據緩衝線程和ADAM讀取線程完成。
        """
        if self.th_buffer:
            self.th_buffer.join()
        if self.th_adam:
            self.th_adam.join()

    def update_duty_cycles(self, fan_duty=None, pump_duty=None):
        """更新風扇和泵的工作週期。
        
        Args:
            fan_duty (float, optional): 風扇的工作週期。如果為None，則不更新。
            pump_duty (float, optional): 泵的工作週期。如果為None，則不更新。
        """
        with self.buffer_lock:
            current_values = self.buffer[-4:-2]
            if fan_duty is not None:
                current_values[0] = fan_duty
            if pump_duty is not None:
                current_values[1] = pump_duty
            self.buffer[-4:-2] = current_values
    def update_target_temperature(self, target_temperature=None,gpu_target_temperature=None):
        """更新目標溫度。
        
        Args:
            target_temperature (float, optional): 目標溫度。如果為None，則不更新。
            gpu_target_temperature (float, optional): GPU目標溫度。如果為None，則不更新。
        """
        with self.buffer_lock:
            current_values = self.buffer[-2:] 
            if target_temperature is not None:
                current_values[0] = target_temperature
            if gpu_target_temperature is not None:
                current_values[1] = gpu_target_temperature
            self.buffer[-2:] = current_values
    def update_else_data(self, data_list):
        """更新額外的數據。
        
        Args:
            data_list (list or float): 要添加到緩衝區的數據。
        """
        with self.buffer_lock:
            if isinstance(data_list, list):
                for data in data_list:
                    self.buffer = np.append(self.buffer, data)
            else:
                self.buffer = np.append(self.buffer, data_list)

    def plot_experiment_results(self):
        """繪製實驗結果圖。
        
        從CSV文件讀取數據並繪製溫度和控制信號的圖表。
        圖表將保存為PNG文件。
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

    def join_threads(self):
        """等待所有線程完成。
        
        等待數據緩衝線程和ADAM讀取線程完成。
        """
        if self.th_buffer:
            self.th_buffer.join()
        if self.th_adam:
            self.th_adam.join()

    def start_adam(self):
        """啟動ADAM控制器和數據緩衝。
        
        設置目錄，啟動數據緩衝線程和ADAM讀取線程。
        """
        self.setup_directories()
        self.start_data_buffer()
        self.start_adam_controller()
        
    def stop_adam(self):
        """停止ADAM控制器和數據緩衝。
        
        停止所有線程，關閉串口連接。
        """
        self.stop_threading('buffer')
        self.stop_threading('adam')
        self.join_threads()
        self.closeport()
        print('Experiment ended.')

            
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
        data_acquisition.stop_adam()
