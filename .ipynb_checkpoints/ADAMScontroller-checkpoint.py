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
from tabulate import tabulate
class DataAcquisition:
    def __init__(self,exp_nam,exp_var, port='/dev/ttyUSB0', buffer_size=10, data_update_rate=1):
        self.exp_nam = exp_nam
        self.port = port
        self.buffer_size = buffer_size
        self.data_update_rate = data_update_rate
        self.buffer = np.zeros(buffer_size)
        self.flag_buffer = True
        self.flag_adam = True
        self.buffer_lock = threading.Lock()
        self.ser_adam1 = None
        self.exp_var= exp_var
        self.path= './experiment'
        self.path1 = f"{self.path}/{self.exp_nam}"
        self.setup_directories()
    

    def setup_directories(self):
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def data_buffer(self):
        #file_path = os.path.join(self.path1, time.ctime().replace(" ", "_").replace(":", "_") + '.csv')
        file_path = os.path.join(self.path, self.exp_var +'.csv')
        try:
            with open(file_path, mode='a+', newline='') as csv_file:
                writer = csv.writer(csv_file)
                print('Buffer start!')
                while self.flag_buffer:
                    with self.buffer_lock:  # 確保資料同步
                        #new_data = [time.ctime()]
                        #new_data.extend(self.buffer)
                        #print("Writing data to CSV:", new_data)  # Debug: Print the data being written
                        
                        current_time = time.ctime()
                        table_header=['GPU溫度','加熱器溫度','CDU出水溫(GPU入水溫)','CDU回水溫(GPU出水溫)','t4','環境溫度','t6','t7','t8','t9']
                        new_data = [current_time] + self.buffer.tolist()
                        #將註解部分打開後可以使用表格的方式將寫入csv檔案的資料呈現
                        table_data = [new_data] 
                        print("Writing data to CSV:")  # Debug: Print the data being written
                        print(tabulate(table_data, headers=table_header, tablefmt='grid'))
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

if __name__ == "__main__":
    exp_name = '2024.8.30測試'  # 實驗名稱
    exp_var  = '希望能正常使用'
    data_acquisition = DataAcquisition(exp_name,exp_var,port='COM6')

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