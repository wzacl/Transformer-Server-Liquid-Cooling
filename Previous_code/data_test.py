import time
import ADAMScontroller
import pwmcontroller as ctrl
import multi_channel_pwmcontroller as multi_ctrl
import json
import os
import csv

# 初始化控制器
adam_port = '/dev/ttyUSB0'
fan1_port = '/dev/ttyAMA0'
fan2_port = '/dev/ttyAMA2'
pump_port = '/dev/ttyAMA3'

exp_name = 'FunctionTest'
exp_var = 'test2'

custom_headers = ['time','GPU Watt','GPU temp','heater temp','CDU inlet temperature','CDU outlet temperature','env_temp','air_intlet','air_outlet','TMP8','fan_duty','pump_duty']

# 創建控制器物件
adam = ADAMScontroller.DataAcquisition(exp_name, exp_var, port=adam_port, csv_headers=custom_headers)
fan1 = multi_ctrl.multichannel_PWMController(fan1_port)
fan2 = multi_ctrl.multichannel_PWMController(fan2_port)
pump = ctrl.XYKPWMController(pump_port)

# 設置ADAM控制器
adam.setup_directories()
adam.start_data_buffer()
adam.start_adam_controller()

try:
    #Temperatures = adam.buffer.tolist()
    #print("temperature")
    for i in range(5):
        fan1.set_all_duty_cycle(30+i)
        fan2.set_all_duty_cycle(30+i)
        
        pump.set_duty_cycle(30+i)
        Temperatures = adam.buffer.tolist()
        print(f"GPU | {Temperatures[0]}", end=" | ")
        print(f"Heater | {Temperatures[1]}", end=" | ")
        print(f"CDU return | {Temperatures[2]}", end=" | ")
        print(f"CDU out | {Temperatures[3]}", end=" | ")
        print(f"envi | {Temperatures[4]}", end=" | ")
        print(f"air in | {Temperatures[5]}", end=" | ")
        print(f"air out | {Temperatures[6]}")
        # print("Temperature", Temperatures)
        time.sleep(3)

    
    #adam.update_duty_cycles(60, 60)
    
    
        
except KeyboardInterrupt:
    print("實驗被用戶中斷")
finally:
    adam.stop_threading('buffer')
    adam.stop_threading('adam')
    adam.closeport()
    fan1.set_all_duty_cycle(40)
    fan2.set_all_duty_cycle(40)
    pump.set_duty_cycle(50)
    print('實驗結束.')