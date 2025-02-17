import time
import ADAMScontroller
import pwmcontroller as ctrl
import multi_channel_pwmcontroller as multi_ctrl
import json
import os
import csv
#import pid_test
from collections import deque
from simple_pid import PID

# 初始化控制器
adam_port = '/dev/ttyUSB0'
fan1_port = '/dev/ttyAMA4'
fan2_port = '/dev/ttyAMA5'
pump_port = '/dev/ttyAMA3'

exp_name = '0205_'
exp_var = '25'

custom_headers = ['time','GPU Watt','GPU temp','heater temp','CDU inlet temperature','CDU outlet temperature','env_temp','air_intlet','air_outlet','TMP8','fan_duty','pump_duty']

# 創建控制器物件
adam = ADAMScontroller.DataAcquisition(exp_name, exp_var, port=adam_port, csv_headers=custom_headers)
fan1 = multi_ctrl.multichannel_PWMController(fan1_port)
fan2 = multi_ctrl.multichannel_PWMController(fan2_port)
#pump = ctrl.XYKPWMController(pump_port)

# 設置ADAM控制器
adam.setup_directories()
adam.start_data_buffer()
adam.start_adam_controller()

def clamp(x):
    if abs(x) < 2:
        return 0
    else:
        return x

try:
    counter = 0
    flag = True
    delta =0
    target = 30
    reference =target + delta
    controller = PID(Kp=-6, Ki=-0.09, Kd=0, setpoint = reference, output_limits=(30,100),sample_time=3)
    #controller = pid_test.PID(6, 0.2, 27, 30, 100) # Kp, Ki, setpoint, lower. upper
    #T_return = deque ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], maxlen = 6)
    #kp = 5
    #M = 0
    while flag==True:
        
        Temperatures = adam.buffer.tolist()
        #delta = Temperatures[2] - Temperatures[3]
        #print("delta=", delta)
        #reference = target + delta
        #controller.setpoint = reference
        #controller = PID(Kp=-6, Ki=-0.01, Kd=0, setpoint = reference, output_limits=(30,100),sample_time=3) # Kp, Ki, setpoint, lower. upper
        #print("err=",reference-Temperatures[2])
        if any(Temperatures):
            print(f"GPU | {Temperatures[0]}", end=" | ")
            print(f"Heater | {Temperatures[1]}", end=" | ")
            print(f"CDU return | {Temperatures[2]}", end=" | ")
            print(f"CDU out | {Temperatures[3]}", end=" | ")
            print(f"envi | {Temperatures[4]}", end=" | ")
            print(f"air in | {Temperatures[5]}", end=" | ")
            print(f"air out | {Temperatures[6]}")

            #delta = Temperatures[2] - Temperatures[3]
            #reference = target + delta
            #controller.setpoint = reference
            
            fan = controller(Temperatures[3])
            #T_return.append(Temperatures[2])
            #M3 = T_return[5] - T_return[2]
            #print("return=",T_return)
            #M +=  kp * clamp(M3)
            #print("compensator=", M)
            #fan += M
            fan = round(fan)
            print("fan speed", fan)
            
            fan1.set_all_duty_cycle(fan)
            fan2.set_all_duty_cycle(fan)
            #pump.set_duty_cycle(100)
            adam.update_duty_cycles(fan, delta)
            counter += 1
            print("counter=", counter)
        time.sleep(3)
        

        
except KeyboardInterrupt:
    flag==False
    print("實驗被用戶中斷")
    fan1.close()
    fan2.close()
    #pump.close()
finally:
    adam.stop_threading('buffer')
    adam.stop_threading('adam')
    adam.closeport()
    fan1.set_all_duty_cycle(30)
    fan2.set_all_duty_cycle(30)
    #pump.set_duty_cycle(60)
    fan1.close()
    fan2.close()
    #pump.close()
    print('實驗結束.')