import time
import sys
sys.path.append('/home/inventec/Desktop/2KWCDU/code_manage/Control_Unit')

import ADAMScontroller
import pwmcontroller as ctrl
import multi_channel_pwmcontroller as multi_ctrl
from simple_pid import PID

# 初始化控制器
adam_port = '/dev/ttyUSB0'
fan1_port = '/dev/ttyAMA4'
fan2_port = '/dev/ttyAMA5'
pump_port = '/dev/ttyAMA3'

exp_name = '/home/inventec/Desktop/2KWCDU/data_collection/PID_pump'
exp_var = '250212PID-pump'

custom_headers = ['time', 'T_GPU', 'T_heater', 'T_CDU_in', 'T_CDU_out', 'T_env', 'T_air_in', 'T_air_out', 'TMP8', 'fan_duty', 'pump_duty','T_w_delta', 'GPU_Watt']

# 創建控制器物件
adam = ADAMScontroller.DataAcquisition(exp_name, exp_var, port=adam_port, csv_headers=custom_headers)
fan1 = multi_ctrl.multichannel_PWMController(fan1_port)
fan2 = multi_ctrl.multichannel_PWMController(fan2_port)
pump = ctrl.XYKPWMController(pump_port)

# 設置ADAM控制器
adam.setup_directories()
adam.start_data_buffer()
adam.start_adam_controller()

# 設置風扇初始轉速
fan_duty=80
fan1.set_all_duty_cycle(fan_duty)
fan2.set_all_duty_cycle(fan_duty)

try:
    counter = 0
    flag = True
    delta = 0
    target = 68
    reference = target + delta
    sample_time = 2  # 定義取樣時間
    controller = PID(Kp=-10, Ki=-0.8, Kd=0, setpoint=target, output_limits=(40, 100), sample_time=sample_time)

    while flag:
        Temperatures = adam.buffer.tolist()
        if any(Temperatures):
            print(f"T_GPU | {Temperatures[0]}", end=" | ")
            print(f"T_heater | {Temperatures[1]}", end=" | ")
            print(f"T_CDU_in | {Temperatures[2]}", end=" | ")
            print(f"T_CDU_out | {Temperatures[3]}", end=" | ")
            print(f"T_env | {Temperatures[4]}", end=" | ")
            print(f"T_air_in | {Temperatures[5]}", end=" | ")
            print(f"T_air_out | {Temperatures[6]}\n")

            delta = Temperatures[2] - Temperatures[3]
            reference = target + delta
            controller.setpoint = reference

            pump_duty = round(controller(Temperatures[0])/2)*2
            print(f"pump  speed={pump_duty}\n")

            adam.update_duty_cycles(fan_duty,pump_duty)
            adam.update_else_data(delta)
            counter += 1
            print(f"counter={counter}")
        time.sleep(sample_time)

except KeyboardInterrupt:
    print("實驗被用戶中斷")
except Exception as e:
    print(f"發生錯誤: {e}")
except KeyboardInterrupt:
    print("\n實驗被用戶中斷")
finally:
    adam.stop_threading('buffer')
    adam.stop_threading('adam')
    adam.closeport()
    fan1.set_all_duty_cycle(40)
    fan2.set_all_duty_cycle(40)
    pump.set_duty_cycle(100)
    print("實驗結束，所有裝置恢復到安全狀態。")
