'''''
GB_PID_pump.py

GB_PID控制器，利用Guaranteed Bounded PID控制泵轉速

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
import os
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Control_Unit')
import ADAMScontroller
import pwmcontroller as ctrl
import multi_channel_pwmcontroller as multi_ctrl
from simple_pid import PID



class GB_PID_pump:
    def __init__(self,target, Guaranteed_Bounded_PID_range=0.5,sample_time=1):
        self.target = target
        self.GB = Guaranteed_Bounded_PID_range
        self.sample_time = sample_time
        self.controller = PID(
            Kp=-5,  # 調整比例增益
            Ki=-0.5,  # 調整積分增益
            Kd=0,  # 添加微分項以改善響應
            setpoint=target,
            output_limits=(40, 100),
            sample_time=sample_time
        )
        self.controller.setpoint = target
    def GB_PID(self, T_real, target):
        delta = abs(T_real - target)
        if delta <= self.GB:
            return target  # 當誤差在範圍內，返回目標溫度
        else:
            return T_real  # 當誤差超出範圍，返回實際溫度

if __name__ == '__main__':
        # 初始化控制器
    adam_port = '/dev/ttyUSB0'
    fan1_port = '/dev/ttyAMA4'
    fan2_port = '/dev/ttyAMA5'
    pump_port = '/dev/ttyAMA3'
    #資料儲存位置(不要動)
    exp_name = '/home/inventec/Desktop/2KWCDU_修改版本/data_manage/PID_pump'
    #實驗檔案名稱(可自行更動)
    exp_var = '250212PID-pump'
    #修改自訂資料表頭，確保列數一致
    custom_headers = [
        'time', 
        'T_GPU', 
        'T_heater', 
        'T_CDU_in', 
        'T_CDU_out', 
        'T_env', 
        'T_air_in', 
        'T_air_out', 
        'TMP8', 
        'fan_duty', 
        'pump_duty',
        'GPU_Watt'
    ]
    # 創建控制器物件
    adam = ADAMScontroller.DataAcquisition(
        exp_var=exp_var,
        exp_name=exp_name,
        port=adam_port,
        csv_headers=custom_headers
    )
    fan1 = multi_ctrl.multichannel_PWMController(fan1_port)
    fan2 = multi_ctrl.multichannel_PWMController(fan2_port)
    pump = ctrl.XYKPWMController(pump_port)

    # 設置初始轉速
    pump_duty=40
    pump.set_duty_cycle(pump_duty)
    fan_duty=60
    fan1.set_all_duty_cycle(fan_duty)
    fan2.set_all_duty_cycle(fan_duty)
    # 設置ADAM控制器

   

    try:
        counter = 0
        flag = True
        target = 68
        sample_time = 1  # 定義取樣時間
        Guaranteed_Bounded_PID_range = 0.5
        Controller = GB_PID_pump( target, Guaranteed_Bounded_PID_range, sample_time)
        adam.start_adam()
        while flag:
            Temperatures = adam.buffer.tolist()
            if any(Temperatures):
                # 獲取溫度數據
                T_GPU = Temperatures[0]  # 定義 T_GPU 變量
                T_CDU_out = Temperatures[3]
                T_env = Temperatures[4]
                
                print(f"T_GPU: {T_GPU} | T_CDU_out: {T_CDU_out} | T_env: {T_env}")
                print(f"counter: {counter} | pump speed: {pump_duty}")
                print("----------------------------------------")

                # 使用 GB_PID 計算控制輸出
                control_temp = Controller.GB_PID(T_GPU, target)
                pump_duty = round(Controller.controller(control_temp)/10)*10
                    
                # 更新泵的轉速
                pump.set_duty_cycle(pump_duty)
                adam.update_duty_cycles(fan_duty, pump_duty)

                counter += 1

            time.sleep(sample_time)

    except KeyboardInterrupt:
        print("實驗被用戶中斷")
    except Exception as e:
        print(f"發生錯誤: {e}")

    finally:
        adam.stop_threading('buffer')
        adam.stop_threading('adam')
        adam.closeport()
        fan1.set_all_duty_cycle(40)
        fan2.set_all_duty_cycle(40)
        pump.set_duty_cycle(100)
        print("實驗結束，所有裝置恢復到安全狀態。")
        # 繪製實驗結果圖
        print("繪製實驗結果圖...")
        adam.plot_experiment_results()
        print("實驗結果圖繪製完成。")
