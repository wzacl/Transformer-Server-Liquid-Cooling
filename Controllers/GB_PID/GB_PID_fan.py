"""
GB_PID_fan.py

GB_PID控制器，利用Guaranteed Bounded PID控制風扇轉速。

本模塊實現了一個保證有界PID控制器，用於控制冷卻系統中的風扇轉速。
當溫度誤差在指定範圍內時，控制器使用目標溫度作為輸入；當誤差超出範圍時，
使用實際溫度作為輸入，以確保系統穩定性。

本研究中的晶片瓦數對應的電源供應器參數設置如下：
    1KW：220V_8A
    1.5KW：285V_8A
    1.9KW：332V_8A

對應的風扇與泵最低轉速如下：
    泵：40% duty cycle
    風扇：30% duty cycle
"""
import time
import sys
import os
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Control_Unit')
import ADAMScontroller
import pwmcontroller as ctrl
import multi_channel_pwmcontroller as multi_ctrl
from simple_pid import PID



class GB_PID_fan:
    """保證有界PID控制器類，用於控制風扇轉速。
    
    該控制器實現了一種改進的PID控制策略，當溫度誤差在指定範圍內時使用目標溫度作為
    控制輸入，當誤差超出範圍時使用實際溫度，以提高系統穩定性和控制效果。
    
    Attributes:
        GB (float): 保證有界的範圍值，定義可接受的溫度誤差範圍。
        sample_time (float): 控制器的取樣時間，單位為秒。
        controller (PID): 內部PID控制器實例，用於計算控制輸出。
    """
    
    def __init__(self, target, Guaranteed_Bounded_PID_range=0.5, sample_time=1):
        """初始化GB_PID_fan控制器。
        
        Args:
            target (float): 目標溫度值。
            Guaranteed_Bounded_PID_range (float, optional): 保證有界的範圍值。默認為0.5。
            sample_time (float, optional): 控制器取樣時間，單位為秒。默認為1。
        """
        self.GB = Guaranteed_Bounded_PID_range
        self.sample_time = sample_time
        self.controller = PID(
            Kp=-2,  # 調整比例增益
            Ki=-0.2,  # 調整積分增益
            Kd=0,  # 添加微分項以改善響應
            setpoint=target,
            output_limits=(30, 100),
            sample_time=sample_time
        )
        self.controller.setpoint = target
        
    def GB_PID(self, T_real, target):
        """實現保證有界PID控制策略。
        
        根據實際溫度和目標溫度之間的誤差，決定使用哪個溫度值作為控制輸入。
        
        Args:
            T_real (float): 實際測量的溫度值。
            target (float): 目標溫度值。
            
        Returns:
            float: 控制輸入溫度值，當誤差在範圍內返回目標溫度，否則返回實際溫度。
        """
        delta = abs(T_real - target)
        if delta <= self.GB:
            return T_real  # 當誤差在範圍內，返回目標溫度
        else:
            return T_real  # 當誤差超出範圍，返回實際溫度
            
    def update_target(self, target):
        """更新控制器的目標溫度。
        
        Args:
            target (float): 新的目標溫度值。
        """
        self.controller.setpoint = target

if __name__ == '__main__':
        # 初始化控制器
    adam_port = '/dev/ttyUSB0'
    fan1_port = '/dev/ttyAMA4'
    fan2_port = '/dev/ttyAMA5'
    pump_port = '/dev/ttyAMA3'
    #資料儲存位置(不要動)
    exp_name = '/home/inventec/Desktop/2KWCDU_修改版本/data_manage/PID_fan'
    #實驗檔案名稱(可自行更動)
    exp_var = '250602PID-fan'
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
        target = 28
        sample_time = 1  # 定義取樣時間
        Guaranteed_Bounded_PID_range = 0.5
        Controller = GB_PID_fan( target, Guaranteed_Bounded_PID_range, sample_time)
        adam.start_adam()
        while flag:
            Temperatures = adam.buffer.tolist()
            if any(Temperatures):
                # 獲取溫度數據
                T_GPU = Temperatures[0]  # 定義 T_GPU 變量
                T_CDU_out = Temperatures[3]
                T_env = Temperatures[4]
                


                # 使用 GB_PID 計算控制輸出
                control_temp = Controller.GB_PID(T_CDU_out, target)
                fan_duty = round(Controller.controller(control_temp)/10)*10
                print(f"T_GPU: {T_GPU} | T_CDU_out: {T_CDU_out} | T_env: {T_env}")
                print(f"counter: {counter} | fan speed: {fan_duty}")
                print("----------------------------------------")
                # 更新泵的轉速
                fan1.set_all_duty_cycle(fan_duty)
                fan2.set_all_duty_cycle(fan_duty)
                adam.update_duty_cycles(fan_duty=fan_duty)

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
        print("實驗結束，所有裝置恢復到安全狀態。")
        # 繪製實驗結果圖
        print("繪製實驗結果圖...")
        adam.plot_experiment_results()
        print("實驗結果圖繪製完成。")
