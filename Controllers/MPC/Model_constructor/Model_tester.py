import numpy as np
import time

class Model_tester:
    def __init__(self, fan1, fan2, pump, adam):
        self.fan1 = fan1
        self.fan2 = fan2
        self.pump = pump
        self.adam = adam
        self.test_mode = None  # 1: 只變動風扇, 2: 只變動泵, 3: 隨機變動
        self.start_time = None
        self.wait_time = 60  # 初始等待 60 秒
        self.run_time = None
        self.device_type = None  # 記錄目前變動的是風扇還是泵
        self.has_changed = False
        self.phase = "wait"  # "wait" = 等待 60 秒, "running" = 變動後開始計時, "end" = 結束
        self.fan_duty_sequence = [30, 60, 50, 60, 70, 60, 100, 60]
        self.pump_duty_sequence = [40, 60, 50, 60, 70, 60, 100, 60]
        self.current_index = 0

    def start_test(self, mode, run_time=None):
        """啟動指定測試模式"""
        self.test_mode = mode
        self.start_time = time.time()
        self.phase = "wait"  # 進入等待階段

        if mode == 1:
            self.run_time = 180  # 變動後運行 180 秒
            self.device_type = "fan"
            print(f"[測試 1] 先維持原風扇轉速 60 秒，然後變動風扇")

        elif mode == 2:
            self.run_time = 50  # 變動後運行 50 秒
            self.device_type = "pump"
            print(f"[測試 2] 先維持原泵轉速 60 秒，然後變動泵")

        elif mode == 3:
            # 測試 3: 隨機變動風扇或泵（允許手動設定運行時間）
            self.run_time = run_time if run_time else (np.random.randint(7, 200) if np.random.rand() > 0.5 else np.random.randint(3, 45))
            self.device_type = "fan" if np.random.rand() > 0.5 else "pump"

            if self.device_type == "fan":
                new_fan_duty = int(np.random.choice(np.arange(30, 100, 10)))
                self.fan1.set_all_duty_cycle(new_fan_duty)
                self.fan2.set_all_duty_cycle(new_fan_duty)
                self.adam.update_duty_cycles(fan_duty=new_fan_duty)
                print(f"[測試 3] 隨機變動風扇轉速至 {new_fan_duty}%，運行 {self.run_time} 秒")

            else:
                new_pump_duty = int(np.random.choice(np.arange(40, 100, 10)))
                self.pump.set_duty_cycle(new_pump_duty)
                self.adam.update_duty_cycles(pump_duty=new_pump_duty)
                print(f"[測試 3] 隨機變動泵轉速至 {new_pump_duty}%，運行 {self.run_time} 秒")

    def update_test(self):
        """檢查測試是否結束，並執行測試邏輯"""
        if self.test_mode is None:
            return

        elapsed_time = time.time() - self.start_time

        # 先等待 60 秒，然後開始變動設備轉速
        if self.phase == "wait" and elapsed_time >= self.wait_time:
            if self.test_mode == 1:
                new_fan_duty = self.fan_duty_sequence[self.current_index]
                self.fan1.set_all_duty_cycle(new_fan_duty)
                self.fan2.set_all_duty_cycle(new_fan_duty)
                self.adam.update_duty_cycles(fan_duty=new_fan_duty)
                print(f"[測試 1] 風扇轉速變動至 {new_fan_duty}%，開始運行 180 秒")
                self.current_index = (self.current_index + 1) % len(self.fan_duty_sequence)

            elif self.test_mode == 2:
                new_pump_duty = self.pump_duty_sequence[self.current_index]
                self.pump.set_duty_cycle(new_pump_duty)
                self.adam.update_duty_cycles(pump_duty=new_pump_duty)
                print(f"[測試 2] 泵轉速變動至 {new_pump_duty}%，開始運行 50 秒")
                self.current_index = (self.current_index + 1) % len(self.pump_duty_sequence)

            # 進入正式運行階段
            self.start_time = time.time()  # 重新計時
            self.phase = "running"

        elif self.phase == "running" and elapsed_time >= self.run_time:
            if self.test_mode == 3:
                # 測試 3: 持續隨機變動風扇或泵，允許自訂運行時間
                self.start_test(3)
            else:
                print(f"[測試 {self.test_mode}] 測試結束，回到正常運行")
                self.test_mode = None
                self.device_type = None
                self.phase = "end"  # ✅ 設定為 "end"，表示實驗結束
