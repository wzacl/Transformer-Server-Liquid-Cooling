import numpy as np
import time

class Model_tester:
    def __init__(self, fan1, fan2, pump, adam):
        self.fan1 = fan1
        self.fan2 = fan2
        self.pump = pump
        self.adam = adam
        self.test_mode = None  # 1: 只變動風扇, 2: 只變動泵, 3: 隨機變動, 4: 自訂風扇變動, 5: 自訂泵變動, 6: 固定風扇與泵轉速
        self.start_time = None
        self.wait_time = 30  # 初始等待 30 秒
        self.device_type = None  # 記錄目前變動的是風扇還是泵
        self.phase = "wait"  # "wait" = 等待 60 秒, "running" = 變動後開始計時, "end" = 結束
        self.fan_duty_sequence = [30, 60, 50, 60, 70, 60, 100, 60, 50, 60, 70,80,90,100,30,60,90]
        self.pump_duty_sequence = [40, 60, 50, 60, 70, 60, 100, 60, 50, 60, 70, 80, 90, 100, 90, 80, 70, 60, 30, 100, 60, 100]
        self.current_index = 0
        self.run_time = 0  # 預設運行時間
        self.total_run_time = None  # 總運行時長
        self.elapsed_total_time = 0  # 記錄已執行時間
        #self.custom_fan_duty_sequence = [60,70,80,90,100,90,80,70,60,50,40,30,40,50,60]  # 自訂風扇轉速序列
        self.custom_fan_duty_sequence = [30,35,40,45,50,55,60,50,55,40,35,30,45,60,75,85,90,100,95,80,70,60]  # 自訂風扇轉速序列
        #self.custom_pump_duty_sequence = [60,70,80,90,100,90,80,70,60,50,40]  # 自訂泵轉速序列
        self.custom_pump_duty_sequence = [40,55,40,45,50,55,60,50,55,40,35,30,45,60,75,85,90,100,95,80,70,60]  # 自訂泵轉速序列
        self.custom_time_sequence = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]  # 自訂時間序列
        self.fixed_fan_duty = 60  # 固定風扇轉速
        self.fixed_pump_duty = 60  # 固定泵轉速

    def start_test(self, mode, total_run_time=None, custom_fan_duty=None, custom_pump_duty=None, 
                  custom_time_sequence=None, fixed_fan_duty=None, fixed_pump_duty=None, wait_time=None):
        """啟動指定測試模式"""
        self.test_mode = mode
        self.start_time = time.time()
        self.phase = "wait"  # 進入等待階段
        self.elapsed_total_time = 0  # 累計運行時間歸零
        self.run_time = 0  # 確保 `run_time` 不會是 `None`
        self.current_index = 0  # 重置序列索引
        
        # 如果提供了等待時間，則更新等待時間
        if wait_time is not None:
            self.wait_time = wait_time
            print(f"設定初始等待時間為 {self.wait_time} 秒")

        if mode == 1:
            self.run_time = 100  # 變動後運行 100 秒
            self.device_type = "fan"
            print(f"[測試 1] 先維持原風扇轉速 {self.wait_time} 秒，然後變動風扇")

        elif mode == 2:
            self.run_time = 25  # 變動後運行 25 秒
            self.device_type = "pump"
            print(f"[測試 2] 先維持原泵轉速 {self.wait_time} 秒，然後變動泵")

        elif mode == 3:
            # 測試 3: 隨機變動風扇或泵，直到達到總運行時長
            self.total_run_time = total_run_time  # 設定總運行時長
            self.run_time = 6  # 預設最小運行時間，避免 NoneType 問題
            self.device_type = "fan" if np.random.rand() > 0.5 else "pump"
            print(f"[測試 3] 開始隨機變動風扇或泵，總運行時長 {self.total_run_time} 秒")
            
        elif mode == 4:
            # 測試 4: 固定泵轉速，自訂風扇轉速變動
            self.device_type = "fan"
            
            # 如果提供了自訂序列，則使用提供的序列
            if custom_fan_duty is not None:
                self.custom_fan_duty_sequence = custom_fan_duty
            if custom_time_sequence is not None:
                self.custom_time_sequence = custom_time_sequence
            if fixed_pump_duty is not None:
                self.fixed_pump_duty = fixed_pump_duty
            
            # 設定固定泵轉速
            if self.fixed_pump_duty is not None:
                self.pump.set_duty_cycle(self.fixed_pump_duty)
                self.adam.update_duty_cycles(pump_duty=self.fixed_pump_duty)
                print(f"[測試 4] 泵轉速固定為 {self.fixed_pump_duty}%")
            
            self.run_time = self.custom_time_sequence[0] if self.custom_time_sequence else 0
            print(f"[測試 4] 自訂風扇轉速變動序列: {self.custom_fan_duty_sequence}")
            print(f"[測試 4] 自訂變動時間序列: {self.custom_time_sequence}")
            print(f"[測試 4] 先維持原風扇轉速 {self.wait_time} 秒，然後開始變動")
            
        elif mode == 5:
            # 測試 5: 固定風扇轉速，自訂泵轉速變動
            self.device_type = "pump"
            
            # 如果提供了自訂序列，則使用提供的序列
            if custom_pump_duty is not None:
                self.custom_pump_duty_sequence = custom_pump_duty
            if custom_time_sequence is not None:
                self.custom_time_sequence = custom_time_sequence
            if fixed_fan_duty is not None:
                self.fixed_fan_duty = fixed_fan_duty
            
            # 設定固定風扇轉速
            if self.fixed_fan_duty is not None:
                self.fan1.set_all_duty_cycle(self.fixed_fan_duty)
                self.fan2.set_all_duty_cycle(self.fixed_fan_duty)
                self.adam.update_duty_cycles(fan_duty=self.fixed_fan_duty)
                print(f"[測試 5] 風扇轉速固定為 {self.fixed_fan_duty}%")
            
            self.run_time = self.custom_time_sequence[0] if self.custom_time_sequence else 0
            print(f"[測試 5] 自訂泵轉速變動序列: {self.custom_pump_duty_sequence}")
            print(f"[測試 5] 自訂變動時間序列: {self.custom_time_sequence}")
            print(f"[測試 5] 先維持原泵轉速 {self.wait_time} 秒，然後開始變動")
            
        elif mode == 6:
            # 測試 6: 固定風扇與泵轉速
            self.device_type = "both"
            
            # 如果提供了固定轉速，則使用提供的值
            if fixed_fan_duty is not None:
                self.fixed_fan_duty = fixed_fan_duty
            if fixed_pump_duty is not None:
                self.fixed_pump_duty = fixed_pump_duty
            if total_run_time is not None:
                self.total_run_time = total_run_time
            else:
                self.total_run_time = 300  # 預設運行5分鐘
            
            # 設定固定風扇轉速
            self.fan1.set_all_duty_cycle(self.fixed_fan_duty)
            self.fan2.set_all_duty_cycle(self.fixed_fan_duty)
            self.adam.update_duty_cycles(fan_duty=self.fixed_fan_duty)
            
            # 設定固定泵轉速
            self.pump.set_duty_cycle(self.fixed_pump_duty)
            self.adam.update_duty_cycles(pump_duty=self.fixed_pump_duty)
            
            print(f"[測試 6] 風扇轉速固定為 {self.fixed_fan_duty}%")
            print(f"[測試 6] 泵轉速固定為 {self.fixed_pump_duty}%")
            print(f"[測試 6] 先維持 {self.wait_time} 秒，然後固定運行 {self.total_run_time} 秒")

    def update_test(self):
        """檢查測試是否結束，並執行測試邏輯"""
        if self.test_mode is None:
            return

        elapsed_time = time.time() - self.start_time

        # 先等待指定時間，然後開始變動設備轉速
        if self.phase == "wait" and elapsed_time >= self.wait_time:
            self.start_time = time.time()  # 重新計時
            self.phase = "running"
            self.current_index = 0  # 重置序列索引

        elif self.phase == "running":
            if self.test_mode == 3:
                # 測試 3: 隨機變動風扇或泵，直到達到總運行時長
                if self.elapsed_total_time < self.total_run_time:
                    if elapsed_time >= self.run_time:  # 確保上一個變動已完成
                        self.device_type = "fan" if np.random.rand() > 0.5 else "pump"

                        if self.device_type == "fan":
                            self.run_time = np.random.randint(6, 181)  # 風扇運行 6~180 秒
                            new_fan_duty = int(np.random.choice(np.arange(30, 101, 5)))  # 風扇轉速 30~100%
                            self.fan1.set_all_duty_cycle(new_fan_duty)
                            self.fan2.set_all_duty_cycle(new_fan_duty)
                            self.adam.update_duty_cycles(fan_duty=new_fan_duty)
                            print(f"[測試 3] 隨機變動風扇轉速至 {new_fan_duty}%，運行 {self.run_time} 秒")

                        else:
                            self.run_time = np.random.randint(6, 41)  # 泵運行 6~40 秒
                            new_pump_duty = int(np.random.choice(np.arange(40, 101, 5)))  # 泵轉速 40~100%
                            self.pump.set_duty_cycle(new_pump_duty)
                            self.adam.update_duty_cycles(pump_duty=new_pump_duty)
                            print(f"[測試 3] 隨機變動泵轉速至 {new_pump_duty}%，運行 {self.run_time} 秒")

                        self.elapsed_total_time += self.run_time  # 更新總運行時間
                        self.start_time = time.time()  # 重新計時
                else:
                    # 測試 3 結束
                    print(f"[測試 3] 總運行時長達到 {self.total_run_time} 秒，測試結束")
                    self.test_mode = None
                    self.device_type = None
                    self.phase = "end"
                    
            elif self.test_mode == 4:
                # 測試 4: 固定泵轉速，自訂風扇轉速變動
                if elapsed_time >= self.run_time and self.current_index < len(self.custom_fan_duty_sequence):
                    new_fan_duty = self.custom_fan_duty_sequence[self.current_index]
                    current_time = self.custom_time_sequence[self.current_index]
                    
                    self.fan1.set_all_duty_cycle(new_fan_duty)
                    self.fan2.set_all_duty_cycle(new_fan_duty)
                    self.adam.update_duty_cycles(fan_duty=new_fan_duty)
                    print(f"[測試 4] 風扇轉速變動至 {new_fan_duty}%，運行 {current_time} 秒")
                    
                    self.run_time = current_time
                    self.current_index += 1
                    self.start_time = time.time()  # 重新計時
                    
                elif self.current_index >= len(self.custom_fan_duty_sequence):
                    # 自訂序列完成後，測試結束
                    print(f"[測試 4] 自訂風扇變動序列測試結束，回到正常運行")
                    self.test_mode = None
                    self.device_type = None
                    self.phase = "end"
                    
            elif self.test_mode == 5:
                # 測試 5: 固定風扇轉速，自訂泵轉速變動
                if elapsed_time >= self.run_time and self.current_index < len(self.custom_pump_duty_sequence):
                    new_pump_duty = self.custom_pump_duty_sequence[self.current_index]
                    current_time = self.custom_time_sequence[self.current_index]
                    
                    self.pump.set_duty_cycle(new_pump_duty)
                    self.adam.update_duty_cycles(pump_duty=new_pump_duty)
                    print(f"[測試 5] 泵轉速變動至 {new_pump_duty}%，運行 {current_time} 秒")
                    
                    self.run_time = current_time
                    self.current_index += 1
                    self.start_time = time.time()  # 重新計時
                    
                elif self.current_index >= len(self.custom_pump_duty_sequence):
                    # 自訂序列完成後，測試結束
                    print(f"[測試 5] 自訂泵變動序列測試結束，回到正常運行")
                    self.test_mode = None
                    self.device_type = None
                    self.phase = "end"
                    
            elif self.test_mode == 6:
                # 測試 6: 固定風扇與泵轉速
                if elapsed_time >= self.total_run_time:
                    # 運行時間結束，測試結束
                    print(f"[測試 6] 固定風扇轉速 {self.fixed_fan_duty}% 和泵轉速 {self.fixed_pump_duty}% 運行 {self.total_run_time} 秒完成，測試結束")
                    self.test_mode = None
                    self.device_type = None
                    self.phase = "end"

            elif elapsed_time >= self.run_time:
                # 測試 1 和 2 的序列模式
                if self.test_mode == 1 and self.current_index < len(self.fan_duty_sequence):
                    new_fan_duty = self.fan_duty_sequence[self.current_index]
                    self.fan1.set_all_duty_cycle(new_fan_duty)
                    self.fan2.set_all_duty_cycle(new_fan_duty)
                    self.adam.update_duty_cycles(fan_duty=new_fan_duty)
                    print(f"[測試 1] 風扇轉速變動至 {new_fan_duty}%，開始運行 180 秒")
                    self.current_index += 1
                    self.start_time = time.time()  # 重新計時

                elif self.test_mode == 2 and self.current_index < len(self.pump_duty_sequence):
                    new_pump_duty = self.pump_duty_sequence[self.current_index]
                    self.pump.set_duty_cycle(new_pump_duty)
                    self.adam.update_duty_cycles(pump_duty=new_pump_duty)
                    print(f"[測試 2] 泵轉速變動至 {new_pump_duty}%，開始運行 50 秒")
                    self.current_index += 1
                    self.start_time = time.time()  # 重新計時

                else:
                    # 所有 duty sequence 完成後，測試結束
                    print(f"[測試 {self.test_mode}] 測試結束，回到正常運行")
                    self.test_mode = None
                    self.device_type = None
                    self.phase = "end"
