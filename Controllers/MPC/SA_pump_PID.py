#usr/bin/env python3
import time
import sys
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Control_Unit')
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Controllers/MPC/Model_constructor')
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Controllers/GB_PID')
import ADAMScontroller
import pwmcontroller as ctrl
import multi_channel_pwmcontroller as multi_ctrl
from simple_pid import PID
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque
import math
import os
import csv
import random
import Optimal_algorithm.SA_Optimizer as SA_Optimizer
import GB_PID_pump as Pump_pid
from tabulate import tabulate
import threading
import select
import termios
import tty

adam_port = '/dev/ttyUSB0'
fan1_port = '/dev/ttyAMA4'
fan2_port = '/dev/ttyAMA5'
pump_port = '/dev/ttyAMA3'

#設置實驗資料放置的資料夾
exp_name = '/home/inventec/Desktop/2KWCDU_修改版本/data_manage/control_data/Fan_MPC_SA_data'
#設置實驗資料檔案名稱
exp_var = 'Fan_MPC_data_GPU1.5KW_1(285V_8A)_SA_test_smooth_6.csv'
#設置實驗資料標題
custom_headers = ['time', 'T_GPU', 'T_heater', 'T_CDU_in', 'T_CDU_out', 'T_env', 'T_air_in', 'T_air_out', 'TMP8', 'fan_duty', 'pump_duty']

adam = ADAMScontroller.DataAcquisition(exp_name=exp_name, exp_var=exp_var, port=adam_port, csv_headers=custom_headers)
fan1 = multi_ctrl.multichannel_PWMController(fan1_port)
fan2 = multi_ctrl.multichannel_PWMController(fan2_port)
pump = ctrl.XYKPWMController(pump_port)

# 設定MinMaxScaler的路徑，此scaler用於將輸入數據歸一化到[0,1]區間
# 該scaler是在訓練模型時保存的，確保預測時使用相同的數據縮放方式
scaler_path = '/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400/1.5_1KWscalers.jlib' 
# 設置初始轉速
pump_duty=60
pump.set_duty_cycle(pump_duty)
fan_duty=60
fan1.set_all_duty_cycle(fan_duty)
fan2.set_all_duty_cycle(fan_duty)

# 設置ADAM控制器
adam.start_adam()
adam.update_duty_cycles(fan_duty, pump_duty)

# 設置初始目標溫度
GPU_target = 71  # GPU目標溫度
target_temp = 30  # 冷卻水目標溫度
P_max = 100  # 最大功率

# 使用帶有溫度預測平滑處理功能的SA優化器
print("\n📊 系統初始化")
print("⚡ 初始化SA優化器 | 冷卻水目標溫度: {}°C | 最大功率: {}%".format(target_temp, P_max))

# 創建一個替代的print函數，用於捕獲SA_Optimizer的輸出
original_print = print
sa_optimization_logs = []

def log_print(*args, **kwargs):
    """捕獲優化過程的輸出以便後續顯示"""
    message = " ".join(map(str, args))
    sa_optimization_logs.append(message)
    # 原始print函數仍然可用，但是被我們控制
    # 只有在詳細模式下才實際輸出
    if show_optimization_details:
        original_print(*args, **kwargs)

# 創建一個修改版的優化器，使用我們的log_print
class ModifiedSA_Optimizer(SA_Optimizer.SA_Optimizer):
    def optimize(self):
        """使用模擬退火算法進行優化的修改版本，捕獲日誌"""
        global sa_optimization_logs
        sa_optimization_logs = []  # 清空之前的日誌
        
        # 替換print函數來捕獲輸出
        temp_print = SA_Optimizer.print
        SA_Optimizer.print = log_print
        
        try:
            result = super().optimize()
            return result
        finally:
            # 恢復原始print函數
            SA_Optimizer.print = temp_print
            
# 創建SA優化器
sa_optimizer = ModifiedSA_Optimizer(
    adam=adam, 
    window_size=35, 
    P_max=P_max, 
    target_temp=target_temp
)

#設置風扇控制頻率
control_frequency = 5  # 控制頻率 (s)

#設置泵PID控制器
counter = 0
sample_time = 1  # 定義取樣時間
Guaranteed_Bounded_PID_range = 0.5
Controller = Pump_pid.GB_PID_pump(target=GPU_target, Guaranteed_Bounded_PID_range=Guaranteed_Bounded_PID_range, sample_time=sample_time)

# 儲存歷史數據以顯示趨勢
prev_temp_gpu = None
prev_temp_cdu = None
prev_fan_duty = None
prev_pump_duty = None

# 顯示設定
show_optimization_details = False
settings_mode = False  # 溫度設定模式
current_setting = 0    # 0: 無, 1: GPU溫度, 2: 冷卻水溫度

# 格式化輸出的顏色代碼
class Colors:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

# 豐富的圖標集
class Icons:
    TEMP = "🌡️"
    PUMP = "💧"
    FAN = "🌀"
    GPU = "🖥️"
    UP = "📈"
    DOWN = "📉"
    STABLE = "📊"
    WARNING = "⚠️"
    ERROR = "❌"
    SUCCESS = "✅"
    INFO = "ℹ️"
    TIME = "⏱️"
    POWER = "⚡"
    SETTINGS = "⚙️"
    BULB = "💡"
    CHART = "📈"
    GRAPH = "📊"
    LIGHTNING = "⚡"
    TARGET = "🎯"
    CHECK = "✓"
    CROSS = "✗"
    SEARCH = "🔍"
    GEAR = "⚙️"
    COOL = "❄️"
    HOT = "🔥"
    ALERT = "🚨"
    CLOUD = "☁️"
    WAIT = "⏳"
    ROCKET = "🚀"
    STAR = "⭐"
    LOCK = "🔒"
    UNLOCK = "🔓"
    CLOCK = "🕒"
    WRENCH = "🔧"
    LINK = "🔗"
    WAVE = "〰️"
    UP_ARROW = "↑"
    DOWN_ARROW = "↓"
    RIGHT_ARROW = "→"
    LEFT_ARROW = "←"
    KEYBOARD = "⌨️"
    
# 趨勢顯示函數
def get_trend_symbol(current, previous):
    if previous is None:
        return ""
    if current > previous:
        return f"{Colors.RED}{Icons.UP}{Colors.RESET}"
    elif current < previous:
        return f"{Colors.GREEN}{Icons.DOWN}{Colors.RESET}"
    else:
        return f"{Colors.BLUE}{Icons.STABLE}{Colors.RESET}"

# 使用tabulate套件顯示表格
def display_table(title, headers, data, tablefmt="grid"):
    """使用tabulate套件顯示美觀的表格"""
    print(f"\n{Colors.BOLD}{title}{Colors.RESET}")
    
    colored_headers = [f"{Colors.CYAN}{h}{Colors.RESET}" for h in headers]
    
    # 處理數據中的顏色代碼
    processed_data = []
    for row in data:
        processed_row = []
        for cell in row:
            processed_row.append(cell)
        processed_data.append(processed_row)
    
    print(tabulate(processed_data, headers=colored_headers, tablefmt=tablefmt))

# 清除終端輸出
def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

# 設置終端為非阻塞模式的工具函數
def set_input_nonblocking():
    fd = sys.stdin.fileno()
    # 保存原始tty設置
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        return fd, old_settings
    except:
        return None, old_settings

# 恢復終端設置
def restore_terminal_settings(fd, old_settings):
    if fd is not None:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# 檢查是否有按鍵輸入
def is_data():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

# 顯示優化日誌的函數
def display_optimization_logs(logs, detailed=False):
    if not logs:
        return
    
    # 提取關鍵信息
    initial_solution = None
    final_solution = None
    iterations = []
    
    for log in logs:
        if "初始解" in log:
            initial_solution = log
        elif "最終解" in log:
            final_solution = log
        elif "嘗試解" in log or "發現更好的解" in log:
            iterations.append(log)
    
    if detailed:
        # 詳細模式：顯示全部日誌
        print(f"\n{Colors.BOLD}{Icons.INFO} SA優化過程詳情{Colors.RESET}")
        print("┌" + "─" * 50 + "┐")
        for log in logs:
            # 對不同類型的日誌使用不同顏色
            if "初始解" in log:
                print(f"│ {Colors.CYAN}{log}{Colors.RESET}")
            elif "嘗試解" in log:
                print(f"│ {Colors.YELLOW}{log}{Colors.RESET}")
            elif "接受" in log:
                print(f"│ {Colors.GREEN}{log}{Colors.RESET}")
            elif "拒絕" in log:
                print(f"│ {Colors.RED}{log}{Colors.RESET}")
            elif "發現更好的解" in log:
                print(f"│ {Colors.MAGENTA}{log}{Colors.RESET}")
            elif "當前溫度" in log:
                print(f"│ {Colors.BLUE}{log}{Colors.RESET}")
            elif "最終解" in log:
                print(f"│ {Colors.GREEN}{log}{Colors.RESET}")
            elif "最佳化完成" in log:
                print(f"│ {Colors.GREEN}{log}{Colors.RESET}")
            else:
                print(f"│ {log}")
        print("└" + "─" * 50 + "┘")
    else:
        # 簡潔模式：只顯示關鍵信息
        if initial_solution:
            print(f"{Colors.CYAN}{Icons.INFO} {initial_solution}{Colors.RESET}")
        
        # 顯示優化過程的統計
        if iterations:
            accepted = sum(1 for log in logs if "接受" in log)
            rejected = sum(1 for log in logs if "拒絕" in log)
            better_solutions = sum(1 for log in logs if "發現更好的解" in log)
            total_iterations = len(iterations)
            
            print(f"{Colors.YELLOW}{Icons.GEAR} 優化統計: {total_iterations}次迭代, {accepted}次接受, {rejected}次拒絕, {better_solutions}次改進{Colors.RESET}")
        
        if final_solution:
            print(f"{Colors.GREEN}{Icons.SUCCESS} {final_solution}{Colors.RESET}")

# 顯示溫度設置界面
def display_settings_mode():
    global current_setting, GPU_target, target_temp
    
    clear_terminal()
    print(f"\n{Colors.BG_YELLOW}{Colors.BLACK}{Colors.BOLD} 溫度設定模式 {Colors.RESET}\n")
    
    # GPU溫度設置
    gpu_color = Colors.MAGENTA if current_setting == 1 else ""
    gpu_indicator = f"{Colors.MAGENTA}{Icons.RIGHT_ARROW}{Colors.RESET} " if current_setting == 1 else "  "
    print(f"{gpu_indicator}{gpu_color}{Icons.GPU} GPU目標溫度: {GPU_target}°C {Colors.RESET}")
    
    # 冷卻水溫度設置
    water_color = Colors.CYAN if current_setting == 2 else ""
    water_indicator = f"{Colors.CYAN}{Icons.RIGHT_ARROW}{Colors.RESET} " if current_setting == 2 else "  "
    print(f"{water_indicator}{water_color}{Icons.COOL} 冷卻水目標溫度: {target_temp}°C {Colors.RESET}")
    
    print("\n操作說明:")
    print(f"{Icons.UP_ARROW}/{Icons.DOWN_ARROW}: 調整數值 | {Icons.LEFT_ARROW}/{Icons.RIGHT_ARROW}: 切換選項 | Enter: 確認 | Esc: 取消")
    
    return True

# 更新SA優化器目標溫度
def update_optimizer_target_temp():
    global sa_optimizer, target_temp
    sa_optimizer.target_temp = target_temp
    print(f"{Icons.INFO} SA優化器目標溫度已更新為: {target_temp}°C")

# 更新PID控制器目標溫度
def update_controller_target():
    global Controller, GPU_target
    Controller.target = GPU_target
    print(f"{Icons.INFO} PID控制器目標溫度已更新為: {GPU_target}°C")

try:
    # 設置停止條件
    running = True
    
    # 保存原始終端設置
    fd, old_settings = set_input_nonblocking()
    
    while running:
        try:
            # 檢查是否處於設置模式
            if settings_mode:
                display_settings_mode()
                
                # 檢查按鍵輸入
                if is_data():
                    key = sys.stdin.read(1)
                    
                    # 處理退出和確認
                    if ord(key) == 27:  # ESC鍵
                        settings_mode = False
                    elif ord(key) == 13:  # Enter鍵
                        settings_mode = False
                        # 更新SA優化器和PID控制器的目標溫度
                        update_optimizer_target_temp()
                        update_controller_target()
                    
                    # 上下箭頭調整溫度
                    elif ord(key) == 65:  # 上箭頭
                        if current_setting == 1:  # GPU溫度
                            GPU_target = min(85, GPU_target + 1)
                        elif current_setting == 2:  # 冷卻水溫度
                            target_temp = min(40, target_temp + 1)
                    elif ord(key) == 66:  # 下箭頭
                        if current_setting == 1:  # GPU溫度
                            GPU_target = max(60, GPU_target - 1)
                        elif current_setting == 2:  # 冷卻水溫度
                            target_temp = max(20, target_temp - 1)
                    
                    # 左右箭頭切換選項
                    elif ord(key) == 68:  # 左箭頭
                        current_setting = max(1, current_setting - 1)
                    elif ord(key) == 67:  # 右箭頭
                        current_setting = min(2, current_setting + 1)
                
                time.sleep(0.1)  # 降低CPU使用率
                continue
            
            # 正常模式
            Temperatures = adam.buffer.tolist()
            if any(Temperatures):
                clear_terminal()
                
                # 獲取溫度數據
                T_GPU = Temperatures[0]  # 定義 T_GPU 變量
                T_CDU_out = Temperatures[3]
                T_env = Temperatures[4]
                T_air_in = Temperatures[5]
                T_air_out = Temperatures[6]
                fan_duty = Temperatures[8]
                pump_duty = Temperatures[9]
                
                # 計算趨勢
                gpu_trend = get_trend_symbol(T_GPU, prev_temp_gpu)
                cdu_trend = get_trend_symbol(T_CDU_out, prev_temp_cdu)
                fan_trend = get_trend_symbol(fan_duty, prev_fan_duty)
                pump_trend = get_trend_symbol(pump_duty, prev_pump_duty)
                
                # 更新歷史數據
                prev_temp_gpu = T_GPU
                prev_temp_cdu = T_CDU_out
                prev_fan_duty = fan_duty
                prev_pump_duty = pump_duty
                
                # 使用 GB_PID 計算控制輸出
                control_temp = Controller.GB_PID(T_GPU, GPU_target)
                new_pump_duty = round(Controller.controller(control_temp) / 5) * 5
                
                # 計算GPU溫差
                temp_diff = GPU_target - T_GPU
                temp_status = f"{Colors.RED}{Icons.HOT}{Colors.RESET}" if temp_diff < 0 else f"{Colors.GREEN}{Icons.CHECK}{Colors.RESET}" if abs(temp_diff) < 2 else f"{Colors.BLUE}{Icons.COOL}{Colors.RESET}"
                
                # 計算運行時間
                current_time = time.strftime('%H:%M:%S')
                
                # 準備溫度表格數據
                temp_table_data = [
                    [f"{Icons.GPU} GPU", f"{T_GPU:.1f}°C", gpu_trend, f"目標: {GPU_target}°C (差: {abs(temp_diff):.1f}°C) {temp_status}"],
                    [f"{Icons.COOL} CDU出口", f"{T_CDU_out:.1f}°C", cdu_trend, f"目標: {target_temp}°C (差: {abs(T_CDU_out-target_temp):.1f}°C)"],
                    [f"{Icons.CLOUD} 環境", f"{T_env:.1f}°C", "", ""],
                    [f"{Icons.WAVE} 空氣入/出", f"{T_air_in:.1f}°C / {T_air_out:.1f}°C", "", f"差: {abs(T_air_out-T_air_in):.1f}°C"]
                ]
                
                # 準備控制表格數據
                control_table_data = [
                    [f"{Icons.PUMP} 泵轉速", f"{pump_duty}%", pump_trend, f"{new_pump_duty}% ({'+' if new_pump_duty > pump_duty else '-' if new_pump_duty < pump_duty else '='}{abs(new_pump_duty - pump_duty)}%)"],
                    [f"{Icons.FAN} 風扇轉速", f"{fan_duty}%", fan_trend, "等待優化..."]
                ]
                
                # 顯示系統標題
                print(f"\n{Colors.BG_BLUE}{Colors.WHITE}{Colors.BOLD} 伺服器液冷系統監控 {Colors.RESET} {Icons.CLOCK} {current_time} {Icons.SETTINGS} 週期: {counter}")
                
                # 使用tabulate顯示溫度表格
                display_table(f"{Icons.TEMP} 溫度監控", ["感測器", "溫度", "趨勢", "狀態"], temp_table_data)
                
                # 使用tabulate顯示控制表格
                display_table(f"{Icons.GEAR} 控制狀態", ["控制設備", "當前值", "趨勢", "新設定值"], control_table_data)
                
                # 更新泵的轉速
                pump.set_duty_cycle(new_pump_duty)
                adam.update_duty_cycles(pump_duty=new_pump_duty)
                time.sleep(1)

                counter += 1

                # 顯示控制狀態
                print(f"\n{Colors.BOLD}{Icons.WRENCH} 控制策略{Colors.RESET}")
                print("┌" + "─" * 50 + "┐")
                print(f"│ {Colors.MAGENTA}{Icons.PUMP} 泵控制 (PID){Colors.RESET}")
                print(f"│ {Icons.TARGET} 目標GPU溫度: {GPU_target}°C")
                print(f"│ {Icons.CHART} 控制溫度: {control_temp:.1f}°C")
                
                # 使用新的控制頻率來調整SA的優化頻率
                if counter % control_frequency == 0:
                    print(f"├" + "─" * 50 + "┤")
                    print(f"│ {Colors.GREEN}{Icons.FAN} 風扇優化 (SA){Colors.RESET} - {Icons.WAIT} 執行中...")
                    
                    start_time = time.time()
                    optimal_fan_speed, optimal_cost = sa_optimizer.optimize()
                    optimization_time = time.time() - start_time
                    
                    if optimal_fan_speed is not None:
                        fan1.set_all_duty_cycle(int(optimal_fan_speed))
                        fan2.set_all_duty_cycle(int(optimal_fan_speed))
                        adam.update_duty_cycles(fan_duty=int(optimal_fan_speed))
                        fan_change = optimal_fan_speed - fan_duty
                        
                        # 顯示優化結果
                        print(f"│ {Icons.SUCCESS} 最佳風扇轉速: {optimal_fan_speed}% ({'+' if fan_change > 0 else '-' if fan_change < 0 else '='}{abs(fan_change)}%)")
                        print(f"│ {Icons.CHART} 優化成本: {optimal_cost:.2f}")
                        print(f"│ {Icons.TIME} 優化耗時: {optimization_time:.2f}秒")
                        
                        # 顯示簡潔版的優化日誌
                        display_optimization_logs(sa_optimization_logs, detailed=show_optimization_details)
                    else:
                        print(f"│ {Colors.YELLOW}{Icons.WAIT} 數據蒐集中，等待數據蒐集完成...{Colors.RESET}")
                
                print("└" + "─" * 50 + "┘")
                
                # 顯示模型運行狀態
                print(f"\n{Colors.BOLD}{Icons.BULB} 模型資訊{Colors.RESET}")
                print("┌" + "─" * 50 + "┐")
                print(f"│ {Icons.CHART} 預測模型: Transformer | 時間窗口: 35 | 步數: 8")
                print(f"│ {Icons.GEAR} 優化算法: 模擬退火 (SA) | 目標溫度: {target_temp}°C")
                print("└" + "─" * 50 + "┘")
                
                # 使用控制項
                print(f"\n{Colors.BOLD}控制選項:{Colors.RESET}")
                print(f"{Icons.KEYBOARD} 按下 {Colors.BOLD}S{Colors.RESET} 進入溫度設定模式")
                print(f"{Icons.LOCK} 按下 {Colors.BOLD}D{Colors.RESET} 切換優化詳情顯示 ({Colors.GREEN if show_optimization_details else Colors.RED}{'開啟' if show_optimization_details else '關閉'}{Colors.RESET})")
                print(f"{Icons.POWER} 按下 {Colors.BOLD}Ctrl+C{Colors.RESET} 停止程序")
                
                # 處理鍵盤輸入，允許切換顯示模式和進入設置模式
                if is_data():
                    key = sys.stdin.read(1).upper()
                    if key == 'D':
                        show_optimization_details = not show_optimization_details
                        print(f"\n{Icons.INFO} 優化詳情顯示已{Colors.GREEN if show_optimization_details else Colors.RED}{'開啟' if show_optimization_details else '關閉'}{Colors.RESET}")
                        time.sleep(1)
                    elif key == 'S':
                        settings_mode = True
                        current_setting = 1  # 默認選擇GPU溫度
                
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}{Icons.ALERT} 程序已被手動停止{Colors.RESET}")
            running = False
            break
        
except Exception as e:
    print(f"\n{Colors.RED}{Icons.ERROR} 發生錯誤: {e}{Colors.RESET}")
finally:
    # 恢復終端設置
    restore_terminal_settings(fd, old_settings)
    
    # 清理資源
    adam.stop_adam()
    fan1.set_all_duty_cycle(20)
    fan2.set_all_duty_cycle(20)
    pump.set_duty_cycle(40)
    print(f"\n{Colors.GREEN}{Icons.SUCCESS} 程序已結束，資源已釋放{Colors.RESET}")