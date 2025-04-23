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
import termios
import tty
import select

# 新增: 儲存優化結果的佇列
optimization_history = deque(maxlen=5)  # 儲存最近5次的優化結果

adam_port = '/dev/ttyUSB0'
fan1_port = '/dev/ttyAMA4'
fan2_port = '/dev/ttyAMA5'
pump_port = '/dev/ttyAMA3'

# 設定MinMaxScaler的路徑，此scaler用於將輸入數據歸一化到[0,1]區間
# 該scaler是在訓練模型時保存的，確保預測時使用相同的數據縮放方式
scaler_path = '/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400/1.5_1KWscalers.jlib' 
model_path = '/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400/2KWCDU_Transformer_model.pth'
model_name = 'no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400'
#設置實驗資料放置的資料夾
exp_name = f'/home/inventec/Desktop/2KWCDU_修改版本/data_manage/control_data/Fan_MPC_SA_data/{model_name}'
#設置實驗資料檔案名稱
exp_var = 'Fan_MPC_data_test.csv'
#設置實驗資料標題
custom_headers = ['time', 'T_GPU', 'T_heater', 'T_CDU_in', 'T_CDU_out', 'T_env', 'T_air_in', 'T_air_out', 'TMP8', 'fan_duty', 'pump_duty']

adam = ADAMScontroller.DataAcquisition(exp_name=exp_name, exp_var=exp_var, port=adam_port, csv_headers=custom_headers)
fan1 = multi_ctrl.multichannel_PWMController(fan1_port)
fan2 = multi_ctrl.multichannel_PWMController(fan2_port)
pump = ctrl.XYKPWMController(pump_port)


# 設置初始轉速
pump_duty=60
pump.set_duty_cycle(pump_duty)
fan_duty=60
fan1.set_all_duty_cycle(fan_duty)
fan2.set_all_duty_cycle(fan_duty)

# 設置ADAM控制器
adam.start_adam()
adam.update_duty_cycles(fan_duty, pump_duty)

# 設置初始目標溫度和最大功率
P_max = 100
GPU_target = 71
target_temp = 30

# 使用帶有溫度預測平滑處理功能的SA優化器
print("\n" + "="*50)
print("📊 系統初始化中")
print("⚡ 初始化SA優化器 | 冷卻水目標溫度: {}°C | 最大功率: {}%".format(target_temp, P_max))
print("="*50)

# 創建SA優化器
sa_optimizer = SA_Optimizer.SA_Optimizer(
    adam=adam, 
    window_size=35, 
    P_max=P_max, 
    target_temp=target_temp,
    model_path=model_path,
    scaler_path=scaler_path
)

# 設置風扇控制頻率
control_frequency = 4  # 控制頻率 (s)

# 設置泵PID控制器
counter = 0
sample_time = 1  # 定義取樣時間
Guaranteed_Bounded_PID_range = 0.5
Controller = Pump_pid.GB_PID_pump(target=GPU_target, Guaranteed_Bounded_PID_range=Guaranteed_Bounded_PID_range, sample_time=sample_time)

# 儲存歷史數據以顯示趨勢
prev_temp_gpu = None
prev_temp_cdu = None
prev_fan_duty = None
prev_pump_duty = None

# ANSI 顏色代碼
class Colors:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"

# 趨勢顯示符號
UP_ARROW = "↑"
DOWN_ARROW = "↓"
STABLE = "="

# 檢查溫度數值是否有效
def validate_temp(temp):
    """驗證溫度值是否合理"""
    try:
        if temp is None or math.isnan(temp) or abs(temp) > 200:
            return "N/A"
        return f"{float(temp):.1f}°C"
    except:
        return "N/A"

# 修改清除終端輸出的函數，保留優化結果區域
def clear_terminal():
    """清除終端輸出，但保留部分區域"""
    # 使用ANSI轉義序列來清除螢幕並將游標移到開頭
    print("\033[2J\033[H", end="")
    
    # 如果有優化歷史，顯示最近的優化結果
    if optimization_history:
        last_opt = optimization_history[-1]
        print(f"{Colors.BOLD}{Colors.CYAN}🔄 上次優化 [{last_opt['time']}]: {Colors.GREEN}風扇速度 {last_opt['fan_speed']}% | 成本 {last_opt['cost']:.2f}{Colors.RESET}")
        print("="*50)

# 獲取趨勢符號
def get_trend(current, previous):
    if previous is None:
        return ""
    if current > previous + 0.1:
        return f"{Colors.RED}{UP_ARROW}{Colors.RESET}"
    elif current < previous - 0.1:
        return f"{Colors.GREEN}{DOWN_ARROW}{Colors.RESET}"
    else:
        return f"{Colors.BLUE}{STABLE}{Colors.RESET}"

# 將文字對齊到特定寬度
def align_text(text, width, align='left'):
    if align == 'left':
        return text.ljust(width)
    elif align == 'right':
        return text.rjust(width)
    elif align == 'center':
        return text.center(width)
    return text

# 顯示溫度狀態
def display_temp_status(T_GPU, T_CDU_out, T_env, T_air_in, T_air_out, gpu_trend, cdu_trend):
    temp_diff = GPU_target - T_GPU
    cdu_diff = target_temp - T_CDU_out
    
    status = "🔥" if temp_diff < 0 else "✓" if abs(temp_diff) < 2 else "❄️"
    
    print("\n" + "="*50)
    print(f"{Colors.BOLD}🌡️ 溫度監控 | {time.strftime('%H:%M:%S')}{Colors.RESET}")
    print("-"*50)
    
    # 固定寬度顯示，確保排列整齊
    w1, w2, w3, w4 = 12, 10, 6, 22  # 各欄位寬度
    
    # GPU溫度
    temp_val = validate_temp(T_GPU)
    target_val = f"目標: {GPU_target}°C (差: {abs(temp_diff):.1f}°C) {status}"
    print(f"{align_text('GPU溫度:', w1)} {align_text(temp_val, w2)} {align_text(gpu_trend, w3)} | {target_val}")
    
    # 冷卻水出口溫度
    temp_val = validate_temp(T_CDU_out)
    target_val = f"目標: {target_temp}°C (差: {abs(cdu_diff):.1f}°C)"
    print(f"{align_text('冷卻水出口:', w1)} {align_text(temp_val, w2)} {align_text(cdu_trend, w3)} | {target_val}")
    
    # 環境溫度
    print(f"{align_text('環境溫度:', w1)} {align_text(validate_temp(T_env), w2)}")
    
    # 空氣入出口溫度
    air_temps = f"{validate_temp(T_air_in)} / {validate_temp(T_air_out)}"
    print(f"{align_text('空氣入/出口:', w1)} {align_text(air_temps, w2+w3+3)} | 差: {abs(T_air_out-T_air_in):.1f}°C")

# 顯示控制狀態
def display_control_status(pump_duty, fan_duty, new_pump_duty, pump_trend, fan_trend):
    print("\n" + "="*50)
    print(f"{Colors.BOLD}⚙️ 控制狀態 | 週期: {counter}{Colors.RESET}")
    print("-"*50)
    
    # 固定寬度顯示
    w1, w2, w3, w4 = 12, 6, 6, 26
    
    # 泵轉速
    duty_val = f"{pump_duty}%"
    new_val = f"{new_pump_duty}% ({'+' if new_pump_duty > pump_duty else '-' if new_pump_duty < pump_duty else '='}{abs(new_pump_duty - pump_duty)}%)"
    print(f"{align_text('泵轉速:', w1)} {align_text(duty_val, w2)} {align_text(pump_trend, w3)} → {new_val}")
    
    # 風扇轉速
    duty_val = f"{fan_duty}%"
    print(f"{align_text('風扇轉速:', w1)} {align_text(duty_val, w2)} {align_text(fan_trend, w3)} → 等待優化...")

# 顯示控制策略狀態
def display_control_strategy(control_temp):
    print("\n" + "="*50)
    print(f"{Colors.BOLD}🔧 控制策略{Colors.RESET}")
    print("-"*50)
    print(f"💧 泵控制 (PID):")
    print(f"   目標GPU溫度: {GPU_target}°C")
    print(f"   控制溫度: {validate_temp(control_temp)}")

# 顯示風扇優化進度
def display_fan_optimization():
    print("-"*50)
    print(f"{Colors.YELLOW}{Colors.BOLD}🌀 風扇優化 (SA) - ⏳ 執行中...{Colors.RESET}")

# 添加一個用於產生提示音的函數
def alert_sound():
    """產生提示音以引起注意"""
    print('\a', end='', flush=True)  # 使用系統提示音

# 顯示風扇優化結果 (精簡版)
def display_optimization_result(optimal_fan_speed, optimal_cost, fan_duty, optimization_time):
    fan_change = optimal_fan_speed - fan_duty
    
    # 添加到歷史記錄
    timestamp = time.strftime('%H:%M:%S')
    optimization_history.append({
        'time': timestamp,
        'fan_speed': optimal_fan_speed,
        'change': fan_change,
        'cost': optimal_cost,
        'opt_time': optimization_time
    })
    
    # 產生提示音
    alert_sound()
    
    # 添加醒目的分隔線
    print(f"\n{Colors.YELLOW}{'★'*30}{Colors.RESET}")
    
    # 顯示當前結果 (添加醒目顏色和框架)
    print(f"{Colors.BOLD}{Colors.GREEN}✅ 風扇優化完成! {Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}✓ 最佳風扇轉速: {optimal_fan_speed}% ({'+' if fan_change > 0 else '-' if fan_change < 0 else '='}{abs(fan_change)}%){Colors.RESET}")
    print(f"{Colors.BOLD}📊 優化成本: {optimal_cost:.2f} | ⏱️ 優化耗時: {optimization_time:.2f}秒{Colors.RESET}")
    
    # 醒目的結束分隔線
    print(f"{Colors.YELLOW}{'★'*30}{Colors.RESET}\n")

# 新增: 顯示優化歷史記錄的函數
def display_optimization_history():
    if not optimization_history:
        return
    
    print("-"*50)
    print(f"{Colors.BOLD}{Colors.CYAN}📜 優化歷史記錄 (最近{len(optimization_history)}次){Colors.RESET}")
    for i, entry in enumerate(reversed(optimization_history), 1):
        print(f"{i}. [{entry['time']}] 風扇: {entry['fan_speed']}% ({'+' if entry['change'] > 0 else '-' if entry['change'] < 0 else '='}{abs(entry['change'])}%) | 成本: {entry['cost']:.2f}")

# 監視SA優化器的輸出，減少過多日誌
# 為了保持簡潔，我們將替換SA_Optimizer腳本中的print函數
original_print = print

def filter_sa_logs(*args, **kwargs):
    """過濾SA優化器輸出，只保留關鍵信息"""
    message = " ".join(map(str, args))
    
    # 關鍵信息清單 - 只保留這些信息
    key_phrases = [
        "數據蒐集完成",
        "初始解",
        "最終解",
        "最佳化完成",
    ]
    
    # 檢查是否為關鍵信息
    if any(phrase in message for phrase in key_phrases):
        original_print(*args, **kwargs)

try:
    # 設置停止條件
    running = True
    
    while running:
        try:
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
                gpu_trend = get_trend(T_GPU, prev_temp_gpu)
                cdu_trend = get_trend(T_CDU_out, prev_temp_cdu)
                fan_trend = get_trend(fan_duty, prev_fan_duty)
                pump_trend = get_trend(pump_duty, prev_pump_duty)
                
                # 更新歷史數據
                prev_temp_gpu = T_GPU
                prev_temp_cdu = T_CDU_out
                prev_fan_duty = fan_duty
                prev_pump_duty = pump_duty
                
                # 使用 GB_PID 計算控制輸出
                control_temp = Controller.GB_PID(T_GPU, GPU_target)
                new_pump_duty = round(Controller.controller(control_temp) / 10) * 10
                
                # 顯示溫度狀態
                display_temp_status(T_GPU, T_CDU_out, T_env, T_air_in, T_air_out, gpu_trend, cdu_trend)
                
                # 顯示控制狀態
                display_control_status(pump_duty, fan_duty, new_pump_duty, pump_trend, fan_trend)
                
                # 更新泵的轉速
                pump.set_duty_cycle(new_pump_duty)
                adam.update_duty_cycles(pump_duty=new_pump_duty)
                time.sleep(1)

                counter += 1

                # 顯示控制策略
                display_control_strategy(control_temp)
                
                # 使用新的控制頻率來調整SA的優化頻率
                if counter % control_frequency == 0:
                    display_fan_optimization()
                    
                    # 替換SA_Optimizer中的print函數，減少輸出
                    SA_Optimizer.print = filter_sa_logs
                    
                    start_time = time.time()
                    optimal_fan_speed, optimal_cost = sa_optimizer.optimize()
                    optimization_time = time.time() - start_time
                    
                    # 恢復原始print函數
                    SA_Optimizer.print = original_print
                    
                    if optimal_fan_speed is not None:
                        # 先顯示優化結果，再改變設定，使結果更容易被看到
                        display_optimization_result(optimal_fan_speed, optimal_cost, fan_duty, optimization_time)
                        
                        # 然後更新風扇設定
                        fan1.set_all_duty_cycle(int(optimal_fan_speed))
                        fan2.set_all_duty_cycle(int(optimal_fan_speed))
                        adam.update_duty_cycles(fan_duty=int(optimal_fan_speed))
                        
                        # 顯示優化歷史記錄
                        display_optimization_history()
                        
                        # 顯示控制說明
                        print("\n" + "="*50)
                        print(f"{Colors.BOLD}📋 控制選項{Colors.RESET}")
                        print("-"*50)
                        print(f"📌 目標設定: GPU={GPU_target}°C | 冷卻水={target_temp}°C")
                        print(f"📝 按下 Ctrl+C 停止程序")
                        
                        # 添加延遲，讓使用者有更多時間查看顯示資訊
                        # 由於整個循環已經有 sleep(1)，不需要在這裡再添加額外的延遲
                
        except KeyboardInterrupt:
            print("\n\n⚠️ 程序已被手動停止")
            running = False
            break
        
except Exception as e:
    print(f"\n❌ 發生錯誤: {e}")
finally:
    # 清理資源
    adam.stop_adam()
    fan1.set_all_duty_cycle(20)
    fan2.set_all_duty_cycle(20)
    pump.set_duty_cycle(40)
    print(f"\n✅ 程序已結束，資源已釋放")