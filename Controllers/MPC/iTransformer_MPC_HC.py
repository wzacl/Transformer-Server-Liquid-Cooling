#!/usr/bin/env python3
"""冷卻系統控制器模組

此模組實現了基於模型預測控制(MPC)的冷卻系統控制器，用於管理GPU冷卻系統的風扇和泵的速度。
系統使用模擬退火算法優化風扇速度，並使用PID控制器管理泵速。
"""
import sys
import os

# -- 新增的 sys.path 修改開始 --
# 獲取目前檔案 (iTransformer_MPC.py) 的絕對路徑
_current_file_path = os.path.abspath(__file__)
# 導航到專案根目錄 (2KWCDU_修改版本)
# .../code_manage/Controllers/MPC/iTransformer_MPC.py -> .../2KWCDU_修改版本/
_project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_current_file_path))))

# 如果專案根目錄不在 sys.path 中，則將其加入到最前面
if _project_root_dir not in sys.path:
    sys.path.insert(0, _project_root_dir)
# -- 新增的 sys.path 修改結束 --

import time
# sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Control_Unit') # 建議後續移除
# sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Controllers/MPC/Model_constructor') # 建議後續移除
# sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Controllers/GB_PID') # 建議後續移除
from collections import deque
import math
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from code_manage.Control_Unit import ADAMScontroller  # 數據採集控制器 - 修改為絕對導入
from code_manage.Control_Unit import pwmcontroller as ctrl  # PWM控制器 - 修改為絕對導入
from code_manage.Control_Unit import multi_channel_pwmcontroller as multi_ctrl  # 多通道PWM控制器 - 修改為絕對導入
from simple_pid import PID  # PID控制器庫 (假設為已安裝的第三方庫)
from code_manage.Controllers.MPC.Optimal_algorithm import HC_iTransformer as HC_iTransformer  # 爬山優化器 - 修改為絕對導入
from code_manage.Controllers.GB_PID import GB_PID_pump as Pump_pid  # 泵控制PID - 修改為絕對導入

class HardwareConfig:
    """硬體配置類
    
    存儲系統硬體組件的連接端口配置，包括ADAM數據採集模組、風扇和泵的串口設置。
    
    Attributes:
        adam_port: ADAM數據採集模組的串口路徑
        fan1_port: 風扇1的串口路徑
        fan2_port: 風扇2的串口路徑
        pump_port: 泵的串口路徑
    """
    def __init__(self, 
                 adam_port: str = '/dev/ttyUSB0',
                 fan1_port: str = '/dev/ttyAMA4',
                 fan2_port: str = '/dev/ttyAMA5',
                 pump_port: str = '/dev/ttyAMA3'):
        self.adam_port = adam_port
        self.fan1_port = fan1_port
        self.fan2_port = fan2_port
        self.pump_port = pump_port

class ModelConfig:
    """模型配置類
    
    存儲MPC控制器使用的預測模型配置和實驗數據設置。
    
    Attributes:
        scaler_path: 數據標準化器的路徑
        model_path: 預測模型的路徑
        model_name: 預測模型的名稱
        exp_name: 實驗名稱/路徑
        exp_var: 實驗數據文件名
        custom_headers: 數據列標題列表
    """
    def __init__(self, 
                 scaler_path: str,
                 model_path: str,
                 exp_name: str,
                 exp_var: str = 'Fan_MPC_data_test.csv',
                 custom_headers: List[str] = None):
        self.scaler_path = scaler_path
        self.model_path = model_path
        self.exp_name = exp_name
        self.exp_var = exp_var
        self.custom_headers = custom_headers or [
            'time', 'T_GPU', 'T_heater', 'T_CDU_in', 'T_CDU_out', 
            'T_env', 'T_air_in', 'T_air_out', 'TMP8', 'fan_duty', 'pump_duty','target_temperature','gpu_target_temperature'
        ]

class ExperimentMode:
    """實驗模式類
    
    實現目標溫度的自動週期性變化，用於測試控制系統在不同目標溫度下的適應性能。
    
    Attributes:
        enabled: 是否啟用實驗模式
        period: 溫度變化週期（秒）
        gpu_targets: GPU目標溫度列表
        system_targets: 系統目標溫度列表
        start_time: 實驗模式開始時間
        current_index: 當前目標溫度索引
        control_params: 控制參數對象的引用
    """
    def __init__(self, 
                 control_params,
                 period: int = 300,
                 gpu_targets: List[float] = None,
                 system_targets: List[float] = None):
        """初始化實驗模式
        
        Args:
            control_params: 控制參數對象引用
            period: 溫度變化週期（秒）
            gpu_targets: GPU目標溫度列表，默認在70-75°C之間變化
            system_targets: 系統目標溫度列表，默認在28-32°C之間變化
        """
        self.enabled = False
        self.period = period
        self.gpu_targets = gpu_targets or [70, 72, 75, 73]
        self.system_targets = system_targets or [28, 30, 32, 30]
        self.start_time = 0
        self.current_index = 0
        self.control_params = control_params
        
    def start(self):
        """啟動實驗模式"""
        self.enabled = True
        self.start_time = time.time()
        self.current_index = 0
        print(f"\n🧪 實驗模式已啟動！溫度將每 {self.period} 秒變化一次")
        self._update_target_temperatures()
        
    def stop(self):
        """停止實驗模式"""
        self.enabled = False
        print("\n🧪 實驗模式已停止")
        
    def toggle(self):
        """切換實驗模式狀態"""
        if self.enabled:
            self.stop()
        else:
            self.start()
    
    def update(self):
        """更新實驗模式，檢查是否需要變更目標溫度
        
        Returns:
            bool: 如果目標溫度已更新則返回True
        """
        if not self.enabled:
            return False
            
        elapsed = time.time() - self.start_time
        should_be_index = int((elapsed % (self.period * len(self.gpu_targets))) // self.period)
        
        if should_be_index != self.current_index:
            self.current_index = should_be_index
            self._update_target_temperatures()
            return True
            
        return False
        
    def _update_target_temperatures(self):
        """根據當前索引更新目標溫度"""
        gpu_target = self.gpu_targets[self.current_index]
        system_target = self.system_targets[self.current_index]
        
        # 更新控制參數中的目標溫度
        self.control_params.update_targets(gpu_target=gpu_target, target_temp=system_target)
        
        print(f"\n🧪 實驗模式：已切換到目標溫度設定 #{self.current_index+1}")
        print(f"   GPU目標: {gpu_target}°C, 系統目標: {system_target}°C")
    
    def set_parameters(self, period: Optional[int] = None,
                      gpu_targets: Optional[List[float]] = None,
                      system_targets: Optional[List[float]] = None):
        """設置實驗模式參數
        
        Args:
            period: 溫度變化週期（秒）
            gpu_targets: GPU目標溫度列表
            system_targets: 系統目標溫度列表
        """
        restart = self.enabled
        if restart:
            self.stop()
            
        if period is not None:
            self.period = period
        if gpu_targets is not None:
            self.gpu_targets = gpu_targets
        if system_targets is not None:
            self.system_targets = system_targets
            
        # 檢查溫度列表長度是否一致
        if len(self.gpu_targets) != len(self.system_targets):
            raise ValueError("GPU和系統目標溫度列表長度必須相同")
            
        if restart:
            self.start()
            
        print(f"\n🧪 實驗模式參數已更新:")
        print(f"   週期: {self.period}秒")
        print(f"   GPU目標溫度: {self.gpu_targets}")
        print(f"   系統目標溫度: {self.system_targets}")

class ControlParameters:
    """控制參數類
    
    存儲控制系統的運行參數，包括目標溫度、控制頻率和初始設定值。
    支持動態更新目標溫度以適應不同實驗需求。
    
    Attributes:
        p_max: 最大功率限制
        gpu_target: GPU目標溫度(°C)
        target_temp: 系統目標溫度(°C)
        control_frequency: 控制頻率(每N次循環執行一次優化)
        initial_fan_duty: 風扇初始佔空比(%)
        initial_pump_duty: 泵初始佔空比(%)
    """ 
    def __init__(self,
                 p_max: float = 100,
                 gpu_target: float = 71,
                 target_temp: float = 30,
                 control_frequency: int = 10,
                 initial_fan_duty: float = 60,
                 initial_pump_duty: float = 60):
        self.p_max = p_max
        self._gpu_target = gpu_target
        self._target_temp = target_temp
        self.control_frequency = control_frequency
        self.initial_fan_duty = initial_fan_duty
        self.initial_pump_duty = initial_pump_duty
        self._observers = []

    def register_observer(self, observer):
        """註冊一個觀察者以接收溫度目標更改通知。
        
        Args:
            observer: 實現update_target_temp方法的對象
        """
        self._observers.append(observer)

    def _notify_observers(self):
        """通知所有觀察者目標溫度已更改"""
        for observer in self._observers:
            observer.update_target_temp(self._gpu_target, self._target_temp)

    @property
    def gpu_target(self) -> float:
        """獲取GPU目標溫度"""
        return self._gpu_target

    @gpu_target.setter
    def gpu_target(self, value: float):
        """設置新的GPU目標溫度並通知觀察者
        
        Args:
            value: 新的目標溫度值(°C)
        """
        if value != self._gpu_target:
            self._gpu_target = value
            self._notify_observers()

    @property
    def target_temp(self) -> float:
        """獲取系統目標溫度"""
        return self._target_temp

    @target_temp.setter
    def target_temp(self, value: float):
        """設置新的系統目標溫度並通知觀察者
        
        Args:
            value: 新的目標溫度值(°C)
        """
        if value != self._target_temp:
            self._target_temp = value
            self._notify_observers()

    def update_targets(self, gpu_target: Optional[float] = None, 
                      target_temp: Optional[float] = None):
        """同時更新多個目標溫度
        
        Args:
            gpu_target: 新的GPU目標溫度(°C)，如果為None則保持不變
            target_temp: 新的系統目標溫度(°C)，如果為None則保持不變
        """
        changed = False
        if gpu_target is not None and gpu_target != self._gpu_target:
            self._gpu_target = gpu_target
            changed = True
        if target_temp is not None and target_temp != self._target_temp:
            self._target_temp = target_temp
            changed = True
        if changed:
            self._notify_observers()

class DisplayManager:
    """顯示管理類
    
    管理控制系統的終端顯示，提供溫度和控制狀態的可視化。
    
    Attributes:
        UP_ARROW: 上升趨勢符號
        DOWN_ARROW: 下降趨勢符號
        STABLE: 穩定趨勢符號
        optimization_history: 優化歷史記錄
    """
    class Colors:
        """ANSI 顏色代碼
        
        用於終端顯示的顏色代碼常量。
        
        Attributes:
            RESET: 重置所有格式
            RED: 紅色文字
            GREEN: 綠色文字
            YELLOW: 黃色文字
            BLUE: 藍色文字
            CYAN: 青色文字
            WHITE: 白色文字
            BOLD: 粗體文字
        """
        RESET = "\033[0m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        CYAN = "\033[36m"
        WHITE = "\033[37m"
        BOLD = "\033[1m"

    def __init__(self):
        self.UP_ARROW = "↑"
        self.DOWN_ARROW = "↓"
        self.STABLE = "="
        self.optimization_history = deque(maxlen=5)

    def clear_terminal(self):
        """清除終端輸出並顯示最近的優化歷史。
        
        清空終端屏幕並在頂部顯示最近一次優化的結果。
        """
        print("\033[2J\033[H", end="")
        if self.optimization_history:
            last_opt = self.optimization_history[-1]
            print(f"{self.Colors.BOLD}{self.Colors.CYAN}🔄 上次優化 [{last_opt['time']}]: "
                  f"{self.Colors.GREEN}風扇速度 {last_opt['fan_speed']}% | "
                  f"成本 {last_opt['cost']:.2f}{self.Colors.RESET}")
            print("="*50)

    def get_trend(self, current, previous):
        """獲取數值變化趨勢的視覺指示符。
        
        Args:
            current: 當前數值
            previous: 先前數值
            
        Returns:
            帶有顏色格式的趨勢符號字符串
        """
        if previous is None:
            return ""
        if current > previous + 0.1:
            return f"{self.Colors.RED}{self.UP_ARROW}{self.Colors.RESET}"
        elif current < previous - 0.1:
            return f"{self.Colors.GREEN}{self.DOWN_ARROW}{self.Colors.RESET}"
        return f"{self.Colors.BLUE}{self.STABLE}{self.Colors.RESET}"
    
    def validate_temp(self, temp):
        """驗證溫度值是否合理
        
        Args:
            temp: 待驗證的溫度值
            
        Returns:
            格式化的溫度字符串或"N/A"
        """
        try:
            if temp is None or math.isnan(temp) or abs(temp) > 200:
                return "N/A"
            return f"{float(temp):.1f}°C"
        except:
            return "N/A"
            
    def align_text(self, text, width, align='left'):
        """將文字對齊到特定寬度
        
        Args:
            text: 要對齊的文字
            width: 目標寬度
            align: 對齊方式 (left, right, center)
            
        Returns:
            對齊後的文字
        """
        if align == 'left':
            return str(text).ljust(width)
        elif align == 'right':
            return str(text).rjust(width)
        elif align == 'center':
            return str(text).center(width)
        return str(text)

    def display_temp_status(self, temps, trends, targets):
        """顯示溫度狀態信息。
        
        Args:
            temps: 各測量點溫度字典
            trends: 各測量點溫度變化趨勢字典
            targets: 各測量點目標溫度字典
        """
        T_GPU = temps.get('T_GPU')
        T_CDU_out = temps.get('T_CDU_out')
        T_env = temps.get('T_env')
        T_air_in = temps.get('T_air_in')
        T_air_out = temps.get('T_air_out')
        
        gpu_trend = trends.get('gpu', '')
        cdu_trend = trends.get('cdu', '')
        
        gpu_target = targets.get('gpu_target')
        cdu_target = targets.get('cdu_target')
        
        temp_diff = gpu_target - T_GPU if T_GPU is not None and gpu_target is not None else 0
        cdu_diff = cdu_target - T_CDU_out if T_CDU_out is not None and cdu_target is not None else 0
        
        status = "🔥" if temp_diff < 0 else "✓" if abs(temp_diff) < 2 else "❄️"
        
        print("\n" + "="*50)
        print(f"{self.Colors.BOLD}🌡️ 溫度監控 | {time.strftime('%H:%M:%S')}{self.Colors.RESET}")
        print("-"*50)
        
        # 固定寬度顯示，確保排列整齊
        w1, w2, w3, w4 = 12, 10, 6, 22  # 各欄位寬度
        
        # GPU溫度
        temp_val = self.validate_temp(T_GPU)
        target_val = f"目標: {gpu_target}°C (差: {abs(temp_diff):.1f}°C) {status}"
        print(f"{self.align_text('GPU溫度:', w1)} {self.align_text(temp_val, w2)} {self.align_text(gpu_trend, w3)} | {target_val}")
        
        # 冷卻水出口溫度
        temp_val = self.validate_temp(T_CDU_out)
        target_val = f"目標: {cdu_target}°C (差: {abs(cdu_diff):.1f}°C)"
        print(f"{self.align_text('冷卻水出口:', w1)} {self.align_text(temp_val, w2)} {self.align_text(cdu_trend, w3)} | {target_val}")
        
        # 環境溫度
        print(f"{self.align_text('環境溫度:', w1)} {self.align_text(self.validate_temp(T_env), w2)}")
        
        # 空氣入出口溫度
        if T_air_in is not None and T_air_out is not None:
            air_temps = f"{self.validate_temp(T_air_in)} / {self.validate_temp(T_air_out)}"
            air_diff = abs(T_air_out-T_air_in) if T_air_in is not None and T_air_out is not None else 0
            print(f"{self.align_text('空氣入/出口:', w1)} {self.align_text(air_temps, w2+w3+3)} | 差: {air_diff:.1f}°C")

    def display_control_status(self, duties, trends):
        """顯示控制器狀態信息。
        
        Args:
            duties: 各執行器佔空比字典
            trends: 各執行器佔空比變化趨勢字典
        """
        pump_duty = duties.get('pump_duty')
        fan_duty = duties.get('fan_duty')
        new_pump_duty = duties.get('new_pump_duty')
        
        pump_trend = trends.get('pump', '')
        fan_trend = trends.get('fan', '')
        counter = duties.get('counter', 0)
        
        print("\n" + "="*50)
        print(f"{self.Colors.BOLD}⚙️ 控制狀態 | 週期: {counter}{self.Colors.RESET}")
        print("-"*50)
        
        # 固定寬度顯示
        w1, w2, w3, w4 = 12, 6, 6, 26
        
        # 泵轉速
        duty_val = f"{pump_duty}%"
        new_val = f"{new_pump_duty}% ({'+' if new_pump_duty > pump_duty else '-' if new_pump_duty < pump_duty else '='}{abs(new_pump_duty - pump_duty)}%)"
        print(f"{self.align_text('泵轉速:', w1)} {self.align_text(duty_val, w2)} {self.align_text(pump_trend, w3)} → {new_val}")
        
        # 風扇轉速
        duty_val = f"{fan_duty}%"
        print(f"{self.align_text('風扇轉速:', w1)} {self.align_text(duty_val, w2)} {self.align_text(fan_trend, w3)} → 等待優化...")
        
    def display_control_strategy(self, control_temp, gpu_target):
        """顯示控制策略信息。
        
        Args:
            control_temp: 控制溫度
            gpu_target: GPU目標溫度
        """
        print("\n" + "="*50)
        print(f"{self.Colors.BOLD}🔧 控制策略{self.Colors.RESET}")
        print("-"*50)
        print(f"💧 泵控制 (PID):")
        print(f"   目標GPU溫度: {gpu_target}°C")
        print(f"   控制溫度: {self.validate_temp(control_temp)}")
        
    def display_fan_optimization(self):
        """顯示風扇優化進度。
        
        顯示風扇優化計算過程的狀態信息。
        """
        print("-"*50)
        print(f"{self.Colors.YELLOW}{self.Colors.BOLD}🌀 風扇優化 (SA) - ⏳ 執行中...{self.Colors.RESET}")
        
    def display_optimization_result(self, optimal_fan_speed, optimal_cost, 
                                  fan_duty, optimization_time):
        """顯示風扇優化結果。
        
        Args:
            optimal_fan_speed: 優化後的風扇速度(%)
            optimal_cost: 優化算法計算的成本
            fan_duty: 優化前的風扇速度(%)
            optimization_time: 優化計算耗時(秒)
        """
        fan_change = optimal_fan_speed - fan_duty
        
        # 添加到歷史記錄
        timestamp = time.strftime('%H:%M:%S')
        self.optimization_history.append({
            'time': timestamp,
            'fan_speed': optimal_fan_speed,
            'change': fan_change,
            'cost': optimal_cost,
            'opt_time': optimization_time
        })
        
        # 產生提示音
        print('\a', end='', flush=True)  # 使用系統提示音
        
        # 添加醒目的分隔線
        print(f"\n{self.Colors.YELLOW}{'★'*30}{self.Colors.RESET}")
        
        # 顯示當前結果
        print(f"{self.Colors.BOLD}{self.Colors.GREEN}✅ 風扇優化完成! {self.Colors.RESET}")
        print(f"{self.Colors.BOLD}{self.Colors.GREEN}✓ 最佳風扇轉速: {optimal_fan_speed}% ({'+' if fan_change > 0 else '-' if fan_change < 0 else '='}{abs(fan_change)}%){self.Colors.RESET}")
        print(f"{self.Colors.BOLD}📊 優化成本: {optimal_cost:.2f} | ⏱️ 優化耗時: {optimization_time:.2f}秒{self.Colors.RESET}")
        
        # 醒目的結束分隔線
        print(f"{self.Colors.YELLOW}{'★'*30}{self.Colors.RESET}\n")
        
        # 添加短暫延遲，確保使用者能看到結果
        time.sleep(1)
        
    def display_optimization_history(self):
        """顯示優化歷史記錄。
        
        顯示最近幾次的風扇優化結果記錄。
        """
        if not self.optimization_history:
            return
        
        print("-"*50)
        print(f"{self.Colors.BOLD}{self.Colors.CYAN}📜 優化歷史記錄 (最近{len(self.optimization_history)}次){self.Colors.RESET}")
        for i, entry in enumerate(reversed(self.optimization_history), 1):
            print(f"{i}. [{entry['time']}] 風扇: {entry['fan_speed']}% ({'+' if entry['change'] > 0 else '-' if entry['change'] < 0 else '='}{abs(entry['change'])}%) | 成本: {entry['cost']:.2f}")
    
    def display_control_options(self, gpu_target, target_temp, experiment_mode=False):
        """顯示控制選項和說明。
        
        Args:
            gpu_target: GPU目標溫度
            target_temp: 系統目標溫度
            experiment_mode: 實驗模式狀態
        """
        print("\n" + "="*50)
        print(f"{self.Colors.BOLD}📋 控制選項{self.Colors.RESET}")
        print("-"*50)
        print(f"📌 目標設定: GPU={gpu_target}°C | 冷卻水={target_temp}°C")
        if experiment_mode:
            print(f"🧪 實驗模式: 啟用中")
        print(f"📝 按下 Ctrl+C 停止程序")

class HardwareController:
    """硬體控制類
    
    管理系統的硬體組件，包括數據採集模組、風扇和泵。
    
    Attributes:
        adam: ADAM數據採集模組控制器
        fan1: 風扇1控制器
        fan2: 風扇2控制器
        pump: 泵控制器
    """
    def __init__(self, config: HardwareConfig, model_config: ModelConfig):
        """初始化硬體控制器。
        
        Args:
            config: 硬體配置對象
            model_config: 模型配置對象
        """
        self.adam = ADAMScontroller.DataAcquisition(
            exp_name=model_config.exp_name,
            exp_var=model_config.exp_var,
            port=config.adam_port,
            csv_headers=model_config.custom_headers
        )
        self.fan1 = multi_ctrl.multichannel_PWMController(config.fan1_port)
        self.fan2 = multi_ctrl.multichannel_PWMController(config.fan2_port)
        self.pump = ctrl.XYKPWMController(config.pump_port)

    def initialize_hardware(self, control_params: ControlParameters):
        """初始化硬體設備。
        
        設置初始控制參數並啟動數據採集。
        
        Args:
            control_params: 控制參數對象
        """
        self.pump.set_duty_cycle(control_params.initial_pump_duty)
        self.fan1.set_all_duty_cycle(control_params.initial_fan_duty)
        self.fan2.set_all_duty_cycle(control_params.initial_fan_duty)
        self.adam.start_adam()
        self.adam.update_duty_cycles(
            control_params.initial_fan_duty,
            control_params.initial_pump_duty
        )

    def cleanup(self):
        """清理硬體資源。
        
        停止數據採集並將執行器設置為安全狀態。
        """
        self.adam.stop_adam()
        self.fan1.set_all_duty_cycle(20)
        self.fan2.set_all_duty_cycle(20)
        self.pump.set_duty_cycle(40)

class CoolingSystemController:
    """冷卻系統主控制器
    
    整合硬體控制、模型預測控制和顯示管理的主控制器。
    實現了目標溫度更新的觀察者接口。
    
    Attributes:
        display: 顯示管理器
        hardware: 硬體控制器
        control_params: 控制參數
        sa_optimizer: 模擬退火優化器
        pump_controller: 泵PID控制器
        experiment_mode: 實驗模式控制器
        prev_states: 先前狀態記錄
        counter: 控制循環計數器
        running: 運行狀態標誌
    """
    def __init__(self,
                 hardware_config: HardwareConfig,
                 model_config: ModelConfig,
                 control_params: ControlParameters):
        """初始化冷卻系統控制器。
        
        Args:
            hardware_config: 硬體配置對象
            model_config: 模型配置對象
            control_params: 控制參數對象
        """
        self.display = DisplayManager()
        self.hardware = HardwareController(hardware_config, model_config)
        self.control_params = control_params
        
        # 註冊為控制參數的觀察者
        self.control_params.register_observer(self)
        
        # 初始化實驗模式
        self.experiment_mode = ExperimentMode(control_params)
        
        # 儲存原始 print 函數用於過濾日誌
        self.original_print = print
        
        # 初始化優化器和控制器
        self.sa_optimizer = HC_iTransformer.HC_Optimizer(
            adam=self.hardware.adam,
            window_size=40,
            P_max=control_params.p_max,
            target_temp=control_params.target_temp,
            model_path=model_config.model_path,
            scaler_path=model_config.scaler_path
        )
        
        self.pump_controller = Pump_pid.GB_PID_pump(
            target=control_params.gpu_target,
            Guaranteed_Bounded_PID_range=0.5,
            sample_time=1
        )

        # 狀態追蹤
        self.prev_states = {
            'temp_gpu': None,
            'temp_cdu': None,
            'fan_duty': None,
            'pump_duty': None
        }
        self.counter = 0
        self.running = True

    def update_target_temp(self, gpu_target: float, target_temp: float):
        """更新目標溫度的觀察者方法
        
        當控制參數中的目標溫度發生變化時被調用。
        
        Args:
            gpu_target: 新的GPU目標溫度
            target_temp: 新的系統目標溫度
        """
        if gpu_target is not None:
            self.hardware.adam.update_target_temperature(gpu_target_temperature=gpu_target)
            # 更新泵控制器的目標溫度
            self.pump_controller.update_target(gpu_target)
        if target_temp is not None:
            self.hardware.adam.update_target_temperature(target_temperature=target_temp)
            # 更新優化器的目標溫度
            self.sa_optimizer.target_temp = target_temp
        
        # 記錄溫度變化
        print(f"\n🎯 目標溫度已更新：GPU = {gpu_target}°C, 系統 = {target_temp}°C")
        
    def start_experiment_mode(self, period: int = 300, 
                           gpu_targets: List[float] = None,
                           system_targets: List[float] = None):
        """啟動實驗模式
        
        Args:
            period: 溫度變化週期（秒）
            gpu_targets: GPU目標溫度列表
            system_targets: 系統目標溫度列表
        """
        if gpu_targets or system_targets or period != 300:
            self.experiment_mode.set_parameters(
                period=period,
                gpu_targets=gpu_targets,
                system_targets=system_targets
            )
        self.experiment_mode.start()
        
    def stop_experiment_mode(self):
        """停止實驗模式"""
        self.experiment_mode.stop()
        
    def toggle_experiment_mode(self):
        """切換實驗模式狀態"""
        self.experiment_mode.toggle()

    def run(self):
        """運行控制系統。
        
        啟動控制循環並處理異常情況。
        """
        try:
            print("\n" + "="*50)
            print("📊 系統初始化中")
            print("⚡ 初始化SA優化器 | 冷卻水目標溫度: {}°C | 最大功率: {}%".format(
                self.control_params.target_temp, self.control_params.p_max))
            print("="*50)
            
            self.hardware.initialize_hardware(self.control_params)
            while self.running:
                # 檢查實驗模式是否需要更新目標溫度
                self.experiment_mode.update()
                
                # 執行控制循環
                self._control_loop()
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n⚠️ 程序已被手動停止")
        except Exception as e:
            print(f"\n❌ 發生錯誤: {e}")
        finally:
            self.hardware.cleanup()
            print(f"\n✅ 程序已結束，資源已釋放")

    def _control_loop(self):
        """控制迴圈的主要邏輯。
        
        執行一次控制循環，包括獲取溫度、計算控制輸出和更新執行器。
        """
        # 獲取溫度數據
        temps_data = self._get_temperatures()
        if not temps_data:
            return

        # 更新顯示
        self.display.clear_terminal()
        
        # 計算趨勢
        trends = {
            'gpu': self.display.get_trend(temps_data['T_GPU'], self.prev_states['temp_gpu']),
            'cdu': self.display.get_trend(temps_data['T_CDU_out'], self.prev_states['temp_cdu']),
            'fan': self.display.get_trend(temps_data['fan_duty'], self.prev_states['fan_duty']),
            'pump': self.display.get_trend(temps_data['pump_duty'], self.prev_states['pump_duty'])
        }
        
        # 更新歷史數據
        self.prev_states['temp_gpu'] = temps_data['T_GPU']
        self.prev_states['temp_cdu'] = temps_data['T_CDU_out']
        self.prev_states['fan_duty'] = temps_data['fan_duty']
        self.prev_states['pump_duty'] = temps_data['pump_duty']
        
        # 計算控制輸出
        control_temp = self.pump_controller.GB_PID(
            temps_data['T_GPU'],
            self.control_params.gpu_target
        )
        new_pump_duty = round(
            self.pump_controller.controller(control_temp) / 10
        ) * 10
        
        # 顯示溫度狀態
        self.display.display_temp_status(
            temps=temps_data,
            trends=trends,
            targets={
                'gpu_target': self.control_params.gpu_target,
                'cdu_target': self.control_params.target_temp
            }
        )
        
        # 顯示控制狀態
        self.display.display_control_status(
            duties={
                'pump_duty': temps_data['pump_duty'],
                'fan_duty': temps_data['fan_duty'],
                'new_pump_duty': new_pump_duty,
                'counter': self.counter
            },
            trends=trends
        )

        # 更新泵的轉速
        self.hardware.pump.set_duty_cycle(new_pump_duty)
        self.hardware.adam.update_duty_cycles(pump_duty=new_pump_duty)
        
        # 顯示控制策略
        self.display.display_control_strategy(control_temp, self.control_params.gpu_target)

        # 定期執行風扇優化
        if self.counter % self.control_params.control_frequency == 0:
            self.display.display_fan_optimization()
            
            # 替換SA_Optimizer中的print函數，減少輸出
            HC_iTransformer.print = lambda *args, **kwargs: self.filter_sa_logs(*args, **kwargs)
            
            start_time = time.time()
            optimal_fan_speed, optimal_cost = self.sa_optimizer.optimize()
            optimization_time = time.time() - start_time
            
            # 恢復原始print函數
            HC_iTransformer.print = print
            
            if optimal_fan_speed is not None:
                # 先顯示優化結果，再改變設定，使結果更容易被看到
                self.display.display_optimization_result(
                    optimal_fan_speed,
                    optimal_cost,
                    temps_data['fan_duty'],
                    optimization_time
                )
                
                # 更新風扇設定
                self.hardware.fan1.set_all_duty_cycle(int(optimal_fan_speed))
                self.hardware.fan2.set_all_duty_cycle(int(optimal_fan_speed))
                self.hardware.adam.update_duty_cycles(fan_duty=int(optimal_fan_speed))
                
                # 顯示優化歷史記錄
                self.display.display_optimization_history()
                
                # 顯示控制選項
                self.display.display_control_options(
                    self.control_params.gpu_target,
                    self.control_params.target_temp,
                    self.experiment_mode.enabled
                )
        
        self.counter += 1

    def _get_temperatures(self) -> Dict[str, float]:
        """獲取溫度數據。
        
        從數據採集模組獲取當前溫度和控制狀態。
        
        Returns:
            包含各測量點溫度和控制狀態的字典，如果無數據則返回空字典
        """
        temps = self.hardware.adam.buffer.tolist()
        if any(temps):
            return {
                'T_GPU': temps[0],
                'T_CDU_out': temps[3],
                'T_env': temps[4],
                'T_air_in': temps[5],
                'T_air_out': temps[6],
                'fan_duty': temps[8],
                'pump_duty': temps[9]
            }
        return {}

    def filter_sa_logs(self, *args, **kwargs):
        """過濾SA優化器輸出，只保留關鍵信息"""
        message = " ".join(map(str, args))
        
        # 關鍵信息清單 - 只保留這些信息
        key_phrases = [
            "數據蒐集完成",
            "初始解",
            "嘗試解",  # 迭代過程
            "預測溫度序列",
            "預測溫度變化方向",
            "斜率",
            "接受",    # 接受新解
            "拒絕",    # 拒絕新解
            "當前溫度", # 冷卻過程
            "發現更好的解", # 更新最佳解
            "最終解",
            "最佳化完成",
        ]
        
        # 檢查是否為關鍵信息
        if any(phrase in message for phrase in key_phrases):
            self.original_print(*args, **kwargs)

if __name__ == "__main__":

        # 配置示例
        hardware_config = HardwareConfig()
        model_config_parameters = HC_iTransformer.ModelConfig(
            input_dim=7,
            d_model=16,
            n_heads=8,
            e_layers=1,
            d_ff=16,
            dropout=0.01,
            seq_len=40,
            pred_len=8,
        )
        model_config = ModelConfig(
            scaler_path="/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/iTransformer/seq40_pred8_dmodel16_dff16_nheads8_elayers1_dropout0.01_lr0.0001_batchsize512_epochs140/scalers.jlib",
            model_path="/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/iTransformer/seq40_pred8_dmodel16_dff16_nheads8_elayers1_dropout0.01_lr0.0001_batchsize512_epochs140/best_model.pth",
            exp_name="/home/inventec/Desktop/2KWCDU_修改版本/data_manage/control_data/Fan_MPC_HC_data/iTransformer/seq40_pred8_dmodel16_dff16_nheads8_elayers1_dropout0.01_lr0.0001_batchsize512_epochs140",
            exp_var="Fan_MPC_data_var_target_test",
        )
        control_params = ControlParameters()

        
        # 創建並運行控制器
        controller = CoolingSystemController(
            hardware_config,
            model_config,
            control_params
        )
        
        # 測試實驗模式 (若需要測試，取消以下註釋)
        controller.start_experiment_mode(
            period=300,  # 5分鐘變化一次
            gpu_targets=[70,70,70,70,70,70],
            system_targets=[28,30,32,30,32,28]
        )
        
        controller.run() 

 
