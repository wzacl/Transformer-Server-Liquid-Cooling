# /usr/bin/python3
# 漸進式風扇轉速最佳化器
# 用於最佳化風扇轉速以降低 CDU 出水溫度
import time
import sys
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Control_Unit')
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Controllers/MPC/Model_constructor')
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Controllers/MPC')
import matplotlib.pyplot as plt
import numpy as np
import time
from code_manage.Controllers.MPC.model import Model
import torch
from code_manage.Controllers.MPC.Model_constructor import Sequence_Window_Processor as swp
import math
import os
import csv
import random

import pandas as pd
import numpy as np
import torch
import argparse
import joblib
import os

# 設置設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用設備: {device}")

def parse_args():
    parser = argparse.ArgumentParser(description='使用iTransformer模型進行預測')
    
    # 模型和數據參數
    parser.add_argument('--model_path', type=str, required=True, 
                        help='訓練好的模型路徑')
    parser.add_argument('--scaler_path', type=str, required=True, 
                        help='歸一化器路徑')
    parser.add_argument('--input_file', type=str, required=True, 
                        help='輸入數據文件')
    parser.add_argument('--output_file', type=str, default='predictions.csv', 
                        help='預測結果輸出文件')
    
    # 特徵參數
    parser.add_argument('--features', type=str, 
                        default='T_GPU,T_heater,T_CDU_in,T_CDU_out,T_air_in,T_air_out,fan_duty,pump_duty', 
                        help='輸入特徵，以逗號分隔')
    parser.add_argument('--target', type=str, default='T_CDU_out', help='預測目標變量')
    
    # 預測參數
    parser.add_argument('--seq_length', type=int, default=20, help='輸入序列長度')
    parser.add_argument('--pred_length', type=int, default=6, help='預測序列長度')
    
    return parser.parse_args()

class ModelConfig:
    """
    模型配置類，統一管理模型參數
    """
    def __init__(self, input_dim=7, d_model=16, n_heads=8, e_layers=1, d_ff=16, 
                 dropout=0.01, seq_len=40, pred_len=8, embed='timeF', freq='h',
                 class_strategy='cls', activation='gelu', output_attention=False, use_norm=True):
        """
        初始化模型配置
        
        Args:
            input_dim (int): 輸入特徵維度
            d_model (int): 模型隱藏層維度
            n_heads (int): 注意力頭數
            e_layers (int): 編碼器層數
            d_ff (int): 前饋網絡維度
            dropout (float): Dropout比率
            seq_len (int): 輸入序列長度
            pred_len (int): 預測序列長度
            embed (str): 嵌入類型
            freq (str): 時間頻率
            class_strategy (str): 分類策略
            activation (str): 激活函數
            output_attention (bool): 是否輸出注意力權重
            use_norm (bool): 是否使用層歸一化
        """
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embed = embed
        self.freq = freq
        self.class_strategy = class_strategy
        self.activation = activation
        self.output_attention = output_attention
        self.use_norm = use_norm

def load_model(model_path, config):
    """
    加載訓練好的模型
    
    Args:
        model_path: 模型路徑
        config: 模型配置
        
    Returns:
        加載好的模型
    """
    model = Model(
        input_dim=config.input_dim,
        d_model=config.d_model,
        n_heads=config.n_heads,
        e_layers=config.e_layers,
        d_ff=config.d_ff,
        dropout=config.dropout
    ).to(device)
    
    # 加載模型權重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # 設置為評估模式
    model.eval()
    
    return model

class SA_Optimizer:
    def __init__(self, adam, window_size=35, P_max=100, target_temp=25,
                 model_path='/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/iTransformer/seq40_pred8_dmodel16_dff16_nheads8_elayers1_dropout0.01_lr0.0001_batchsize512_epochs140/best_model.pth',
                 scaler_path='/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/iTransformer/seq40_pred8_dmodel16_dff16_nheads8_elayers1_dropout0.01_lr0.0001_batchsize512_epochs140/scalers.jlib',
                 figure_path='/home/inventec/Desktop/2KWCDU_修改版本/data_manage/control_data/Fan_MPC_FHO_data'):
        """初始化模擬退火(SA)風扇轉速最佳化器。
        
        Args:
            adam: ADAM控制器實例。
            window_size (int, optional): 預測用的數據窗口大小。預設為35。
            P_max (int, optional): 最大功率值。預設為100。
            target_temp (int, optional): 目標維持溫度。預設為25。
            model_path (str, optional): 訓練好的transformer模型路徑。預設為預定義路徑。
            scaler_path (str, optional): 數據縮放器路徑。預設為預定義路徑。
            figure_path (str, optional): 輸出圖表保存路徑。預設為預定義路徑。
        """
        # 控制參數
        self.target_temp = target_temp  # 目標溫度
        self.P_max = P_max  # 最大功率值
        self.max_speed_change = 10  # 最大轉速變化限制
        self.previous_fan_speed = None  # 前一次風扇轉速
        self.back_step = 10  # 回退步長

        # 風扇轉速限制
        self.default_speed = 30  # 預設轉速
        self.min_speed = 30  # 最小轉速
        self.max_speed = 100  # 最大轉速

        
        # 動態轉速下限控制參數
        # 觸發條件說明：
        # 1. 動態轉速下限：在目標溫度正負1度之間觸發
        #    - 當目標溫度為28度時，轉速下限為65%
        #    - 當目標溫度為34度時，轉速下限為30%
        #    - 28-34度之間使用線性插值
        # 
        # 2. 最大轉速變化限制：在目標溫度正負0.5度之間觸發
        #    - 限制單次轉速變化不超過±10%
        # 
        # 3. 完全自由範圍：在大於目標溫度正負1度時
        #    - 轉速變動範圍為30-100%
        
        # 模擬退火參數
        self.T_max = 1.0  # 初始溫度，增加以允許更大範圍探索
        self.T_min = 0.1  # 最終溫度，降低以確保更精確的收斂
        self.alpha = 0.7  # 冷卻率，調整為較慢的降溫
        self.max_iterations = 10  # 每個溫度的迭代次數，增加以提高每個溫度的探索
        self.base_step = 5  # 基本步長，保持為5%
        
        # 目標函數參數
        self.w_temp = 1  # 溫度控制項權重
        self.w_speed = 0  # 速度平滑項權重
        self.w_energy = 0  # 能量消耗項權重
        self.error_band = 0.2  # 溫度控制項誤差帶
        
        # 最佳化結果追蹤
        self.best_solution = None  # 最佳解決方案
        self.best_cost = float('inf')  # 最佳成本值
        self.cost_history = []  # 成本歷史記錄
        
        # 模型和數據處理相關
        self.model_path = model_path  # 模型路徑
        self.scaler_path = scaler_path  # 縮放器路徑
        self.figure_path = figure_path  # 圖表保存路徑
        self.adam = adam  # ADAM控制器
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 計算設備
        
        # 使用統一的模型配置
        self.model_config = ModelConfig(
            input_dim=6,
            d_model=16,
            n_heads=2,
            e_layers=1,
            d_ff=32,
            dropout=0.01,
            seq_len=25,
            pred_len=8
        )
        
        # 創建模型實例 - 修正初始化方式
        self.model = Model(
            self.model_config
        ).to(self.device)
        
        # 載入模型權重 - 修正加載方式
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            # 檢查點包含模型狀態字典
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 直接嘗試加載
            self.model.load_state_dict(checkpoint)
        self.model.eval()  # 設置模型為評估模式
        
        # 數據處理器
        self.data_processor = swp.SequenceWindowProcessor(
            window_size=window_size, 
            adams_controller=self.adam, 
            scaler_path=self.scaler_path, 
            device=self.device
        )

    def predict_temp(self, fan_speed, data):
        """使用Transformer模型預測溫度。
        
        Args:
            fan_speed (float): 用於預測的風扇轉速值。
            data (list): 輸入數據序列。
            
        Returns:
            list or None: 預測的溫度序列，若預測失敗則返回None。
        """
        data_copy = data.copy()  # 複製數據以避免修改原始數據
        
        # 輸出原始數據的最後一行以確認特徵索引
        last_row = data_copy[-1].copy()
        print(f"🔍 原始數據: {[f'{val:.2f}' for val in last_row]}")
        print(f"📊 特徵順序: T_GPU, T_CDU_in, T_CDU_out, T_air_in, T_air_out, fan_duty, pump_duty")
        
 
        # 將風扇序列向左平移self.back_step個時間步
        data_copy[self.back_step:, 4] = data_copy[:-self.back_step, 4]  # 將序列向左平移self.back_step步
        data_copy[-self.back_step:, 4] = fan_speed  # 用新的風扇轉速填充後self.back_step個時間步
        
        input_tensor = self.data_processor.transform_input_data(data_copy)  # 轉換輸入數據為張量

        if input_tensor is not None:
            with torch.no_grad():
                # 檢查模型輸出
                model_output = self.model(input_tensor)
                
                # 輸出模型輸出的形狀以便調試
                print(f"📐 模型輸出形狀: {[output.shape for output in model_output if isinstance(output, torch.Tensor)]}")
                
                # 取得第一個輸出張量並轉換為NumPy數組
                scaled_predictions = model_output[0].cpu().numpy()  # 獲取縮放後的預測結果
                print(f"📊 原始縮放預測形狀: {scaled_predictions.shape}")
                
                # 使用修改後的反轉縮放方法
                predicted_temps = self.data_processor.inverse_transform_predictions(scaled_predictions)  # 反轉縮放
                
                return predicted_temps
        return None
    
    def fan_speed_energy(self, fan_speed):
        """計算風扇轉速的能量消耗。
        
        Args:
            fan_speed (float): 風扇轉速。
        """
        return (fan_speed*0.1) **3
    def objective_function(self, fan_speed, predicted_temps, error, current_temp):
        """計算最佳化的目標函數值。
        
        Args:
            fan_speed (float): 當前風扇轉速。
            predicted_temps (list): 預測的溫度序列。
            error (float): 溫度誤差。
            current_temp (float): 當前溫度。
            
        Returns:
            float: 目標函數值（成本）。
        """
        if predicted_temps is None:
            return float('inf')  # 若預測失敗，返回無窮大成本

        speed_energy = self.fan_speed_energy(fan_speed)

        # 只計算預測序列中所有溫度差
        temp_error = 0
        for i in predicted_temps:
            if abs(i - self.target_temp) > self.error_band:
                temp_diff = (abs(i - self.target_temp)*6)**2  # 溫度差的平方
                temp_error += temp_diff
            else:
                temp_error += 0
        
        # 計算當前溫度與目標溫度的差值
        temp_diff = current_temp - self.target_temp
        
        # 風扇轉速獎勵機制
        speed_reward = 0
        if temp_diff < -1:
            # 當前溫度比目標溫度低1度以上，獎勵最低風扇轉速
            speed_reward = -(self.max_speed - fan_speed) *0.7  # 越接近最低轉速，獎勵越大
        elif temp_diff > 1:
            # 當前溫度比目標溫度高1度以上，獎勵最高風扇轉速
            speed_reward = -(fan_speed - self.min_speed) * 0.7  # 越接近最高轉速，獎勵越大

        # 總成本（加入速度獎勵，負值表示獎勵會降低總成本）
        total_cost = self.w_temp * temp_error + self.w_energy * speed_energy + speed_reward
        
        return total_cost

    def calculate_dynamic_speed_limit(self, current_temp):
        """根據目標溫度計算動態轉速下限
        
        Args:
            current_temp (float): 當前溫度
            
        Returns:
            int: 動態計算的轉速下限，若不在溫度範圍內則返回預設下限30%
        """
        # 檢查是否在目標溫度正負1度範圍內（動態轉速下限觸發條件）
        if abs(current_temp - self.target_temp) <= 1:
            '''''
            # 根據目標溫度計算轉速下限
            if self.target_temp <= 28:
                # 目標溫度小於等於28度時，下限為65%
                dynamic_limit = 65
                print(f"🎯 動態轉速下限啟用: 目標溫度={self.target_temp}°C (≤28), 當前溫度={current_temp:.1f}°C, 下限={dynamic_limit}%")
            elif self.target_temp >= 34:
                # 目標溫度大於等於34度時，下限為30%
                dynamic_limit = 30
                print(f"🎯 動態轉速下限啟用: 目標溫度={self.target_temp}°C (≥34), 當前溫度={current_temp:.1f}°C, 下限={dynamic_limit}%")
            else:
                # 目標溫度在28-34度之間，使用線性插值
                # 目標溫度28度 -> 轉速下限65%
                # 目標溫度34度 -> 轉速下限30%
                speed_limit = 65 - (65 - 30) * (self.target_temp - 28) / (34 - 28)
                dynamic_limit = max(30, min(65, int(speed_limit // 5 * 5)))  # 確保是5的倍數且在合理範圍內
                print(f"🎯 動態轉速下限啟用: 目標溫度={self.target_temp}°C (28-34範圍), 當前溫度={current_temp:.1f}°C, 計算下限={dynamic_limit}%")
            '''''
            return 45
        else:
            # 不在正負1度範圍內，返回預設下限30%
            print(f"📊 溫度差異 {abs(current_temp - self.target_temp):.1f}°C > 1.0°C: 使用預設下限30%")
            return 30

    def generate_neighbor(self, current_speed, current_temp=None):
        """生成鄰近解。確保生成的風扇轉速始終是5%的倍數，以匹配控制系統的實際步長。
        
        Args:
            current_speed (float): 當前風扇轉速
            current_temp (float): 當前溫度，用於計算動態轉速下限
            
        Returns:
            int: 新生成的風扇轉速值，保證是5%的倍數
        """
        # 初始化步長為5%，對應實際風扇調節的最小單位
        
        if self.previous_fan_speed is not None:
            # 根據當前溫度決定搜索寬度
            # 但始終保持步長為5的倍數
            max_steps = max(1, int(self.T_current))  # 至少允許1個步長的變化
            
            # 隨機選擇步數（以5%為單位）
            step_count = random.randint(-max_steps, max_steps)
            
            # 計算轉速變化，確保是5的倍數
            delta = step_count * self.base_step
            
            # 計算新的轉速值
            new_speed = current_speed + delta
        else:
            # 首次運行，隨機生成一個5%的倍數作為初始解
            # 從60%到100%之間，以5%為步長生成隨機值
            possible_speeds = list(range(60, 105, 5))  # [60, 65, 70, ..., 100]
            new_speed = random.choice(possible_speeds)
        
        # 計算動態轉速下限
        if current_temp is not None:
            min_speed = self.calculate_dynamic_speed_limit(current_temp)
        else:
            min_speed = self.default_speed  # 預設最低轉速
        
        # 確保轉速值在有效範圍內（動態下限%-100%）
        # 並且結果為5的倍數（向下取整到最近的5的倍數）
        new_speed = max(min_speed, min(self.max_speed, new_speed))
        new_speed = int(new_speed // self.base_step * self.base_step)  # 確保是5的倍數
        
        return int(new_speed)

    def optimize(self):
        """執行模擬退火最佳化算法。
        
        Returns:
            tuple: 包含(最佳風扇轉速, 最佳成本)的元組，若數據收集失敗則返回(None, None)。
        """
        fixed_window_data = self.data_processor.get_window_data(normalize=False)  # 獲取窗口數據
        if fixed_window_data is None:
            return None, None

        else:
            current_temp = self.adam.buffer[3]  # 當前溫度
            # 移除對未定義變量的引用，直接將誤差設為0
            error = 0  # 初始化誤差為0
            print("✅ 數據蒐集完成，開始進行模擬退火最佳化")
        
        # 初始解
        if self.adam.buffer[8] is not None:
            self.previous_fan_speed = self.adam.buffer[8]
            current_speed = self.previous_fan_speed  # 使用前一次的風扇轉速
        else:
            self.adam.update_duty_cycles(fan_duty=60)
            current_speed = self.adam.buffer[8]  # 默認轉速
        
        best_speed = current_speed  # 最佳轉速初始值
        
        # 計算初始解的成本
        predicted_temps = self.predict_temp(current_speed, fixed_window_data)  # 預測溫度
        current_cost = self.objective_function(fan_speed=current_speed, predicted_temps=predicted_temps, error=error, current_temp=current_temp)  # 計算當前成本
        best_cost = current_cost  # 最佳成本初始值
        
        # 顯示初始解的預測溫度變化方向
        if predicted_temps is not None and len(predicted_temps) > 0:
            predicted_slope = (predicted_temps[-1] - current_temp) / len(predicted_temps)  # 預測溫度斜率
            direction = "降溫" if predicted_slope < 0 else "升溫"  # 溫度變化方向
            print(f"🌡️ 初始解: 風扇轉速 = {current_speed}% (5%的倍數), 預測溫度變化方向: {direction}, 斜率: {predicted_slope:.4f}")
            # 顯示每個時間步的預測溫度
            print(f"   預測溫度序列: {[f'{temp:.2f}' for temp in predicted_temps]}")
        
        # 模擬退火主循環
        T = self.T_max  # 初始溫度
        while T > self.T_min:
            self.T_current = T  # 保存當前溫度用於生成鄰近解
            
            for _ in range(self.max_iterations):
                # 生成新解
                new_speed = self.generate_neighbor(current_speed, current_temp)  # 生成鄰近解，傳入當前溫度
                predicted_temps = self.predict_temp(new_speed, fixed_window_data)  # 預測新解的溫度
                new_cost = self.objective_function(fan_speed=new_speed, predicted_temps=predicted_temps, error=error, current_temp=current_temp)  # 計算新解的成本
                
                # 計算成本差異
                delta_cost = new_cost - current_cost  # 成本變化
                
                # 顯示所有解的預測溫度變化方向
                if predicted_temps is not None and len(predicted_temps) > 0:
                    predicted_slope = (predicted_temps[-1] - current_temp) / len(predicted_temps)  # 預測溫度斜率
                    direction = "降溫" if predicted_slope < 0 else "升溫"  # 溫度變化方向
                    print(f"🔍 嘗試解: 風扇轉速 = {new_speed}% (步長: 5%), 預測溫度變化方向: {direction}, 斜率: {predicted_slope:.4f}, 成本: {new_cost:.2f}")
                    # 顯示每個時間步的預測溫度
                    print(f"   預測溫度序列: {[f'{temp:.2f}' for temp in predicted_temps]}")
                
                '''Metropolis準則
                如果新解的成本比當前解更低，則接受新解
                如果新解的成本比當前解更高，則以一定的概率接受新解，這個概率與溫度T和成本差異delta_cost有關
                '''
                accept = delta_cost < 0 or random.random() < math.exp(-delta_cost / T)
                if accept :
                    current_speed = new_speed  # 更新當前解
                    current_cost = new_cost  # 更新當前成本
                    
                    # 更新最佳解
                    if current_cost < best_cost:
                        best_speed = current_speed  # 更新最佳轉速
                        best_cost = current_cost  # 更新最佳成本
                        print(f"🌟 發現更好的解: 風扇轉速 = {best_speed}%, 成本 = {best_cost:.2f}")
                
                # 顯示是否接受新解
                print(f"   {'✅ 接受' if accept else '❌ 拒絕'}此解")
            
            # 降溫
            T *= self.alpha  # 溫度下降
            print(f"🌡️ 當前溫度: {T:.2f}, 當前最佳轉速: {best_speed}%")
        
        # 更新歷史記錄
        self.cost_history.append(best_cost)  # 記錄成本歷史
        self.previous_fan_speed = best_speed  # 更新前一次風扇轉速
        
        # 顯示最終解的預測溫度變化方向
        final_predicted_temps = self.predict_temp(best_speed, fixed_window_data)  # 預測最終解的溫度
        if final_predicted_temps is not None and len(final_predicted_temps) > 0:
            final_predicted_slope = (final_predicted_temps[-1] - current_temp) / len(final_predicted_temps)  # 最終預測溫度斜率
            final_direction = "降溫" if final_predicted_slope < 0 else "升溫"  # 最終溫度變化方向
            print(f"📊 最終解: 風扇轉速 = {best_speed}%, 預測溫度變化方向: {final_direction}, 斜率: {final_predicted_slope:.4f}")
            # 顯示每個時間步的最終預測溫度
            print(f"   最終預測溫度序列: {[f'{temp:.2f}' for temp in final_predicted_temps]}")
        
        dynamic_min_speed = self.calculate_dynamic_speed_limit(current_temp)
        current_system_speed = self.adam.buffer[8] if self.adam.buffer[8] is not None else 60
        
        # 根據當前溫度與目標溫度的差異決定轉速控制策略
        temp_diff = abs(self.adam.buffer[3] - self.target_temp)
        
        if temp_diff <= 0.5:
            # 在目標溫度正負0.5度之間：應用最大轉速變化限制
            max_change = self.max_speed_change  # 使用已定義的最大變化率
            
            # 計算允許的轉速範圍（限制變化幅度）
            min_allowed_speed = max(dynamic_min_speed, current_system_speed - max_change)
            max_allowed_speed = min(100, current_system_speed + max_change)
            
            print(f"🎯 溫度差異 {temp_diff:.1f}°C ≤ 0.5°C: 應用最大轉速變化限制 ±{max_change}%")
            
        elif temp_diff <= 1.0:
            # 在目標溫度正負0.5-1.0度之間：應用動態轉速下限，但允許較大變化
            min_allowed_speed = dynamic_min_speed
            max_allowed_speed = 100
            
            print(f"🎯 溫度差異 {temp_diff:.1f}°C 在0.5-1.0°C之間: 應用動態轉速下限 {dynamic_min_speed}%，允許到100%")
            
        else:
            # 大於目標溫度正負1度：轉速變動範圍為30到100
            min_allowed_speed = 30
            max_allowed_speed = 100
            
            print(f"🎯 溫度差異 {temp_diff:.1f}°C > 1.0°C: 轉速變動範圍為30-100%")
        
        # 限制最佳風扇轉速在允許範圍內
        if best_speed < min_allowed_speed:
            best_speed = int(min_allowed_speed)
            print(f"⚠️ 轉速調整: {best_speed}% -> {min_allowed_speed}% (低於下限)")
        elif best_speed > max_allowed_speed:
            best_speed = int(max_allowed_speed)
            print(f"⚠️ 轉速調整: {best_speed}% -> {max_allowed_speed}% (超過上限)")
        
        print(f"✅ 最佳化完成: 風扇轉速 = {best_speed}%, 最終成本 = {best_cost:.2f}")
        print(f"📊 轉速範圍: {min_allowed_speed}% - {max_allowed_speed}%")
        return best_speed, best_cost

    def plot_cost(self):
        """繪製成本歷史圖表"""
        if len(self.cost_history) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(self.cost_history)), self.cost_history, marker='o')
            plt.title('模擬退火最佳化成本歷史')
            plt.xlabel('迭代次數')
            plt.ylabel('成本值')
            plt.grid(True)
            
            # 確保圖表保存目錄存在
            os.makedirs(self.figure_path, exist_ok=True)
            
            # 生成時間戳記
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # 保存圖表
            plt.savefig(f"{self.figure_path}/sa_cost_history_{timestamp}.png")
            plt.close()
            print(f"✅ 成本歷史圖表已保存至: {self.figure_path}/sa_cost_history_{timestamp}.png")


# 測試代碼
if __name__ == "__main__":
    # 創建模型配置
    model_config = ModelConfig(
        input_dim=7,
        d_model=16,
        n_heads=8,
        e_layers=1,
        d_ff=16,
        dropout=0.01,
        seq_len=25,
        pred_len=8
    )
    
    # 創建優化器實例
    optimizer = SA_Optimizer(
        adam=None,
        target_temp=25,
        P_max=100,
        window_size=35
    )
    
    # 執行優化
    optimal_fan_speed, optimal_cost = optimizer.optimize()
    
    # 繪製成本歷史
    optimizer.plot_cost()
    
    # 顯示最佳結果
    print(f"\n最佳風扇轉速: {optimal_fan_speed}%, 最佳成本: {optimal_cost:.2f}")
