# /usr/bin/python3
# 漸進式風扇轉速最佳化器
# 用於最佳化風扇轉速以降低 CDU 出水溫度
import time
import sys
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Control_Unit')
sys.path.append('/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Controllers/MPC/Model_constructor')
import matplotlib.pyplot as plt
import numpy as np
import time
import Transformer
import torch
import Sequence_Window_Processor as swp
import scipy.optimize as optimize
import math
import os
import csv
import random


class SA_Optimizer:
    def __init__(self, adam, window_size=35, P_max=100, target_temp=25,
                 model_path='/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400/2KWCDU_Transformer_model.pth',
                 scaler_path='/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400/1.5_1KWscalers.jlib',
                 figure_path='/home/inventec/Desktop/2KWCDU_修改版本/data_manage/control_data/Fan_MPC_FHO_data'):
        """
        模擬退火(SA)風扇轉速最佳化器初始化
        """
        # 保留原有的模型和數據處理相關參數
        self.P_max = P_max
        self.target_temp = target_temp
        self.best_solution = None
        self.best_cost = float('inf')
        self.cost_history = []
        self.base_step = 5
        
        # 模擬退火參數
        self.T_max = 10.0  # 初始溫度
        self.T_min = 1.0    # 最終溫度
        self.alpha = 0.65   # 冷卻率
        self.max_iterations = 1  # 每個溫度的迭代次數
        self.max_speed_change = 15  # 最大轉速變化限制
        
        # 目標函數權重保持不變
        self.w_temp = 1
        self.w_power = 0.001
        
        # 保留原有的模型初始化代碼
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.figure_path = figure_path
        self.adam = adam
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Transformer.TransformerModel(input_dim=7, hidden_dim=16, 
            output_dim=1, num_layers=1, num_heads=8, dropout=0.01)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        self.data_processor = swp.SequenceWindowProcessor(window_size=window_size, 
            adams_controller=self.adam, scaler_path=self.scaler_path, device=self.device)
        self.previous_fan_speed = None

    def predict_temp(self, fan_speed, data):
        """使用 Transformer 模型進行溫度預測"""
        data_copy = data.copy()
        data_copy[-1][5] = fan_speed
        input_tensor = self.data_processor.transform_input_data(data_copy)

        if input_tensor is not None:
            with torch.no_grad():
                scaled_predictions = self.model(input_tensor, num_steps=8)[0].cpu().numpy()
                predicted_temps = self.data_processor.inverse_transform_predictions(scaled_predictions, smooth=True)
                return predicted_temps
        return None

    def objective_function(self, fan_speed, predicted_temps, error, current_temp):
        """目標函數，加入過熱與過冷的懲罰項"""
        if predicted_temps is None:
            return float('inf')

        #斜率變化計算項
        if predicted_temps is not None and len(predicted_temps) > 0:
            # 計算預測溫度的斜率
            predicted_slope = 0
            if len(predicted_temps) > 1:
                predicted_slope = (predicted_temps[0] - current_temp)
            
            # 根據error判斷期望的斜率方向
            # 如果error > 0，表示當前溫度高於目標溫度，鼓勵負斜率（降溫）
            # 如果error < 0，表示當前溫度低於目標溫度，鼓勵正斜率（升溫）
            desired_direction = -1 if error > 0 else 1
            actual_direction = -1 if predicted_slope < 0 else 1
            
            # 如果斜率方向與期望方向不一致，增加懲罰
            if desired_direction != actual_direction:
                slope_penalty = 200
            else:
                slope_penalty = 0
            
            # 如果溫度接近目標值，減少斜率懲罰以避免過度調整
            if abs(error) < 0.5:
                slope_penalty *= 0.5
        # 溫度控制項
        temp_error = 0

        # 速度平滑項
        speed_smooth = 0
        if self.previous_fan_speed is not None:
            speed_change = fan_speed - self.previous_fan_speed
            speed_smooth = speed_change ** 2
            
            # 當溫度與目標溫度接近時，增加速度平滑項的權重，使轉速更快收斂
            if abs(current_temp - self.target_temp) < 1.0:
                # 溫度越接近目標，速度平滑權重越高
                temp_diff_ratio = max(0.1, 1 - abs(current_temp - self.target_temp))
                smooth_weight = 3.0 * temp_diff_ratio  # 當溫度非常接近時，權重最高可達3.0
                speed_smooth *= smooth_weight
                
        # 只計算預測序列中所有溫度差
        for i in predicted_temps:
            temp_diff = abs(i - self.target_temp)
            if temp_diff > 0.3:
                temp_error += math.sqrt(temp_diff) * 20
            else:
                temp_error += 0

        # 功率消耗項
        power_consumption = (fan_speed/100) ** 3 * self.P_max
        
        # 總成本
        total_cost = (self.w_temp * temp_error + 
                     self.w_power * power_consumption  + slope_penalty + speed_smooth)
        
        return total_cost

    def generate_neighbor(self, current_speed, current_temp=None):
        """生成鄰近解"""
        if self.previous_fan_speed is not None:
            # 在當前溫度下動態調整步長
            max_change = min(self.max_speed_change, abs(self.T_current))
            # 確保變化是self.base_step的倍數
            max_steps = int(max_change / self.base_step)
            if max_steps == 0:
                max_steps = 1
                
            # 當系統溫度與目標溫度接近時，限制下限為1個基本步長
            if current_temp is not None and abs(current_temp - self.target_temp) < 0.5:
                min_steps = 1  # 最小步長為1個基本步長
                steps = random.randint(min_steps, max_steps) * (1 if random.random() > 0.5 else -1)
            else:
                steps = random.randint(-max_steps, max_steps)
                
            delta = steps * self.base_step
            new_speed = current_speed + delta
        else:
            # 首次運行時的範圍更大
            new_speed = random.uniform(40, 100)
            # 近似到最接近的self.base_step倍數
            new_speed = round(new_speed / self.base_step) * self.base_step
            
        # 確保在合理範圍內
        new_speed = max(40, min(100, new_speed))
        # 如果有前一個速度，確保變化不超過限制
        if self.previous_fan_speed is not None:
            max_change = self.max_speed_change
            lower_bound = self.previous_fan_speed - max_change
            upper_bound = self.previous_fan_speed + max_change
            new_speed = max(lower_bound, min(upper_bound, new_speed))
            # 近似到最接近的self.base_step倍數
            new_speed = round(new_speed / self.base_step) * self.base_step
        
        return int(new_speed)

    def optimize(self):
        """使用模擬退火算法進行優化"""
        fixed_window_data = self.data_processor.get_window_data(normalize=False)
        if fixed_window_data is None:
            return None, None

        else:
            current_temp = fixed_window_data[-1][1]
            past_temp = fixed_window_data[-10][1]
            error = current_temp - past_temp
            print("✅ 數據蒐集完成，開始進行模擬退火最佳化")
        
        # 初始解
        if self.previous_fan_speed is not None:
            current_speed = self.previous_fan_speed
        else:

            temp_change = abs(current_temp - past_temp)
            
            if abs(current_temp - self.target_temp) > 2:
                # 基本轉速計算
                base_speed = min(100, max(60, 60 + (current_temp - self.target_temp) * 10))
                # 如果溫度變化大於0.2，提高初始搜索轉速
                if temp_change > 0.2:   
                    current_speed = min(100, base_speed + temp_change * 5)
                else:
                    current_speed = base_speed
            else:
                current_speed = 50
        
        current_speed = round(current_speed)
        best_speed = current_speed
        
        # 計算初始解的成本
        predicted_temps = self.predict_temp(current_speed, fixed_window_data)
        current_cost = self.objective_function(current_speed, predicted_temps, error,current_temp)
        best_cost = current_cost
        
        # 顯示初始解的預測溫度變化方向
        if predicted_temps is not None and len(predicted_temps) > 0:
            predicted_slope = (predicted_temps[-1] - current_temp) / len(predicted_temps)
            direction = "降溫" if predicted_slope < 0 else "升溫"
            print(f"🌡️ 初始解: 風扇轉速 = {current_speed}%, 預測溫度變化方向: {direction}, 斜率: {predicted_slope:.4f}")
            # 顯示每個時間步的預測溫度
            print(f"   預測溫度序列: {[f'{temp:.2f}' for temp in predicted_temps]}")
        
        # 模擬退火主循環
        T = self.T_max
        while T > self.T_min:
            self.T_current = T  # 保存當前溫度用於生成鄰近解
            
            for _ in range(self.max_iterations):
                # 生成新解
                new_speed = self.generate_neighbor(current_speed, current_temp)
                predicted_temps = self.predict_temp(new_speed, fixed_window_data)
                new_cost = self.objective_function(new_speed, predicted_temps, error,current_temp)
                
                # 計算成本差異
                delta_cost = new_cost - current_cost
                
                # 顯示所有解的預測溫度變化方向
                if predicted_temps is not None and len(predicted_temps) > 0:
                    predicted_slope = (predicted_temps[-1] - current_temp) / len(predicted_temps)
                    direction = "降溫" if predicted_slope < 0 else "升溫"
                    print(f"🔍 嘗試解: 風扇轉速 = {new_speed}%, 預測溫度變化方向: {direction}, 斜率: {predicted_slope:.4f}, 成本: {new_cost:.2f}")
                    # 顯示每個時間步的預測溫度
                    print(f"   預測溫度序列: {[f'{temp:.2f}' for temp in predicted_temps]}")
                
                # Metropolis準則
                accept = delta_cost < 0 or random.random() < math.exp(-delta_cost / T)
                if accept:
                    current_speed = new_speed
                    current_cost = new_cost
                    
                    # 更新最佳解
                    if current_cost < best_cost:
                        best_speed = current_speed
                        best_cost = current_cost
                        print(f"🌟 發現更好的解: 風扇轉速 = {best_speed}%, 成本 = {best_cost:.2f}")
                
                # 顯示是否接受新解
                print(f"   {'✅ 接受' if accept else '❌ 拒絕'}此解")
            
            # 降溫
            T *= self.alpha
            print(f"🌡️ 當前溫度: {T:.2f}, 當前最佳轉速: {best_speed}%")
        
        # 更新歷史記錄
        self.cost_history.append(best_cost)
        self.previous_fan_speed = best_speed
        
        # 顯示最終解的預測溫度變化方向
        final_predicted_temps = self.predict_temp(best_speed, fixed_window_data)
        if final_predicted_temps is not None and len(final_predicted_temps) > 0:
            final_predicted_slope = (final_predicted_temps[-1] - current_temp) / len(final_predicted_temps)
            final_direction = "降溫" if final_predicted_slope < 0 else "升溫"
            print(f"📊 最終解: 風扇轉速 = {best_speed}%, 預測溫度變化方向: {final_direction}, 斜率: {final_predicted_slope:.4f}")
            # 顯示每個時間步的最終預測溫度
            print(f"   最終預測溫度序列: {[f'{temp:.2f}' for temp in final_predicted_temps]}")
        
        print(f"✅ 最佳化完成: 風扇轉速 = {best_speed}%, 最終成本 = {best_cost:.2f}")
        return best_speed, best_cost



# 測試代碼
if __name__ == "__main__":
    optimizer = SA_Optimizer(adam=None, target_temp=25, P_max=100)
    optimal_fan_speed, optimal_cost = optimizer.optimize()
    optimizer.plot_cost()
    print(f"\nOptimal Fan Speed: {optimal_fan_speed}% with Cost: {optimal_cost:.2f}")
