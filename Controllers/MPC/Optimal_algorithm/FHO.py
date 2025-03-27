# /usr/bin/python3
# 火鷹演算法 (Firehawk Optimization)
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
import math
import os
import csv




class FirehawkOptimizer:
    def __init__(self, adam, num_firehawks=10, max_iter=50, fan_speeds=None, P_max=100, target_temp=25,
                 window_size=35,
                 model_path='/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400/2KWCDU_Transformer_model.pth',
                 scaler_path='/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400/1.5_1KWscalers.jlib',
                 figure_path='/home/inventec/Desktop/2KWCDU_修改版本/data_manage/control_data/Fan_MPC_FHO_data'):
        """
        火鷹演算法 (FHO) 初始化
        :param num_firehawks: 火鷹數量 (搜尋代理)
        :param max_iter: 最大迭代次數
        :param fan_speeds: 風扇轉速可選範圍 (預設 30% - 100%)
        :param P_max: 風扇最大功耗 (W)
        :param target_temp: 目標 CDU 出水溫度 (°C)
        """
        self.num_firehawks = num_firehawks
        self.max_iter = max_iter
        self.fan_speeds = fan_speeds if fan_speeds is not None else np.arange(30, 110, 10)
        self.P_max = P_max
        self.target_temp = target_temp
        self.best_solution = None
        self.best_cost = float('inf')
        self.cost_history = []
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
        self.previous_fan_speed = None  # 添加这行来记录上一次的风扇转速


    def predict_temp(self, fan_speed, data):
        # 應該先創建副本再修改
        data_copy = data.copy()
        data_copy[-1][5] = fan_speed  # 修改副本
        # 準備輸入數據
        input_tensor = self.data_processor.transform_input_data(data_copy)

        if input_tensor is not None:
            with torch.no_grad():
                scaled_predictions = self.model(input_tensor, num_steps=8)[0].cpu().numpy()  # 預測8步
                predicted_temps = self.data_processor.inverse_transform_predictions(scaled_predictions)  # 返回所有8步預測
            return predicted_temps
        else:
            return None

    def objective_function(self, fan_speed, predicted_temps):
        """ 目標函數：考慮未來8步的溫度誤差、預測準確度變化和轉速懲罰 """
        # 計算預測準確度降低因子
        accuracy_decay = 0
        if self.previous_fan_speed is not None:
            # 轉速變化量越大，預測準確度越低
            speed_change = abs(fan_speed - self.previous_fan_speed)
            # 基礎預測誤差0.2，再加上與轉速變化相關的部分
            accuracy_decay = 0.2 + (speed_change / 100) * 0.5  # 根據轉速變化量調整
        
        # 計算所有8步的溫度誤差，考慮預測準確度降低
        temp_errors = []
        for i, temp in enumerate(predicted_temps):
            time_weight = 1.0 / (i + 1)
            accuracy_factor = 1.0 + accuracy_decay * (i + 1) * 0.1
            
            # 計算修正後的溫度
            if self.previous_fan_speed is not None and fan_speed < self.previous_fan_speed:
                temp_correction = temp + (self.previous_fan_speed - fan_speed) / 100 * 0.5
            else:
                temp_correction = temp
            
            # 溫度差
            temp_diff = abs(temp_correction - self.target_temp)
            
            # 混合懲罰：對小誤差使用線性懲罰，對大誤差使用平方懲罰
            if temp_diff < 0.3:
                # 線性懲罰，放大小誤差的影響
                error = temp_diff * 3.0 * time_weight * accuracy_factor
            else:
                # 平方懲罰，保持對大誤差的敏感度
                error = temp_diff ** 2 * time_weight * accuracy_factor
            
            temp_errors.append(error)
        
        # 總溫度誤差
        total_temp_error = sum(temp_errors)
        
        # 風扇功耗
        power_fan = (fan_speed / 100) ** 3 * self.P_max * 0.01  # 加入功耗考量但權重較小
        
        # 修改過熱懲罰
        overheat_penalty = 0
        for i, temp in enumerate(predicted_temps):
            step_weight = (i + 1) / len(predicted_temps)
            if temp > self.target_temp:
                temp_over = temp - self.target_temp
                
                # 混合懲罰方式
                if temp_over < 0.3:
                    # 小過熱使用線性懲罰但加大權重
                    overheat_penalty += temp_over * 5.0 * step_weight
                else:
                    # 大過熱仍使用平方懲罰
                    overheat_penalty += temp_over ** 2 * 3.0 * step_weight
                
                # 嚴重過熱仍保持更高懲罰
                if temp > (self.target_temp + 2):
                    overheat_penalty += temp_over ** 3 * 2.0
        
        # 轉速變化懲罰 - 修改為溫度敏感型
        speed_change_penalty = 0
        if self.previous_fan_speed is not None:
            speed_change = abs(fan_speed - self.previous_fan_speed)
            
            # 檢查是否有過熱風險
            overheating_risk = any(t > self.target_temp for t in predicted_temps)
            
            # 如果有過熱風險且嘗試提高風扇轉速，幾乎不懲罰
            if overheating_risk and fan_speed > self.previous_fan_speed:
                # 過熱時增加風扇轉速幾乎無懲罰
                if speed_change > 50:  # 只有極端變化才懲罰
                    speed_change_penalty = (speed_change - 50) ** 2 * 0.01
            # 如果沒有過熱風險，使用正常邏輯
            else:
                if self.previous_fan_speed < fan_speed:
                    if speed_change > 30:
                        speed_change_penalty = (speed_change - 30) ** 2 * 0.05
                else:
                    if speed_change > 20:
                        speed_change_penalty = (speed_change - 20) ** 2 * 0.15
        
        # 添加直接風扇轉速獎勵（當溫度接近或超過目標值時）
        fan_reward = 0
        avg_temp = sum(predicted_temps) / len(predicted_temps)
        
        # 當平均溫度接近或超過目標值時，獎勵更高的風扇轉速
        if avg_temp >= (self.target_temp - 1):
            # 溫度越高，獎勵越大
            temp_factor = max(0, (avg_temp - (self.target_temp - 1))) ** 2
            fan_reward = (fan_speed / 100) * temp_factor * 4.0
        
        # 總成本 - 添加風扇獎勵（負號表示獎勵）
        total_cost = (total_temp_error * 2 + 
                      speed_change_penalty +
                      overheat_penalty - 
                      fan_reward)  # 獎勵高風扇轉速（過熱時）
        
        return total_cost

    def optimize(self):
        """ 執行火鷹最佳化過程，確保整個搜索過程基於同一組數據 """
        # 先從 sequence_window 取得當前時間點的固定數據
        fixed_window_data = self.data_processor.get_window_data(normalize=False)

        # 修改火鷹擴散機制
        if fixed_window_data is None:
            return None, None
        else:
            print("✅ 數據蒐集完成，開始進行最佳化")
            
            # 檢查當前溫度狀況
            current_temp = self.data_processor.get_window_data(normalize=False)[-1][4]
            
            # 如果當前溫度已經超過目標，則初始化較高的風扇轉速
            if current_temp is not None and current_temp > self.target_temp:
                # 溫度越高，初始風扇轉速越高
                temp_diff = current_temp - self.target_temp
                init_fan_speed = min(100, max(60, 60 + temp_diff * 10))
                firehawks = np.clip(np.random.normal(init_fan_speed, 10, self.num_firehawks), 30, 100)
            else:
                # 正常初始化
                firehawks = np.random.choice(self.fan_speeds, self.num_firehawks)
            
            firehawks = np.round(firehawks / 10) * 10  # 保持在 10% 單位

            for iteration in range(self.max_iter):
                # 使用固定的 window_data，避免搜尋過程中數據變動
                costs = []
                for fan in firehawks:
                    self.data_processor.override_fan_speed = fan  # 覆蓋風扇轉速
                    predicted_temps = self.predict_temp(fan, fixed_window_data)  # 透過固定數據進行預測
                    
                    if predicted_temps is not None:
                        cost = self.objective_function(fan, predicted_temps)
                        costs.append(cost)
                
                if not costs:
                    print("❌ 無有效數據，跳過此迭代")
                    continue

                best_idx = np.argmin(costs)  # 找到最佳火鷹
                best_firehawk = firehawks[best_idx]

                # 更新全域最佳解
                if costs[best_idx] < self.best_cost:
                    self.best_cost = costs[best_idx]
                    self.best_solution = best_firehawk
                
                # 更新上一次的风扇转速
                self.previous_fan_speed = self.best_solution

                # 火焰擴散機制（限制搜索范围）
                firehawks = np.clip(
                    best_firehawk + np.random.uniform(-10, 10, self.num_firehawks),  # 减小随机变化范围
                    max(30, best_firehawk - 20),  # 限制最大下降幅度
                    min(100, best_firehawk + 20)  # 限制最大上升幅度
                )
                firehawks = np.round(firehawks / 10) * 10  # 保持在 10% 單位

                # 記錄歷史成本
                self.cost_history.append(self.best_cost)

                print(f"Iteration {iteration+1}: Best Fan Speed = {self.best_solution}%, Cost = {self.best_cost:.2f}")
            
            # 循环结束后返回结果
            return self.best_solution, self.best_cost



    def plot_cost(self):
        """ 繪製成本收斂圖 """
        plt.plot(range(len(self.cost_history)), self.cost_history, label="Cost")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Cost")
        plt.title("Firehawk Optimization - Cost Reduction")
        plt.legend()
        plt.show()

# 使用 FHO 來最佳化風扇轉速
if __name__ == "__main__":
    optimizer = FirehawkOptimizer(num_firehawks=2, max_iter=10, target_temp=25, P_max=100)
    optimal_fan_speed, optimal_cost = optimizer.optimize()
    optimizer.plot_cost()

    print(f"\nOptimal Fan Speed: {optimal_fan_speed}% with Cost: {optimal_cost:.2f}")
