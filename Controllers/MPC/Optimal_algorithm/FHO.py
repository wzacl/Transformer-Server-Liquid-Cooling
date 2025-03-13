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
                 window_size=20,
                 model_path='/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/2KWCDU_Transformer_model.pth',
                 scaler_path='/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/1.5_1KWscalers.jlib',
                 figure_path='/home/inventec/Desktop/2KWCDU_修改版本/data_manage/Real_time_Prediction/Model_test_change_fan_pump_3.csv'):
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
        self.model = Transformer.TransformerModel(input_dim=7, hidden_dim=8, 
        output_dim=1, num_layers=1, num_heads=8, dropout=0.01)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        self.data_processor = swp.Sequence_Window_Processor(window_size=window_size, 
        scaler_path=self.scaler_path, device=self.device,adam=self.adam)


    def predict_temp(self, fan_speed,data):
        """ 使用即時預測功能預測 CDU 出水溫度 """
            # 獲取當前的序列資料
        data[-1][5]=fan_speed
        # 將最後一個時間步的風扇轉速替換為fan_speed這個參數輸入
        # 準備輸入數據
        input_tensor = self.data_processor.transform_input_data(data)

        if input_tensor is not None:
            with torch.no_grad():
                scaled_predictions = self.model(input_tensor, num_steps=1)[0].cpu().numpy()
                predicted_temp = self.data_processor.inverse_transform_predictions(scaled_predictions)[0]
            return predicted_temp
        else:
            return None

    def objective_function(self, fan_speed):
        """ 目標函數：最小化溫度誤差 + 風扇功耗 """
        temp_error = np.linalg.norm(self.predict_temp(fan_speed) - self.target_temp) ** 2
        power_fan = (fan_speed / 100) ** 3 * self.P_max
        return temp_error + power_fan

    def optimize(self):
        """ 執行火鷹最佳化過程，確保整個搜索過程基於同一組數據 """
        # 先從 sequence_window 取得當前時間點的固定數據
        fixed_window_data = self.data_processor.get_window_data(normalize=False)

        # 初始化火鷹位置（隨機選擇風扇轉速）
        firehawks = np.random.choice(self.fan_speeds, self.num_firehawks)

        if fixed_window_data is None:
            return None, None
        else:
            print("✅ 數據蒐集完成，開始進行最佳化")  
            for iteration in range(self.max_iter):
                # 使用固定的 window_data，避免搜尋過程中數據變動
                costs = []
                for fan in firehawks:
                    self.data_processor.override_fan_speed = fan  # 覆蓋風扇轉速
                    predicted_temp = self.predict_temp_with_fixed_data(fan, fixed_window_data)  # 透過固定數據進行預測
                    
                    if predicted_temp is not None:
                        cost = self.objective_function(fan, predicted_temp)
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

                # 火焰擴散機制（搜尋擴展）
                firehawks = np.clip(
                    best_firehawk + np.random.uniform(-10, 10, self.num_firehawks), 30, 100
                )
                firehawks = np.round(firehawks / 10) * 10  # 保持在 10% 單位

                # 記錄歷史成本
                self.cost_history.append(self.best_cost)

                print(f"Iteration {iteration+1}: Best Fan Speed = {self.best_solution}%, Cost = {self.best_cost:.2f}")
                
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
    optimizer = FirehawkOptimizer(num_firehawks=2, max_iter=10, target_temp=25)
    optimal_fan_speed, optimal_cost = optimizer.optimize()
    optimizer.plot_cost()

    print(f"\nOptimal Fan Speed: {optimal_fan_speed}% with Cost: {optimal_cost:.2f}")
