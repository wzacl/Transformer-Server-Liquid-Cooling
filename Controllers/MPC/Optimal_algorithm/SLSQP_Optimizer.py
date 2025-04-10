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

def time_window_weight(step, total_steps=8):
    """計算時間窗口權重，使用高斯分佈"""
    mu = total_steps / 2
    sigma = total_steps / 4
    return np.exp(-((step - mu) ** 2) / (2 * sigma ** 2))

class SLSQP_Optimizer:
    def __init__(self, adam, window_size=35, P_max=100, target_temp=25,
                 model_path='/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400/2KWCDU_Transformer_model.pth',
                 scaler_path='/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400/1.5_1KWscalers.jlib',
                 figure_path='/home/inventec/Desktop/2KWCDU_修改版本/data_manage/control_data/Fan_MPC_FHO_data'):
        """
        SLSQP 風扇轉速最佳化器初始化
        :param adam: ADAM 控制器
        :param window_size: 時間窗口大小
        :param P_max: 風扇最大功耗 (W)
        :param target_temp: 目標 CDU 出水溫度 (°C)
        :param base_step_size: 基礎步長 (%)
        :param tolerance: 轉速變化容忍閾值 (%)
        :param stability_factor: 穩定性權重
        :param decision_history_size: 決策平滑所需的連續決策次數
        """
        # 優化參數
        self.P_max = P_max
        self.target_temp = target_temp
        self.best_solution = None
        self.best_cost = float('inf')
        self.cost_history = []
        
        # MPC 相關參數
        self.prediction_horizon = 8
        self.control_horizon = 1
        self.max_speed_change = 20  # 最大轉速變化限制
        
        # 目標函數權重
        self.w_temp = 2.0      # 溫度誤差權重
        self.w_speed = 1.0     # 速度變化權重
        self.w_power = 0.5     # 功率消耗權重
        
        # 模型相關
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
        self.previous_fan_speed = None  # 記錄上一次的風扇轉速

    def predict_temp(self, fan_speed, data):
        """使用 Transformer 模型進行溫度預測"""
        data_copy = data.copy()
        data_copy[-1][5] = fan_speed
        input_tensor = self.data_processor.transform_input_data(data_copy)

        if input_tensor is not None:
            with torch.no_grad():
                scaled_predictions = self.model(input_tensor, num_steps=self.prediction_horizon)[0].cpu().numpy()
                predicted_temps = self.data_processor.inverse_transform_predictions(scaled_predictions, smooth=True)
                return predicted_temps
        return None

    def objective_function(self, fan_speed, predicted_temps):
        """MPC 目標函數"""
        if predicted_temps is None:
            return float('inf')
            
        # 溫度控制項
        temp_error = 0
        for i, temp in enumerate(predicted_temps):
            time_weight = time_window_weight(i, self.prediction_horizon)
            temp_diff = abs(temp - self.target_temp)
            # 非對稱懲罰：過熱比過冷更嚴重
            if temp > self.target_temp:
                temp_error += time_weight * (temp_diff ** 2) * 1.5
            else:
                temp_error += time_weight * (temp_diff ** 2)
        
        # 速度平滑項
        speed_smooth = 0
        if self.previous_fan_speed is not None:
            speed_change = fan_speed - self.previous_fan_speed
            speed_smooth = speed_change ** 2
        
        # 功率消耗項
        power_consumption = (fan_speed/100) ** 3 * self.P_max
        
        # 總成本
        total_cost = (self.w_temp * temp_error + 
                     self.w_speed * speed_smooth + 
                     self.w_power * power_consumption)
        
        return total_cost

    def optimize(self):
        """使用 SLSQP 求解器進行優化"""
        fixed_window_data = self.data_processor.get_window_data(normalize=False)
        if fixed_window_data is None:
            return None, None
        else:
            print("✅ 數據蒐集完成，開始進行漸進式最佳化")
            
        current_temp = fixed_window_data[-1][1]
        
        # 定義 SLSQP 優化問題
        def objective(x):
            fan_speed = x[0]
            predicted_temps = self.predict_temp(fan_speed, fixed_window_data)
            return self.objective_function(fan_speed, predicted_temps)
        
        # 約束條件
        bounds = [(30, 100)]  # 風扇轉速範圍
        constraints = []
        
        if self.previous_fan_speed is not None:
            # 添加轉速變化約束
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: self.max_speed_change - abs(x[0] - self.previous_fan_speed)
            })
        
        # 初始猜測值
        if self.previous_fan_speed is not None:
            x0 = [self.previous_fan_speed]
        else:
            # 根據當前溫度設定初始轉速
            if current_temp > self.target_temp:
                x0 = [min(100, max(60, 60 + (current_temp - self.target_temp) * 10))]
            else:
                x0 = [50]
        
        try:
            # 使用 SLSQP 優化
            result = optimize.minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            if result.success:
                optimal_speed = round(result.x[0])  # 四捨五入到整數
                self.cost_history.append(result.fun)
                
                # 更新最佳解
                if result.fun < self.best_cost:
                    self.best_cost = result.fun
                    self.best_solution = optimal_speed
                
                # 更新上一次的風扇轉速
                self.previous_fan_speed = optimal_speed
                
                print(f"✅ 最佳化成功: 風扇轉速 = {optimal_speed}%, 目標函數值 = {result.fun:.2f}")
                return optimal_speed, result.fun
            else:
                print("❌ 最佳化失敗，保持當前轉速")
                return self.previous_fan_speed, float('inf')
                
        except Exception as e:
            print(f"❌ 最佳化過程出錯: {str(e)}")
            return self.previous_fan_speed, float('inf')

    def plot_cost(self):
        """繪製成本收斂圖"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.cost_history)), self.cost_history, 'b-o', label="成本")
        plt.xlabel("迭代次數")
        plt.ylabel("目標函數成本")
        plt.title("SLSQP最佳化 - 成本收斂圖")
        plt.grid(True)
        plt.legend()
        
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        save_path = os.path.join(self.figure_path, f"slsqp_cost_{timestamp}.png")
        plt.savefig(save_path)
        print(f"🖼️ 成本收斂圖已保存至: {save_path}")
        
        plt.show()

# 測試代碼
if __name__ == "__main__":
    optimizer = SLSQP_Optimizer(adam=None, target_temp=25, P_max=100)
    optimal_fan_speed, optimal_cost = optimizer.optimize()
    optimizer.plot_cost()
    print(f"\nOptimal Fan Speed: {optimal_fan_speed}% with Cost: {optimal_cost:.2f}")
