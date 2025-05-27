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
import Transformer_quantized
import torch
import Sequence_Window_Processor as swp
import math
import os
import csv
import random


class Revised_SA_Optimizer:
    def __init__(self, adam, window_size=35, P_max=100, target_temp=25,
                 model_path='/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400/2KWCDU_Transformer_model.pth',
                 scaler_path='/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400/1.5_1KWscalers.jlib',
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
        self.max_speed_change = 15  # 最大轉速變化限制
        self.previous_fan_speed = None  # 前一次風扇轉速
        
        # 模擬退火參數
        self.T_max = 15.0  # 初始溫度
        self.T_min = 5.0  # 最終溫度
        self.alpha = 0.67  # 冷卻率，每次下降
        self.max_iterations = 4  # 每個溫度的迭代次數
        self.base_step = 5  # 基本步長
        
        # 目標函數參數
        self.w_temp = 1  # 溫度控制項權重
        self.w_speed = 0  # 速度平滑項權重
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
        self.model = Transformer_quantized.TransformerModel(input_dim=7, hidden_dim=16, output_dim=1, num_layers=1, num_heads=8, dropout=0.01)  # Transformer模型
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))  # 載入模型權重
        self.model.eval()  # 設置模型為評估模式
        self.data_processor = swp.SequenceWindowProcessor(window_size=window_size, 
            adams_controller=self.adam, scaler_path=self.scaler_path, device=self.device)  # 數據處理器

    def predict_temp(self, fan_speed, data):
        """使用Transformer模型預測溫度。
        
        Args:
            fan_speed (float): 用於預測的風扇轉速值。
            data (list): 輸入數據序列。
            
        Returns:
            list or None: 預測的溫度序列，若預測失敗則返回None。
        """
        data_copy = data.copy()  # 複製數據以避免修改原始數據
        data_copy[-1][5] = fan_speed  # 設置風扇轉速值
        input_tensor = self.data_processor.transform_input_data(data_copy)  # 轉換輸入數據為張量

        if input_tensor is not None:
            with torch.no_grad():
                scaled_predictions = self.model(input_tensor, num_steps=8)[0].cpu().numpy()  # 獲取縮放後的預測結果
                predicted_temps = self.data_processor.inverse_transform_predictions(scaled_predictions, smooth=False)  # 反轉縮放
                # 將預測溫度四捨五入到小數點後第一位
                rounded_temps = [round(temp, 1) for temp in predicted_temps]
                return rounded_temps
        return None

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

        # 速度平滑項
        speed_smooth = 0
        if self.previous_fan_speed is not None:
            speed_change = abs(fan_speed - self.previous_fan_speed)
            speed_smooth = speed_change**2 
      
        temp_error = 0
        # 只計算預測序列中所有溫度差
        for i in predicted_temps:
            if abs(i - self.target_temp) > self.error_band:
                temp_diff = (abs(i - self.target_temp)*10)**2  # 溫度差的平方
                temp_error += temp_diff
            else:
                temp_error += 0

        # 總成本
        total_cost =self.w_temp * temp_error  # 總成本等於溫度誤差
        
        return total_cost

    def generate_neighbor(self, current_speed):
        """為當前風扇轉速生成鄰近解。
        
        Args:
            current_speed (float): 當前風扇轉速。
            
        Returns:
            int: 新的風扇轉速值。
        """
        if self.previous_fan_speed is not None:
            max_steps = int(abs(self.T_current) / self.base_step)  # 根據當前溫度計算最大步數
            # 特殊處理邊界值情況
            if current_speed == 40:  # 當轉速為最小值時，只能向上生成
                steps = random.randint(0, max_steps)  # 隨機正步長
            elif current_speed == 100:  # 當轉速為最大值時，只能向下生成
                steps = random.randint(-max_steps, 0)  # 隨機負步長
            else:  # 非邊界值時，正常生成
                steps = random.randint(-max_steps, max_steps)  # 隨機步長
                
            # 進行轉速變化
            delta = steps * self.base_step  # 轉速變化量
            new_speed = current_speed + delta  # 新轉速
        else:
            return '生成鄰近解時缺乏前一次風扇轉速，請檢查'     
        # 確保在合理範圍內 (以防萬一)
        new_speed = max(40, min(100, new_speed))  # 限制轉速在40-100之間
        
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
            current_temp = fixed_window_data[-1][1]  # 當前溫度
            past_temp = fixed_window_data[-10][1]  # 過去溫度
            initial_speed = fixed_window_data[-1][5]  # 過去風扇轉速
            error = current_temp - past_temp  # 溫度變化誤差
            print("✅ 數據蒐集完成，開始進行模擬退火最佳化")
        
        # 初始解
        if initial_speed is not None:
            self.previous_fan_speed = initial_speed
        else:
            self.adam.update_duty_cycles(fan_duty=60)
            initial_speed = self.adam.buffer[8]  # 默認轉速
        
        best_speed = initial_speed  # 最佳轉速初始值
        
        # 計算初始解的成本
        initial_predicted_temps = self.predict_temp(initial_speed, fixed_window_data)  # 預測溫度
        initial_cost = self.objective_function(initial_speed, initial_predicted_temps, error, current_temp)  # 計算當前成本
        best_cost = initial_cost  # 最佳成本初始值
        
        # 顯示初始解的預測溫度變化方向
        if initial_predicted_temps is not None and len(initial_predicted_temps) > 0:
            initial_predicted_slope = (initial_predicted_temps[-1] - current_temp) / len(initial_predicted_temps)  # 預測溫度斜率
            direction = "降溫" if initial_predicted_slope < 0 else "升溫"  # 溫度變化方向
            print(f"🌡️ 初始解: 風扇轉速 = {initial_speed}%, 預測溫度變化方向: {direction}, 斜率: {initial_predicted_slope:.4f}")
            # 顯示每個時間步的預測溫度
            print(f"   預測溫度序列: {[f'{temp:.2f}' for temp in initial_predicted_temps]}")
        
        # 模擬退火主循環
        T = self.T_max  # 初始溫度
        while T > self.T_min:
            self.T_current = T  # 保存當前溫度用於生成鄰近解
            
            for _ in range(self.max_iterations):
                # 生成新解
                new_speed = self.generate_neighbor(initial_speed)  # 生成鄰近解
                new_predicted_temps = self.predict_temp(new_speed, fixed_window_data)  # 預測新解的溫度
                new_cost = self.objective_function(new_speed, new_predicted_temps, error, current_temp)  # 計算新解的成本
                
                # 計算成本差異
                delta_cost = new_cost -best_cost  # 成本變化
                
                # 顯示所有解的預測溫度變化方向
                if new_predicted_temps is not None and len(new_predicted_temps) > 0:
                    new_predicted_slope = (new_predicted_temps[-1] - current_temp) / len(new_predicted_temps)  # 預測溫度斜率
                    direction = "降溫" if new_predicted_slope < 0 else "升溫"  # 溫度變化方向
                    print(f"🔍 嘗試解: 風扇轉速 = {new_speed}%, 預測溫度變化方向: {direction}, 斜率: {new_predicted_slope:.4f}, 成本: {new_cost:.2f}")
                    # 顯示每個時間步的預測溫度
                    print(f"   預測溫度序列: {[f'{temp:.2f}' for temp in new_predicted_temps]}")
                
                '''Metropolis準則
                如果新解的成本比當前解更低，則接受新解
                如果新解的成本比當前解更高，則以一定的概率接受新解，這個概率與溫度T和成本差異delta_cost有關
                '''
                accept = delta_cost <= 0 or random.random() < math.exp(-delta_cost / T)  
                if accept:
                    best_speed = new_speed  # 更新當前解
                    best_cost = new_cost  # 更新當前成本
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
        
        print(f"✅ 最佳化完成: 風扇轉速 = {best_speed}%, 最終成本 = {best_cost:.2f}")
        return best_speed, best_cost



# 測試代碼
if __name__ == "__main__":
    optimizer = Revised_SA_Optimizer(adam=None, target_temp=25, P_max=100)
    optimal_fan_speed, optimal_cost = optimizer.optimize()
    optimizer.plot_cost()
    print(f"\nOptimal Fan Speed: {optimal_fan_speed}% with Cost: {optimal_cost:.2f}")
