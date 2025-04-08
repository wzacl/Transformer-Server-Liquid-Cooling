# /usr/bin/python3
# 二分搜索演算法 (Binary Search Optimization)
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




class BinarySearchOptimizer:
    def __init__(self, adam, max_iter=8, P_max=100, target_temp=25,
                 window_size=35, min_speed=30, max_speed=100, tolerance=1,
                 model_path='/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400/2KWCDU_Transformer_model.pth',
                 scaler_path='/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400/1.5_1KWscalers.jlib',
                 figure_path='/home/inventec/Desktop/2KWCDU_修改版本/data_manage/control_data/Fan_MPC_BSO_data/Figure'):
        """
        二分搜索演算法 (BSO) 初始化
        :param adam: ADAM 控制器
        :param max_iter: 最大迭代次數
        :param P_max: 風扇最大功耗 (W)
        :param target_temp: 目標 CDU 出水溫度 (°C)
        :param window_size: 序列窗口大小
        :param min_speed: 最小風扇轉速 (%)
        :param max_speed: 最大風扇轉速 (%)
        :param tolerance: 搜索終止容差 (%)
        """
        # 搜索參數
        self.max_iter = max_iter
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.tolerance = tolerance
        
        # 優化參數
        self.P_max = P_max
        self.target_temp = target_temp
        self.best_solution = None
        self.best_cost = float('inf')
        self.cost_history = []
        self.speed_history = []  # 記錄每次搜索的轉速區間
        
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
        """預測溫度，完全參考 FHO 的實現方式"""
        try:
            # 檢查輸入參數
            if fan_speed is None or data is None:
                print("❌ 預測輸入參數不得為None")
                return None
            
            # 確保fan_speed是數字
            try:
                fan_speed = float(fan_speed)
            except (TypeError, ValueError):
                print(f"❌ 風扇轉速必須是數字，收到: {type(fan_speed)}")
                return None
            
            # 創建數據副本
            data_copy = data.copy()
            # 修改風扇轉速
            data_copy[-1][5] = fan_speed
            
            # 準備輸入數據
            input_tensor = self.data_processor.transform_input_data(data_copy)

            if input_tensor is not None:
                with torch.no_grad():
                    # 預測8步
                    scaled_predictions = self.model(input_tensor, num_steps=8)[0].cpu().numpy()
                    
                    # 先進行不帶平滑的預測
                    predicted_temps = self.data_processor.inverse_transform_predictions(scaled_predictions, smooth=False)
                    
                    # 再進行帶平滑的預測
                    smoothed_temps = self.data_processor.inverse_transform_predictions(scaled_predictions, smooth=True)
                    
                    # 計算平滑處理的差異
                    diff = np.max(np.abs(smoothed_temps - predicted_temps))
                    
                    # 輸出調試信息
                    print(f"🔄 溫度預測結果 (風扇轉速: {fan_speed}%)")
                    print(f"   原始預測: {predicted_temps[:3]}...")
                    print(f"   平滑後預測: {smoothed_temps[:3]}...")
                    print(f"   調整量: {diff:.3f}°C")
                    
                    return smoothed_temps
            return None
            
        except Exception as e:
            print(f"❌ 預測過程發生錯誤: {str(e)}")
            print(f"輸入數據類型: {type(data)}")
            if hasattr(data, 'shape'):
                print(f"輸入數據形狀: {data.shape}")
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
        
        # 風扇功耗 - 使用立方關係計算功耗
        power_fan = (fan_speed / 100) ** 3 * self.P_max  
        
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
                if speed_change > 30:  # 只有極端變化才懲罰
                    speed_change_penalty = (speed_change - 30) ** 2 * 0.01
            # 如果沒有過熱風險，使用正常邏輯
            else:
                if self.previous_fan_speed < fan_speed:
                    if speed_change > 20:
                        speed_change_penalty = (speed_change - 20) ** 2 * 0.05
                else:
                    if speed_change > 10:
                        speed_change_penalty = (speed_change - 10) ** 2 * 0.1
        
        # 溫度控制與能耗平衡
        # 溫度過高時優先考慮降溫，溫度達標時優先考慮節能
        avg_temp = sum(predicted_temps) / len(predicted_temps)
        if avg_temp > self.target_temp:
            # 溫度過高，優先考慮溫度控制
            temp_weight = 3.0
            power_weight = 0.01
        else:
            # 溫度達標，加大能耗權重
            temp_diff_ratio = max(0, 1 - (self.target_temp - avg_temp) / self.target_temp)
            temp_weight = 1.0 + temp_diff_ratio * 2.0  # 溫度越接近目標值，權重越高
            power_weight = 0.05 + (1 - temp_diff_ratio) * 0.1  # 溫度越低，能耗權重越高
        
        # 總成本
        total_cost = (total_temp_error * temp_weight + 
                      power_fan * power_weight + 
                      speed_change_penalty +
                      overheat_penalty)
        
        return total_cost

    def optimize(self):
        """執行二分搜索最佳化風扇轉速"""
        # 先從 sequence_window 取得當前時間點的固定數據
        fixed_window_data = self.data_processor.get_window_data(normalize=False)

        if fixed_window_data is None:
            return None, None
        else:
            print("✅ 數據蒐集完成，開始進行二分搜索最佳化")
            
            # 初始化搜索範圍
            left = self.min_speed
            right = self.max_speed
            
            # 獲取當前溫度
            current_temp = fixed_window_data[-1][1]  # CDU出水溫度
            
            # 啟發式初始化：根據當前溫度與目標溫度的差異調整初始搜索範圍
            if current_temp is not None:
                if current_temp > self.target_temp + 1:
                    # 溫度明顯過高，從高轉速開始搜索
                    left = max(60, self.min_speed)
                elif current_temp < self.target_temp - 1:
                    # 溫度明顯過低，從低轉速開始搜索
                    right = min(70, self.max_speed)
            
            # 記錄初始搜索區間
            self.speed_history.append((left, right))
            
            # 如果有上次風扇轉速，記錄下來
            if self.previous_fan_speed is not None:
                # 確保搜索範圍包含上次的風扇轉速(除非溫度差異很大)
                temp_diff = abs(current_temp - self.target_temp)
                if temp_diff < 2.0:
                    left = min(left, max(self.min_speed, self.previous_fan_speed - 20))
                    right = max(right, min(self.max_speed, self.previous_fan_speed + 20))
            
            # 二分搜索迭代
            iteration = 0
            best_fan_speed = None
            best_cost = float('inf')
            evaluated_speeds = {}  # 記錄已評估的轉速，避免重複計算
            
            while iteration < self.max_iter and (right - left) > self.tolerance:
                # 檢查中點和四分點
                mid = (left + right) // 2
                quarter1 = (left + mid) // 2
                quarter3 = (mid + right) // 2
                
                # 將所有候選點四捨五入到10的倍數
                mid = round(mid / 10) * 10
                quarter1 = round(quarter1 / 10) * 10
                quarter3 = round(quarter3 / 10) * 10
                
                # 確保所有候選點不重複且在範圍內
                candidates = []
                for speed in [left, quarter1, mid, quarter3, right]:
                    speed = min(max(speed, self.min_speed), self.max_speed)
                    speed = round(speed / 10) * 10  # 四捨五入到10的倍數
                    if speed not in candidates:
                        candidates.append(speed)
                
                print(f"🔍 迭代 {iteration+1} | 搜索範圍: [{left}%, {right}%] | 候選轉速: {candidates}")
                
                # 評估候選轉速
                costs = {}
                for speed in candidates:
                    # 檢查是否已評估過此轉速
                    if speed in evaluated_speeds:
                        cost = evaluated_speeds[speed]
                        print(f"📊 評估風扇轉速 {speed}%: 成本 = {cost:.2f} (快取)")
                    else:
                        # 預測溫度並計算成本
                        predicted_temps = self.predict_temp(speed, fixed_window_data)
                        
                        if predicted_temps is not None:
                            cost = self.objective_function(speed, predicted_temps)
                            evaluated_speeds[speed] = cost
                            print(f"📊 評估風扇轉速 {speed}%: 成本 = {cost:.2f}")
                        else:
                            continue
                    
                    costs[speed] = cost
                    
                    # 更新全局最佳解
                    if cost < best_cost:
                        best_cost = cost
                        best_fan_speed = speed
                
                if not costs:
                    print("❌ 無有效評估數據，退出搜索")
                    break
                
                # 找出成本最低的轉速區間
                sorted_speeds = sorted(costs.keys())
                min_cost_speed = min(costs, key=costs.get)
                
                # 縮小搜索範圍到成本最低轉速附近
                if min_cost_speed == sorted_speeds[0]:
                    # 最低點在最左側，向左擴展搜索範圍
                    right = sorted_speeds[1]
                    left = max(self.min_speed, min_cost_speed - (right - min_cost_speed))
                elif min_cost_speed == sorted_speeds[-1]:
                    # 最低點在最右側，向右擴展搜索範圍
                    left = sorted_speeds[-2]
                    right = min(self.max_speed, min_cost_speed + (min_cost_speed - left))
                else:
                    # 最低點在中間
                    idx = sorted_speeds.index(min_cost_speed)
                    left = sorted_speeds[idx-1]
                    right = sorted_speeds[idx+1]
                
                # 記錄本次迭代的搜索區間
                self.speed_history.append((left, right))
                
                # 記錄最佳成本
                self.cost_history.append(best_cost)
                
                # 增加迭代計數
                iteration += 1
            
            # 如果還有額外的迭代次數，使用漸進式步長搜索更精確的最佳解
            while iteration < self.max_iter and best_fan_speed is not None:
                # 在最佳解附近進行細緻搜索
                step_size = max(10, (right - left) // 3)
                candidates = [best_fan_speed]
                
                # 添加步長範圍內的候選轉速
                for step in [-step_size, step_size]:
                    candidate = best_fan_speed + step
                    candidate = min(max(candidate, self.min_speed), self.max_speed)
                    candidate = round(candidate / 10) * 10  # 四捨五入到10的倍數
                    if candidate not in candidates:
                        candidates.append(candidate)
                
                print(f"🔍 迭代 {iteration+1} | 細化搜索 | 候選轉速: {candidates}")
                
                # 評估候選轉速
                costs = {}
                for speed in candidates:
                    # 檢查是否已評估過此轉速
                    if speed in evaluated_speeds:
                        cost = evaluated_speeds[speed]
                        print(f"📊 評估風扇轉速 {speed}%: 成本 = {cost:.2f} (快取)")
                    else:
                        # 預測溫度並計算成本
                        predicted_temps = self.predict_temp(speed, fixed_window_data)
                        
                        if predicted_temps is not None:
                            cost = self.objective_function(speed, predicted_temps)
                            evaluated_speeds[speed] = cost
                            print(f"📊 評估風扇轉速 {speed}%: 成本 = {cost:.2f}")
                        else:
                            continue
                    
                    costs[speed] = cost
                    
                    # 更新全局最佳解
                    if cost < best_cost:
                        best_cost = cost
                        best_fan_speed = speed
                
                # 記錄最佳成本
                self.cost_history.append(best_cost)
                
                # 增加迭代計數
                iteration += 1
                
                # 如果沒有更好的解，提前結束
                if best_fan_speed == candidates[0] and len(candidates) > 1:
                    print("🎯 已找到局部最佳解，提前結束搜索")
                    break
            
            # 轉速變化平滑處理
            if best_fan_speed is None:
                print("⚠️ 搜索過程未找到有效的風扇轉速")
                return None, None
        
            final_fan_speed = best_fan_speed
            if self.previous_fan_speed is not None:
                # 限制單次變化幅度
                speed_change = abs(final_fan_speed - self.previous_fan_speed)
                max_change = 20  # 最大變化幅度
                
                if speed_change > max_change:
                    print(f"⚠️ 轉速變化過大 ({speed_change}%)，限制變化幅度至 {max_change}%")
                    if final_fan_speed > self.previous_fan_speed:
                        final_fan_speed = self.previous_fan_speed + max_change
                    else:
                        final_fan_speed = self.previous_fan_speed - max_change
            
            # 更新上一次的風扇轉速
            self.previous_fan_speed = final_fan_speed
            
            # 更新全局最佳解
            if best_cost < self.best_cost:
                self.best_cost = best_cost
                self.best_solution = best_fan_speed
            
            print(f"✅ 二分搜索完成 | 最佳風扇轉速: {final_fan_speed}% | 成本: {best_cost:.2f}")
            
            # 返回結果前確保值是數字型別
            try:
                return int(final_fan_speed), float(best_cost)
            except (TypeError, ValueError):
                print("❌ 風扇轉速轉換失敗，返回None")
                return None, None

    def plot_cost(self):
        """ 繪製成本收斂圖 """
        plt.figure(figsize=(12, 10))
        
        # 繪製成本收斂子圖
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.cost_history)), self.cost_history, 'b-o', label="成本")
        plt.xlabel("迭代次數")
        plt.ylabel("目標函數成本")
        plt.title("二分搜索最佳化 - 成本收斂圖")
        plt.grid(True)
        plt.legend()
        
        # 繪製搜索區間變化子圖
        if self.speed_history:
            plt.subplot(2, 1, 2)
            iterations = range(len(self.speed_history))
            lower_bounds = [bounds[0] for bounds in self.speed_history]
            upper_bounds = [bounds[1] for bounds in self.speed_history]
            
            plt.plot(iterations, lower_bounds, 'g-o', label="下限")
            plt.plot(iterations, upper_bounds, 'r-o', label="上限")
            if self.previous_fan_speed is not None:
                plt.axhline(y=self.previous_fan_speed, color='purple', linestyle='--', label="最終轉速")
            
            plt.fill_between(iterations, lower_bounds, upper_bounds, alpha=0.2, color='blue')
            plt.xlabel("迭代次數")
            plt.ylabel("風扇轉速範圍 (%)")
            plt.title("搜索區間變化")
            plt.grid(True)
            plt.legend()
        
        # 調整佈局並保存
        plt.tight_layout()
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        save_path = os.path.join(self.figure_path, f"binary_search_{timestamp}.png")
        plt.savefig(save_path)
        print(f"🖼️ 搜索過程分析圖已保存至: {save_path}")
        
        plt.show()

# 使用二分搜索演算法來最佳化風扇轉速
if __name__ == "__main__":
    optimizer = BinarySearchOptimizer(adam=None, target_temp=25, P_max=100, max_iter=8)
    optimal_fan_speed, optimal_cost = optimizer.optimize()
    optimizer.plot_cost()
    print(f"\n最佳風扇轉速: {optimal_fan_speed}% | 成本: {optimal_cost:.2f}")

    if optimal_fan_speed is not None:
        try:
            fan1.set_all_duty_cycle(optimal_fan_speed)
            fan2.set_all_duty_cycle(optimal_fan_speed)
            adam.update_duty_cycles(fan_duty=optimal_fan_speed)
            print(f"✅ 風扇優化完成 | 最佳風扇轉速: {optimal_fan_speed}% | 成本: {optimal_cost:.2f}")
        except (TypeError, ValueError) as e:
            print(f"❌ 風扇轉速設定失敗: {e}")
    else:
        print("❌ 數據蒐集中，等待數據蒐集完成...")
