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
import math
import os
import csv




class ProgressiveFanOptimizer:
    def __init__(self, adam, window_size=35, P_max=100, target_temp=25,
                 base_step_size=10, tolerance=5, stability_factor=0.1,
                 decision_history_size=2,
                 model_path='/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400/2KWCDU_Transformer_model.pth',
                 scaler_path='/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400/1.5_1KWscalers.jlib',
                 figure_path='/home/inventec/Desktop/2KWCDU_修改版本/data_manage/control_data/Fan_MPC_FHO_data'):
        """
        漸進式風扇轉速最佳化器初始化
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
        
        # 漸進式調整參數
        self.base_step_size = base_step_size
        self.tolerance = tolerance 
        self.stability_factor = stability_factor
        self.decision_history = []
        self.required_consistent_decisions = decision_history_size
        
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
        
        # 添加平滑處理統計和記錄
        self.smoothing_stats = {
            'total_predictions': 0,
            'smoothed_predictions': 0,
            'total_smoothing_magnitude': 0.0,
            'max_smoothing': 0.0,
            'history': []  # 保存每次平滑的詳細信息
        }
        
        # 創建平滑記錄檔案
        self.smoothing_log_path = os.path.join('/home/inventec/Desktop/2KWCDU_修改版本/data_manage/control_data/Fan_MPC_FHO_data', 'smoothing_analysis.csv')
        if not os.path.exists(self.smoothing_log_path):
            with open(self.smoothing_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['時間戳', '風扇轉速', '原始預測', '平滑後預測', '差值', '趨勢'])

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
                
                # 使用平滑處理函數處理預測結果
                smoothed_temps = self.data_processor.smooth_predictions(predicted_temps)
                
                # 計算平滑處理的差異
                diff = np.max(np.abs(smoothed_temps - predicted_temps))
                
                # 更新統計數據
                self.smoothing_stats['total_predictions'] += 1
                if diff > 0.05:
                    self.smoothing_stats['smoothed_predictions'] += 1
                    self.smoothing_stats['total_smoothing_magnitude'] += diff
                    self.smoothing_stats['max_smoothing'] = max(self.smoothing_stats['max_smoothing'], diff)
                    
                    # 記錄平滑詳情
                    trend = "上升" if self.data_processor.temp_trend == 1 else "下降" if self.data_processor.temp_trend == -1 else "穩定"
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    
                    # 添加到歷史記錄
                    self.smoothing_stats['history'].append({
                        'timestamp': timestamp,
                        'fan_speed': fan_speed,
                        'original': predicted_temps[0],
                        'smoothed': smoothed_temps[0],
                        'diff': diff,
                        'trend': trend
                    })
                    
                    # 寫入CSV日誌
                    try:
                        with open(self.smoothing_log_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                timestamp, 
                                fan_speed, 
                                predicted_temps[0], 
                                smoothed_temps[0], 
                                diff, 
                                trend
                            ])
                    except Exception as e:
                        print(f"❌ 無法寫入平滑處理記錄: {e}")
                    
                    # 在日誌中記錄平滑前後的差異
                    print(f"🔄 平滑處理調整了預測溫度 (風扇轉速: {fan_speed}%)")
                    print(f"   原始預測: {predicted_temps[:3]}...")
                    print(f"   平滑後預測: {smoothed_temps[:3]}...")
                    print(f"   溫度趨勢: {trend}, 調整量: {diff:.3f}°C")
                
                return smoothed_temps
        else:
            return None
            
    def print_smoothing_statistics(self):
        """
        打印平滑處理統計數據
        """
        if self.smoothing_stats['total_predictions'] == 0:
            print("尚無預測資料")
            return
            
        smoothing_rate = (self.smoothing_stats['smoothed_predictions'] / 
                          self.smoothing_stats['total_predictions'] * 100)
        
        avg_magnitude = 0
        if self.smoothing_stats['smoothed_predictions'] > 0:
            avg_magnitude = (self.smoothing_stats['total_smoothing_magnitude'] / 
                            self.smoothing_stats['smoothed_predictions'])
        
        print("\n📊 溫度預測平滑處理統計")
        print(f"總預測次數: {self.smoothing_stats['total_predictions']}")
        print(f"平滑處理次數: {self.smoothing_stats['smoothed_predictions']} ({smoothing_rate:.1f}%)")
        print(f"平均調整量: {avg_magnitude:.3f}°C")
        print(f"最大調整量: {self.smoothing_stats['max_smoothing']:.3f}°C")
        print(f"詳細記錄已保存至: {self.smoothing_log_path}")

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
                error = temp_diff ** (1/2) * time_weight * accuracy_factor
            
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

    def dynamic_step_size(self, current_temp):
        """根據溫度差異動態調整步長"""
        temp_diff = abs(current_temp - self.target_temp)
        
        if temp_diff > 2.0:  # 差異大
            return min(20, self.base_step_size * 2)
        elif temp_diff > 1.0:  # 差異中等
            return self.base_step_size
        else:  # 差異小
            return max(5, self.base_step_size / 2)
    
    def generate_candidate_speeds(self, current_temp, current_fan_speed):
        """生成候選轉速"""
        step = self.dynamic_step_size(current_temp)
        
        candidates = [current_fan_speed]  # 當前轉速
        
        # 根據溫度差異生成非對稱步長
        if current_temp > self.target_temp:
            # 過熱情況，上調步長大，下調步長小
            up_step = step * 1.5
            down_step = step * 0.5
        else:
            # 過冷情況，下調步長大，上調步長小
            up_step = step * 0.5
            down_step = step * 1.5
        
        # 添加上調轉速
        up_speed = min(100, current_fan_speed + up_step)
        if up_speed != current_fan_speed:
            candidates.append(up_speed)
        
        # 添加下調轉速
        down_speed = max(30, current_fan_speed - down_step)
        if down_speed != current_fan_speed:
            candidates.append(down_speed)
        
        # 四捨五入到10%的單位
        candidates = [round(speed / 10) * 10 for speed in candidates]
        # 去除重複值
        candidates = list(set(candidates))
        
        return candidates

    def optimize(self):
        """執行漸進式最佳化搜索風扇轉速"""
        # 先從 sequence_window 取得當前時間點的固定數據
        fixed_window_data = self.data_processor.get_window_data(normalize=False)

        if fixed_window_data is None:
            return None, None
        else:
            print("✅ 數據蒐集完成，開始進行漸進式最佳化")
            
            # 獲取當前溫度
            current_temp = fixed_window_data[-1][4]  # CDU出水溫度
            
            # 如果沒有先前的風扇轉速，則初始化一個合理值
            if self.previous_fan_speed is None:
                if current_temp is not None and current_temp > self.target_temp:
                    # 溫度越高，初始風扇轉速越高
                    temp_diff = current_temp - self.target_temp
                    self.previous_fan_speed = min(100, max(60, 60 + temp_diff * 10))
                else:
                    # 默認初始轉速
                    self.previous_fan_speed = 50
                
                # 四捨五入到10%單位
                self.previous_fan_speed = round(self.previous_fan_speed / 10) * 10
            
            # 生成候選轉速
            candidates = self.generate_candidate_speeds(current_temp, self.previous_fan_speed)
            print(f"🔍 生成候選轉速: {candidates}")
                
            # 評估每個候選轉速
            best_speed = None
            min_cost = float('inf')
            costs = []
            
            for speed in candidates:
                # 使用預測模型評估
                predicted_temps = self.predict_temp(speed, fixed_window_data)
                
                if predicted_temps is not None:
                    # 計算成本
                    cost = self.objective_function(speed, predicted_temps)
                    costs.append(cost)
                    
                    print(f"📊 評估風扇轉速 {speed}%: 成本 = {cost:.2f}")
                    
                    # 更新最佳解
                    if cost < min_cost:
                        min_cost = cost
                        best_speed = speed
            
            if not costs:
                print("❌ 無有效數據，無法優化")
                return self.previous_fan_speed, float('inf')
                
            # 決策平滑處理
            self.decision_history.append(best_speed)
            if len(self.decision_history) > self.required_consistent_decisions:
                self.decision_history.pop(0)
            
            # 只有連續多次相同決策才真正採納
            if len(set(self.decision_history)) == 1 and len(self.decision_history) >= self.required_consistent_decisions:
                final_speed = best_speed
                print(f"✅ 採納新的風扇轉速: {final_speed}% (連續{self.required_consistent_decisions}次相同決策)")
            else:
                final_speed = self.previous_fan_speed
                print(f"⏳ 保持當前風扇轉速: {final_speed}% (等待決策確認)")
            
            # 記錄歷史成本
            self.cost_history.append(min_cost)
            
            # 更新全局最佳解
            if min_cost < self.best_cost:
                self.best_cost = min_cost
                self.best_solution = best_speed
            
            # 更新上一次的風扇轉速
            self.previous_fan_speed = final_speed
            
            # 在完成優化後輸出簡短的平滑處理統計
            if self.smoothing_stats['smoothed_predictions'] > 0:
                smoothing_rate = (self.smoothing_stats['smoothed_predictions'] / 
                                  self.smoothing_stats['total_predictions'] * 100)
                avg_magnitude = (self.smoothing_stats['total_smoothing_magnitude'] / 
                                self.smoothing_stats['smoothed_predictions'])
                                
                print("\n📊 溫度預測平滑處理簡報")
                print(f"預測平滑比例: {smoothing_rate:.1f}% (共{self.smoothing_stats['smoothed_predictions']}次)")
                print(f"平均調整量: {avg_magnitude:.3f}°C, 最大調整: {self.smoothing_stats['max_smoothing']:.3f}°C")
            
            # 返回結果
            return final_speed, min_cost

    def plot_cost(self):
        """ 繪製成本收斂圖 """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.cost_history)), self.cost_history, 'b-o', label="成本")
        plt.xlabel("迭代次數")
        plt.ylabel("目標函數成本")
        plt.title("漸進式風扇轉速最佳化 - 成本收斂圖")
        plt.grid(True)
        plt.legend()
        
        # 保存圖像
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        save_path = os.path.join(self.figure_path, f"progressive_cost_{timestamp}.png")
        plt.savefig(save_path)
        print(f"🖼️ 成本收斂圖已保存至: {save_path}")
        
        plt.show()





# 使用漸進式優化器來最佳化風扇轉速
if __name__ == "__main__":
    optimizer = ProgressiveFanOptimizer(adam=None, target_temp=25, P_max=100)
    optimal_fan_speed, optimal_cost = optimizer.optimize()
    optimizer.plot_cost()
    print(f"\nOptimal Fan Speed: {optimal_fan_speed}% with Cost: {optimal_cost:.2f}")
