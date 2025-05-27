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

class HC_Optimizer:
    def __init__(self, adam, window_size=25, P_max=100, target_temp=25,
                 model_path='/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/iTransformer/seq40_pred8_dmodel16_dff16_nheads8_elayers1_dropout0.01_lr0.0001_batchsize512_epochs140/best_model.pth',
                 scaler_path='/home/inventec/Desktop/2KWCDU_修改版本/code_manage/Predict_Model/iTransformer/seq40_pred8_dmodel16_dff16_nheads8_elayers1_dropout0.01_lr0.0001_batchsize512_epochs140/scalers.jlib',
                 figure_path='/home/inventec/Desktop/2KWCDU_修改版本/data_manage/control_data/Fan_MPC_FHO_data'):
        """初始化爬山演算法(HC)風扇轉速最佳化器。
        
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
        self.back_step = 15  # 回推步長
        
        # 爬山演算法參數
        self.max_iterations = 15  # 最大迭代次數
        self.base_step = 5  # 基本步長
        
        # 目標函數參數
        self.w_temp = 1  # 溫度控制項權重
        self.w_speed = 0  # 速度平滑項權重
        self.error_band = 0.1  # 溫度控制項誤差帶
        
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
        data_copy[self.back_step:, 4] = data_copy[:-self.back_step, 4]  # 將序列向左平移self.back_step步
        data_copy[-self.back_step:, 4] = fan_speed  # 用新的風扇轉速填充後self.back_step個時間步
        input_tensor = self.data_processor.transform_input_data(data_copy)  # 轉換輸入數據為張量

        if input_tensor is not None:
            with torch.no_grad():
                scaled_predictions = self.model(input_tensor)[0].cpu().numpy()  # 獲取縮放後的預測結果
                predicted_temps = self.data_processor.inverse_transform_predictions(scaled_predictions, smooth=False)  # 反轉縮放

                return predicted_temps
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
                temp_diff = (abs(i - self.target_temp)*6)**2  # 溫度差的平方
                temp_error += temp_diff
            else:
                temp_error += 0

        # 總成本
        total_cost = self.w_temp * temp_error   # 總成本等於溫度誤差
        
        return total_cost

    def generate_neighbor(self, current_speed):
        """產生鄰近解。
        
        Args:
            current_speed (float): 目前的風扇轉速。
            
        Returns:
            int: 新的風扇轉速。
        """
        if self.previous_fan_speed is not None:
            # 在爬山演算法中，採用固定步長或隨機步長進行探索
            steps = random.choice([-1, 1])  # 隨機選擇向上或向下的方向
            delta = steps * self.base_step  # 應用基本步長
            new_speed = current_speed + delta  # 計算新的風扇轉速
        else:
            # 首次運行
            new_speed = random.uniform(40, 100)  # 在合理範圍內隨機選擇
            new_speed = round(new_speed / self.base_step) * self.base_step  # 四捨五入到步長的倍數
            
        # 限制範圍
        new_speed = max(40, min(100, new_speed))  # 確保在允許範圍內
        
        return int(new_speed)  # 返回整數轉速

    def optimize(self):
        """執行爬山演算法最佳化。
        
        Returns:
            tuple: 包含(最佳風扇轉速, 最佳成本)的元組，若數據收集失敗則返回(None, None)。
        """
        fixed_window_data = self.data_processor.get_window_data(normalize=False)  # 獲取窗口數據
        if fixed_window_data is None:
            return None, None

        else:
            current_temp = fixed_window_data[-1][1]  # 當前溫度
            # 移除對未定義變量的引用，直接將誤差設為0
            error = 0  # 初始化誤差為0
            print("✅ 數據蒐集完成，開始進行爬山演算法最佳化")
        
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
        current_cost = self.objective_function(current_speed, predicted_temps, error, current_temp)  # 計算當前成本
        best_cost = current_cost  # 最佳成本初始值
        
        # 顯示初始解的預測溫度變化方向
        if predicted_temps is not None and len(predicted_temps) > 0:
            predicted_slope = (predicted_temps[-1] - current_temp) / len(predicted_temps)  # 預測溫度斜率
            direction = "降溫" if predicted_slope < 0 else "升溫"  # 溫度變化方向
            print(f"🌡️ 初始解: 風扇轉速 = {current_speed}%, 預測溫度變化方向: {direction}, 斜率: {predicted_slope:.4f}")
            # 顯示每個時間步的預測溫度
            print(f"   預測溫度序列: {[f'{temp:.2f}' for temp in predicted_temps]}")
        
        # 爬山演算法主循環
        for iteration in range(self.max_iterations):
            print(f"\n⏱️ 迭代 {iteration+1}/{self.max_iterations}")
            
            # 產生多個鄰居解進行探索
            neighbors = []
            for _ in range(4):  # 嘗試產生4個鄰居解
                neighbor_speed = self.generate_neighbor(current_speed)
                neighbors.append(neighbor_speed)
            
            # 確保鄰居解不重複
            neighbors = list(set(neighbors))
            
            # 評估所有鄰居解
            best_neighbor = None
            best_neighbor_cost = float('inf')
            
            for neighbor_speed in neighbors:
                predicted_temps = self.predict_temp(neighbor_speed, fixed_window_data)
                neighbor_cost = self.objective_function(neighbor_speed, predicted_temps, error, current_temp)
                
                # 顯示所有解的預測溫度變化方向
                if predicted_temps is not None and len(predicted_temps) > 0:
                    predicted_slope = (predicted_temps[-1] - current_temp) / len(predicted_temps)
                    direction = "降溫" if predicted_slope < 0 else "升溫"
                    print(f"🔍 嘗試解: 風扇轉速 = {neighbor_speed}%, 預測溫度變化方向: {direction}, 斜率: {predicted_slope:.4f}, 成本: {neighbor_cost:.2f}")
                    # 顯示每個時間步的預測溫度
                    print(f"   預測溫度序列: {[f'{temp:.2f}' for temp in predicted_temps]}")
                
                # 更新最佳鄰居解
                if neighbor_cost < best_neighbor_cost:
                    best_neighbor = neighbor_speed
                    best_neighbor_cost = neighbor_cost
            
            # 如果找到更好的解，則更新當前解
            if best_neighbor_cost < current_cost:
                current_speed = best_neighbor
                current_cost = best_neighbor_cost
                print(f"✅ 接受更好的解: 風扇轉速 = {current_speed}%, 成本 = {current_cost:.2f}")
                
                # 更新全局最佳解
                if current_cost < best_cost:
                    best_speed = current_speed
                    best_cost = current_cost
                    print(f"🌟 發現更好的解: 風扇轉速 = {best_speed}%, 成本 = {best_cost:.2f}")
            else:
                # 如果沒有找到更好的解，則停止迭代（爬山演算法特性）
                print(f"⚠️ 未找到更好的解，可能已達到局部最優點")
                break
        
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
        
        # 應用最大轉速變化限制
        current_system_speed = self.adam.buffer[8] if self.adam.buffer[8] is not None else 60
        max_change = self.max_speed_change  # 使用已定義的最大變化率
        
        # 計算允許的轉速範圍
        min_allowed_speed = max(40, current_system_speed - max_change)
        max_allowed_speed = min(100, current_system_speed + max_change)
        
        # 限制最佳風扇轉速變化
        if best_speed < min_allowed_speed:
            best_speed = int(min_allowed_speed)
            print(f"⚠️ 轉速變化過大，限制為下限: {best_speed}%")
        elif best_speed > max_allowed_speed:
            best_speed = int(max_allowed_speed)
            print(f"⚠️ 轉速變化過大，限制為上限: {best_speed}%")
        
        print(f"✅ 最佳化完成: 風扇轉速 = {best_speed}%, 最終成本 = {best_cost:.2f}")
        return best_speed, best_cost

    def plot_cost(self):
        """繪製成本歷史圖表"""
        if len(self.cost_history) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(self.cost_history)), self.cost_history, marker='o')
            plt.title('爬山演算法最佳化成本歷史')
            plt.xlabel('迭代次數')
            plt.ylabel('成本值')
            plt.grid(True)
            
            # 確保圖表保存目錄存在
            os.makedirs(self.figure_path, exist_ok=True)
            
            # 生成時間戳記
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # 保存圖表
            plt.savefig(f"{self.figure_path}/hc_cost_history_{timestamp}.png")
            plt.close()
            print(f"✅ 成本歷史圖表已保存至: {self.figure_path}/hc_cost_history_{timestamp}.png")


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
        seq_len=40,
        pred_len=8
    )
    
    # 創建優化器實例
    optimizer = HC_Optimizer(
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
