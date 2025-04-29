import threading
import time
import numpy as np
import joblib
import torch

class SequenceWindowProcessor:
    """
    序列窗口處理器，用於管理時間序列數據的滑動窗口，處理數據標準化和反標準化，
    以及監聽ADAM控制器的數據更新。
    
    該類負責維護一個固定大小的數據緩衝區，用於存儲最近的系統狀態數據，
    並提供數據預處理和後處理功能，為預測模型和MPC控制器提供所需的輸入。
    """
    def __init__(self, window_size=35, adams_controller=None, scaler_path=None, device="cpu"):
        """
        初始化序列窗口處理器
        
        Args:
            window_size (int): 滑動窗口大小，默認為35
            adams_controller: ADAM控制器實例，用於獲取實時數據
            scaler_path (str): 數據標準化器的保存路徑
            device (str): 計算設備，'cpu'或'cuda'
        """
        self.window_size = window_size
        self.adam = adams_controller
        self.device = device
        self.buffer = np.zeros((window_size, 7))  # 直接儲存原始數據
        self.buffer_lock = threading.Lock()

        # 加載 Data Processor (Scaler)
        self.scaler = joblib.load(scaler_path)
        if isinstance(self.scaler, tuple) and len(self.scaler) == 2:
            self.input_scaler, self.output_scaler = self.scaler
        else:
            raise ValueError("載入的 scaler 應為 (input_scaler, output_scaler) 格式，但獲得單一 scaler。")

        self.data_count = 0
        self.adam.data_updated_event.clear()  # 清除初始事件
        self.start_adam_listener()
        

        # 用於溫度變化趨勢追蹤
        self.temp_trend = None  # None: 未知, 1: 上升, -1: 下降, 0: 穩定
        

    def start_adam_listener(self):
        """
        啟動監聽 ADAM 更新事件的執行緒
        
        創建並啟動一個守護線程，用於監聽ADAM控制器的數據更新事件，
        當有新數據時自動更新內部緩衝區。
        """
        thread = threading.Thread(target=self.adam_update_listener, daemon=True)
        thread.start()

    def adam_update_listener(self):
        """
        監聽 ADAM 更新事件，自動更新 buffer
        
        此方法在獨立線程中運行，持續監聽ADAM控制器的數據更新事件，
        當事件被觸發時，調用update_from_adam方法更新數據緩衝區。
        """
        while True:
            self.adam.data_updated_event.wait()  # 等待 ADAM 觸發事件
            self.update_from_adam()
            self.adam.data_updated_event.clear()  # 清除事件

    def update_from_adam(self):
        """
        當 ADAMScontroller 觸發更新時，直接更新 buffer 內的數據
        
        從ADAM控制器獲取最新的系統狀態數據，包括溫度和控制信號，
        並將其添加到滑動窗口緩衝區中，同時更新數據計數器。
        """
        with self.buffer_lock:
            raw_data = np.array([
                self.adam.buffer[0],  # T_GPU
                self.adam.buffer[3],  # T_CDU_out
                self.adam.buffer[2],  # T_CDU_in
                self.adam.buffer[5],  # T_air_in
                self.adam.buffer[6],  # T_air_out
                self.adam.buffer[8],  # fan_duty
                self.adam.buffer[9]   # pump_duty
            ]).reshape(1, -1)
            
            self.buffer = np.roll(self.buffer, -1, axis=0)
            self.buffer[-1, :] = raw_data  # 更新 buffer
            self.data_count += 1

    def get_window_data(self, normalize=False):
        """
        取得時間窗口數據
        
        獲取當前滑動窗口中的所有數據，可選是否進行標準化處理。
        
        Args:
            normalize (bool): 若為 True，則回傳正規化後的數據
            
        Returns:
            numpy.ndarray 或 torch.Tensor: 窗口數據，如果數據不足則返回None
            如果normalize=True，返回標準化後的張量；否則返回原始numpy數組
        """
        with self.buffer_lock:
            if self.data_count < self.window_size:
                print(f"⏳ 當前資料室窗內資料量 {self.data_count}/{self.window_size}，請稍等...")
                return None
            else:
                print(f"✅ 資料室窗內資料量充足，將開始進行預測與最佳化")
            
            data = self.buffer.copy()
            
            if normalize and hasattr(self.input_scaler, "transform"):
                data = torch.tensor(self.input_scaler.transform(data), dtype=torch.float32).unsqueeze(0).to(self.device)
            return data

    def inverse_transform_predictions(self, scaled_predictions, smooth=True):
        """
        反標準化預測數據並可選擇進行平滑處理
        
        將模型輸出的標準化預測結果轉換回原始尺度，並可選擇進行平滑處理
        以減少預測結果中的跳變。
        
        Args:
            scaled_predictions (numpy.ndarray): 標準化後的預測數據
            smooth (bool): 是否進行平滑處理，預設為True
            
        Returns:
            numpy.ndarray: 反標準化（並可能平滑處理）後的預測數據
            
        Raises:
            AttributeError: 如果output_scaler缺少inverse_transform方法
        """
        if hasattr(self.output_scaler, "inverse_transform"):
            inverse_data = self.output_scaler.inverse_transform(scaled_predictions)[:, 0]
            
            # 如果需要平滑處理，則調用平滑函數
            if smooth:
                final_data = self._smooth_predictions(inverse_data)
            else:
                final_data = inverse_data
            return final_data
        else:
            raise AttributeError("output_scaler 缺少 inverse_transform 方法，請檢查 scaler 是否正確載入。")

    def transform_input_data(self,data):
        """
        標準化輸入數據
        
        將原始輸入數據轉換為標準化格式，以便模型處理。
        
        Args:
            data (numpy.ndarray): 原始輸入數據
            
        Returns:
            torch.Tensor: 標準化後的數據張量，已調整為模型所需的維度和設備
            
        Raises:
            AttributeError: 如果input_scaler缺少transform方法
        """
        if hasattr(self.input_scaler, "transform"):
            return torch.tensor(self.input_scaler.transform(data), dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            raise AttributeError("input_scaler 缺少 transform 方法，請檢查 scaler 是否正確載入。")
    
    def _smooth_predictions(self, predictions):
        """
        平滑處理預測溫度序列，特別處理第一個點的跳變問題
        
        通過限制預測序列中第一個點與當前實際溫度的差值，並使用線性插值
        平滑處理前幾個預測點，減少預測結果中的不合理跳變。
        
        Args:
            predictions (numpy.ndarray): 原始預測溫度序列
            
        Returns:
            numpy.ndarray: 平滑處理後的溫度序列
        """
            
        # 獲取當前實際溫度
        current_temp = self.adam.buffer[3]  # T_CDU_out 的位置索引為 3
        
        # 計算預測的第一個點與實際溫度的差值
        first_point_diff = predictions[0] - current_temp
        
        # 判斷溫度變化趨勢
        if abs(first_point_diff) < 0.05:
            # 溫度變化很小，視為穩定
            self.temp_trend = 0
        elif first_point_diff > 0:
            # 溫度上升趨勢
            self.temp_trend = 1
        else:
            # 溫度下降趨勢
            self.temp_trend = -1
            
        # 根據趨勢設定閾值
        if self.temp_trend == 1:
            threshold = 0.1  # 上升趨勢閾值
        elif self.temp_trend == -1:
            threshold = 0.1  # 下降趨勢閾值
        else:
            threshold = 0.05  # 穩定趨勢閾值
            
        # 處理第一個點的跳變
        smoothed_predictions = predictions.copy()
        if abs(first_point_diff) > threshold:
            print(f"⚠️ 檢測到溫度預測跳變: {first_point_diff:.3f}°C，進行平滑處理")
            
            # 計算限制後的第一個點
            if first_point_diff > 0:
                limited_first_point = current_temp + threshold
            else:
                limited_first_point = current_temp - threshold
                
            # 使用線性插值平滑處理前3個點
            smooth_range = min(3, len(predictions))
            original_diff = predictions[0] - limited_first_point
            
            for i in range(smooth_range):
                # 計算平滑係數，從0到1
                smooth_factor = (i / (smooth_range - 1)) if smooth_range > 1 else 1
                
                # 線性插值調整
                adjustment = original_diff * smooth_factor
                smoothed_predictions[i] = limited_first_point + adjustment
                
            print(f"📊 平滑前第一點: {predictions[0]:.3f}°C → 平滑後: {smoothed_predictions[0]:.3f}°C")
        
        return smoothed_predictions