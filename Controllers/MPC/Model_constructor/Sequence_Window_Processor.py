import threading
import time
import numpy as np
import joblib
import torch

class SequenceWindowProcessor:
    def __init__(self, window_size=30, adams_controller=None, scaler_path=None, device="cpu"):
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

    def start_adam_listener(self):
        """
        啟動監聽 ADAM 更新事件的執行緒
        """
        thread = threading.Thread(target=self.adam_update_listener, daemon=True)
        thread.start()

    def adam_update_listener(self):
        """
        監聽 ADAM 更新事件，自動更新 buffer
        """
        while True:
            self.adam.data_updated_event.wait()  # 等待 ADAM 觸發事件
            self.update_from_adam()
            self.adam.data_updated_event.clear()  # 清除事件

    def update_from_adam(self):
        """
        當 ADAMScontroller 觸發更新時，直接更新 buffer 內的數據
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
        :param normalize: 若為 True，則回傳正規化後的數據
        :return: numpy array (window_size, 8)
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

    def inverse_transform_predictions(self, scaled_predictions):
        """
        反標準化預測數據
        """
        if len(scaled_predictions.shape) == 1:
            scaled_predictions = scaled_predictions.reshape(-1, 1)

        if hasattr(self.output_scaler, "inverse_transform"):
            return self.output_scaler.inverse_transform(scaled_predictions)[:, 0]
        else:
            raise AttributeError("output_scaler 缺少 inverse_transform 方法，請檢查 scaler 是否正確載入。")

    def transform_input_data(self,data):
        if hasattr(self.input_scaler, "transform"):
            return torch.tensor(self.input_scaler.transform(data), dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            raise AttributeError("input_scaler 缺少 transform 方法，請檢查 scaler 是否正確載入。")


