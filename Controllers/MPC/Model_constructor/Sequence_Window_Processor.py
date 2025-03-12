import threading
import time
import numpy as np
import joblib
import torch


class SequenceWindowProcessor:
    def __init__(self, window_size=20, adams_controller=None, scaler_path=None, device="cpu"):
        """
        :param window_size: 時序窗口大小
        :param adams_controller: ADAMScontroller 物件
        :param scaler_path: Scaler 檔案 (必須是 (input_scaler, output_scaler) 格式)
        :param device: PyTorch 設備 (CPU/GPU)
        """
        self.window_size = window_size
        self.adam = adams_controller
        self.device = device
        self.buffer = np.zeros((window_size, 7))  # 滑動窗口緩衝區，存最近 20 筆數據
        self.buffer_lock = threading.Lock()
        #self.data_updated_event = self.adam.data_updated_event  # ✅ 取得 ADAMS 事件

        # ✅ 加載 Data Processor (Scaler)
        self.scaler = joblib.load(scaler_path)
        if isinstance(self.scaler, tuple) and len(self.scaler) == 2:
            self.input_scaler, self.output_scaler = self.scaler
        else:
            raise ValueError("載入的 scaler 應為 (input_scaler, output_scaler) 格式，但獲得單一 scaler。")

        # ✅ 追蹤已收集多少數據
        self.data_count = 0

    def transform_input_data(self, data):
        """
        縮放輸入數據，確保格式正確
        :param data: (1, 7) 的 numpy.ndarray
        :return: (1, 7) 的標準化數據
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"輸入數據應為 numpy.ndarray，但收到 {type(data)}")

        if data.shape != (1, 7):
            raise ValueError(f"輸入數據形狀應為 (1, 7)，但收到 {data.shape}")

        # ✅ 確保 scaler 存在，並使用 input_scaler 進行標準化
        if hasattr(self.input_scaler, "transform"):
            scaled_data = self.input_scaler.transform(data)  # 縮放數據
        else:
            raise AttributeError("input_scaler 缺少 transform 方法，請檢查 scaler 是否正確載入。")

        return scaled_data  # 直接回傳標準化數據

    def update_from_adam(self):
        """
        從 ADAMScontroller 更新數據，確保最新數據可用，並維持滑動窗口
        """
        raw_data = np.array([
            self.adam.buffer[0],  # T_GPU
            self.adam.buffer[2],  # T_CDU_in
            self.adam.buffer[4],  # T_env
            self.adam.buffer[5],  # T_air_in
            self.adam.buffer[6],  # T_air_out
            self.adam.buffer[8],  # fan_duty
            self.adam.buffer[9]   # pump_duty
        ]).reshape(1, -1)  # 轉成 (1,7) 矩陣

        processed_data = self.transform_input_data(raw_data)

        # ✅ 滑動窗口機制：將舊數據左移，最後一筆更新為最新數據
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1, :] = processed_data  # 更新最新的數據
        self.data_count += 1

    def get_window_data(self):
        """取得完整的時間窗口數據，確保數據充足"""
        with self.buffer_lock:
            if self.data_count < self.window_size:
                print(f"⏳ 數據未準備好，當前 {self.data_count}/{self.window_size}，請稍等...")
                return None
            else:
                return torch.FloatTensor(self.buffer).unsqueeze(0).to(self.device)

    def inverse_transform_predictions(self, scaled_predictions):
        """
        反標準化模型預測結果，將標準化值轉回原始溫度單位
        :param scaled_predictions: (8, 1) 或 (8,) 的標準化數據
        :return: (8,) 的原始溫度數據
        """
        if len(scaled_predictions.shape) == 1:
            scaled_predictions = scaled_predictions.reshape(-1, 1)

        # ✅ 確保 output_scaler 存在
        if hasattr(self.output_scaler, "inverse_transform"):
            original_predictions = self.output_scaler.inverse_transform(scaled_predictions)[:, 0]
        else:
            raise AttributeError("output_scaler 缺少 inverse_transform 方法，請檢查 scaler 是否正確載入。")

        return original_predictions
