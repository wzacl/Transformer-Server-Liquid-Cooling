import numpy as np
import torch
import joblib


class Data_Processor:
    def __init__(self,scaler_path, device,window_size=20,):
        """
        初始化 Data Processor，讀取 Scaler 並確保輸入/輸出縮放正確
        :param scaler_path: Scaler 檔案路徑
        :param device: PyTorch 設備（CPU / GPU）
        """
        self.scaler = joblib.load(scaler_path)
        self.device = device
        self.window_size = window_size

        # 確保 scaler 是 (input_scaler, output_scaler) 的 tuple
        if isinstance(self.scaler, tuple) and len(self.scaler) == 2:
            self.input_scaler, self.output_scaler = self.scaler
        else:
            raise ValueError("載入的 scaler 應該是包含 (input_scaler, output_scaler) 的 tuple，但獲得單一 scaler。")

    def get_current_features(self, data):
        """
        檢查輸入是否為 (20,7)，縮放後轉換為 PyTorch Tensor
        :param data: 輸入數據 (20, 7)
        :return: PyTorch Tensor, shape = (1, 20, 7)
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"輸入數據類型錯誤，應為 numpy.ndarray，但收到 {type(data)}")
        
        if data.shape != (self.window_size, 7):
            raise ValueError(f"輸入數據形狀錯誤！應該是 (20, 7)，但收到 {data.shape}")

        # 確保 scaler.transform() 可用
        if not hasattr(self.input_scaler, "transform"):
            raise AttributeError("input_scaler 缺少 transform 方法，請檢查 scaler 是否正確載入。")

        # 縮放輸入數據
        scaled_data = self.input_scaler.transform(data)

        # 轉換為 PyTorch Tensor 並移動至指定裝置 (CPU/GPU)
        return torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)

    def inverse_transform_predictions(self, scaled_predictions):
        """
        反變換縮放後的預測數據，確保輸出為原始尺度
        :param scaled_predictions: 縮放後的預測數據 (N, 1)
        :return: 原始尺度的預測數據 (N,)
        """
        if not isinstance(scaled_predictions, np.ndarray):
            raise TypeError(f"輸入數據類型錯誤，應為 numpy.ndarray，但收到 {type(scaled_predictions)}")

        # 確保數據形狀為 (N, 1)，避免維度錯誤
        if len(scaled_predictions.shape) == 1:
            scaled_predictions = scaled_predictions.reshape(-1, 1)

        # 確保 output_scaler 具有 inverse_transform 方法
        if not hasattr(self.output_scaler, "inverse_transform"):
            raise AttributeError("output_scaler 缺少 inverse_transform 方法，請檢查 scaler 是否正確載入。")

        # 反變換輸出數據
        return self.output_scaler.inverse_transform(scaled_predictions)[:, 0]



